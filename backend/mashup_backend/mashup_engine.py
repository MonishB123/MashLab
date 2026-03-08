"""
mashup_engine.py

Strict mashup engine.

Goal:
- Prefer one instrumental as the master timeline.
- Adjust only the opposite vocal, and only a little.
- If the pair is not naturally close enough, reject it instead of forcing it.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from analyze_track import TrackFeatures, analyze_mp3, as_json_dict
from audio_render import (
    export_wav,
    high_pass_filter,
    load_audio,
    normalize_peak,
    pad_or_trim_to_length,
    shift_audio,
    trim_audio,
    trim_leading_silence,
)
from compatibility import CompatibilityResult, compare_tracks


MIN_BEAT_SYNC_SCORE = 0.45
MAX_MANUAL_OFFSET_S = 1.50
ENFORCE_STRICT_REJECTION = False
ENFORCE_MIN_BEAT_SYNC = False


@dataclass
class MashupConfig:
    track_a: str
    track_b: str

    clip_duration: float = 45.0
    sr: int = 44100
    n_segment_candidates: int = 5
    start_a: Optional[float] = None
    start_b: Optional[float] = None

    mashup_mode: str = "auto"  # auto | vocals_a_inst_b | inst_a_vocals_b | full_blend
    use_stem_separation: bool = True

    gain_db_a: float = 0.0          # instrumental master default (increased from -7.0)
    gain_db_b: Optional[float] = -3.0  # vocal default (increased from -5.0)

    fade_in_s: float = 0.05
    fade_out_s: float = 0.20

    wav_out: str = "mashup.wav"
    mp3_out: Optional[str] = None
    stems_dir: Optional[str] = None
    track_a_features: Optional[TrackFeatures] = None
    track_b_features: Optional[TrackFeatures] = None
    segment_score_a: Optional[float] = None
    segment_score_b: Optional[float] = None


@dataclass
class MashupResult:
    success: bool
    wav_out: str
    mp3_out: Optional[str]

    compatibility: CompatibilityResult
    start_a: float
    start_b: float
    segment_score_a: float
    segment_score_b: float
    mashup_mode_used: str
    stem_separation_used: bool

    track_a_features: Dict[str, Any]
    track_b_features: Dict[str, Any]

    summary: str
    error: Optional[str] = None


@dataclass
class RenderDecision:
    mode: str
    comp: CompatibilityResult
    reject_reasons: List[str]



def _apply_fade(y: np.ndarray, sr: int, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    y = y.copy().astype(np.float32)
    fi = int(fade_in_s * sr)
    fo = int(fade_out_s * sr)
    if fi > 1 and fi < len(y):
        y[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 1 and fo < len(y):
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return y



def _reject_summary(reasons: List[str]) -> str:
    return "Mashup rejected: " + "; ".join(reasons) + "."



def _choose_direction(
    comp_ab: CompatibilityResult,
    comp_ba: CompatibilityResult,
    requested_mode: str,
) -> RenderDecision:
    if requested_mode == "inst_a_vocals_b":
        return RenderDecision(
            mode="inst_a_vocals_b",
            comp=comp_ab,
            reject_reasons=list(comp_ab.reject_reasons),
        )

    if requested_mode == "vocals_a_inst_b":
        return RenderDecision(
            mode="vocals_a_inst_b",
            comp=comp_ba,
            reject_reasons=list(comp_ba.reject_reasons),
        )

    # auto: pick the only safe one, or the better one if both are safe.
    safe_ab = comp_ab.layerable
    safe_ba = comp_ba.layerable

    if safe_ab and not safe_ba:
        return RenderDecision("inst_a_vocals_b", comp_ab, [])
    if safe_ba and not safe_ab:
        return RenderDecision("vocals_a_inst_b", comp_ba, [])
    if safe_ab and safe_ba:
        if comp_ab.compatibility_score >= comp_ba.compatibility_score:
            return RenderDecision("inst_a_vocals_b", comp_ab, [])
        return RenderDecision("vocals_a_inst_b", comp_ba, [])

    # Neither is safe. Return the better one, but keep its reasons.
    if comp_ab.compatibility_score >= comp_ba.compatibility_score:
        return RenderDecision("inst_a_vocals_b", comp_ab, list(comp_ab.reject_reasons))
    return RenderDecision("vocals_a_inst_b", comp_ba, list(comp_ba.reject_reasons))



def _separate_clips(
    clip_a_raw: np.ndarray,
    clip_b_raw: np.ndarray,
    sr: int,
    tmpdir: str,
):
    from source_separation import blend_stems, separate_track

    clip_a_path = os.path.join(tmpdir, "clip_a.wav")
    clip_b_path = os.path.join(tmpdir, "clip_b.wav")
    export_wav(clip_a_raw, sr, clip_a_path)
    export_wav(clip_b_raw, sr, clip_b_path)

    # Use full 4-stem separation for higher quality.
    stems_a = separate_track(clip_a_path, out_dir=os.path.join(tmpdir, "a"), sr=sr, four_stem=True)
    stems_b = separate_track(clip_b_path, out_dir=os.path.join(tmpdir, "b"), sr=sr, four_stem=True)

    vocals_a = stems_a.get("vocals")
    vocals_b = stems_b.get("vocals")

    # Blend all non-vocal stems for the instrumental.
    inst_a, _ = blend_stems(stems_a, ["drums", "bass", "other"], normalize=False)
    inst_b, _ = blend_stems(stems_b, ["drums", "bass", "other"], normalize=False)
    return vocals_a, inst_a, vocals_b, inst_b



def _window_onset_corr(a: np.ndarray, b: np.ndarray, max_lag_frames: int) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = (a - float(np.mean(a))) / (float(np.std(a)) + 1e-9)
    b = (b - float(np.mean(b))) / (float(np.std(b)) + 1e-9)
    corr = np.correlate(a, b, mode="full")
    center = len(b) - 1
    lo = max(0, center - max_lag_frames)
    hi = min(len(corr), center + max_lag_frames + 1)
    if hi <= lo:
        return 0.0
    best = float(np.max(corr[lo:hi]))
    norm = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-9
    return float(np.clip(best / norm, 0.0, 1.0))


def _pick_short_sync_window(
    path_inst: str,
    path_vocal: str,
    clip_duration: float,
    analysis_sr: int = 12000,
) -> Tuple[float, float, float, float, float, float]:
    """
    Direction-aware segment selection.

    - path_inst: the track providing the instrumental (wants HIGH energy — chorus/drop)
    - path_vocal: the track providing the vocals (wants MODERATE energy — verse/pre-chorus
      where vocals are prominent but the backing is sparser for cleaner separation)

    Returns: (start_inst_s, start_vocal_s, local_tempo_inst, local_tempo_vocal,
              energy_score_inst, energy_score_vocal)
    """
    y_inst, _ = load_audio(path_inst, sr=analysis_sr)
    y_vocal, _ = load_audio(path_vocal, sr=analysis_sr)

    hop = 512
    env_inst = librosa.onset.onset_strength(y=y_inst, sr=analysis_sr, hop_length=hop).astype(np.float32)
    env_vocal = librosa.onset.onset_strength(y=y_vocal, sr=analysis_sr, hop_length=hop).astype(np.float32)
    tempo_curve_inst = np.ravel(
        librosa.feature.tempo(onset_envelope=env_inst, sr=analysis_sr, hop_length=hop, aggregate=None)
    )
    tempo_curve_vocal = np.ravel(
        librosa.feature.tempo(onset_envelope=env_vocal, sr=analysis_sr, hop_length=hop, aggregate=None)
    )

    rms_inst = librosa.feature.rms(y=y_inst, hop_length=hop)[0]
    rms_vocal = librosa.feature.rms(y=y_vocal, hop_length=hop)[0]
    rms_inst = rms_inst / (np.max(rms_inst) + 1e-9)
    rms_vocal = rms_vocal / (np.max(rms_vocal) + 1e-9)

    frames_clip = max(8, int(clip_duration * analysis_sr / hop))
    step_frames = max(1, int(4.0 * analysis_sr / hop))

    # Skip first/last 10% of each track to avoid intros/outros
    total_frames_inst = len(env_inst)
    total_frames_vocal = len(env_vocal)
    skip_inst = int(total_frames_inst * 0.10)
    skip_vocal = int(total_frames_vocal * 0.10)
    max_start_inst = max(0, total_frames_inst - frames_clip - skip_inst)
    max_start_vocal = max(0, total_frames_vocal - frames_clip - skip_vocal)
    starts_inst = list(range(skip_inst, max_start_inst + 1, step_frames)) or [0]
    starts_vocal = list(range(skip_vocal, max_start_vocal + 1, step_frames)) or [0]

    def _local_tempo(curve: np.ndarray, i0: int, i1: int, fallback: float) -> float:
        w = curve[i0:i1]
        w = w[np.isfinite(w)]
        if len(w) == 0:
            return fallback
        return float(np.median(w))

    def _window_energy(rms: np.ndarray, i0: int, i1: int) -> float:
        w = rms[i0:min(i1, len(rms))]
        return float(np.mean(w)) if len(w) > 0 else 0.0

    global_tempo_inst = float(np.nanmedian(tempo_curve_inst)) if len(tempo_curve_inst) else 120.0
    global_tempo_vocal = float(np.nanmedian(tempo_curve_vocal)) if len(tempo_curve_vocal) else 120.0
    tempos_inst = [_local_tempo(tempo_curve_inst, s, s + frames_clip, global_tempo_inst) for s in starts_inst]
    tempos_vocal = [_local_tempo(tempo_curve_vocal, s, s + frames_clip, global_tempo_vocal) for s in starts_vocal]
    energies_inst = [_window_energy(rms_inst, s, s + frames_clip) for s in starts_inst]
    energies_vocal = [_window_energy(rms_vocal, s, s + frames_clip) for s in starts_vocal]

    # Instrumental track: only keep high-energy windows (>50% of max).
    max_e_inst = max(energies_inst) if energies_inst else 1.0
    valid_inst = [i for i, e in enumerate(energies_inst) if e >= 0.50 * max_e_inst]
    if not valid_inst:
        valid_inst = list(range(len(starts_inst)))

    # Vocal track: keep moderate-energy windows (30-80% of max).
    # This avoids both dead quiet sections AND the loudest chorus where
    # the dense instrumental backing makes separation harder and causes clash.
    max_e_vocal = max(energies_vocal) if energies_vocal else 1.0
    valid_vocal = [i for i, e in enumerate(energies_vocal)
                   if 0.30 * max_e_vocal <= e <= 0.80 * max_e_vocal]
    if not valid_vocal:
        # Fall back: prefer anything above 25%
        valid_vocal = [i for i, e in enumerate(energies_vocal) if e >= 0.25 * max_e_vocal]
    if not valid_vocal:
        valid_vocal = list(range(len(starts_vocal)))

    best_score = -1.0
    best = (0.0, 0.0, global_tempo_inst, global_tempo_vocal, 0.0, 0.0)
    lag_cap = int(1.5 * analysis_sr / hop)
    tempos_vocal_arr = np.array([tempos_vocal[j] for j in valid_vocal], dtype=np.float32)
    valid_vocal_arr = np.array(valid_vocal)

    for i in valid_inst:
        sa = starts_inst[i]
        ta = max(1e-6, float(tempos_inst[i]))
        ea = float(energies_inst[i])
        tempo_diffs = np.abs(tempos_vocal_arr - ta) / ta
        top_vocal_indices = np.argsort(tempo_diffs)[: min(8, len(valid_vocal))]
        win_inst = env_inst[sa : sa + frames_clip]
        for bi in top_vocal_indices:
            j = int(valid_vocal_arr[bi])
            sb = starts_vocal[j]
            tb = max(1e-6, float(tempos_vocal[j]))
            eb = float(energies_vocal[j])
            tempo_score = float(max(0.0, 1.0 - min(1.0, abs(tb - ta) / ta)))
            beat_score = _window_onset_corr(win_inst, env_vocal[sb : sb + frames_clip], lag_cap)

            # Vocal energy preference: peaks around 0.55 of max, drops off
            # at extremes. This favors verse/pre-chorus sections where vocals
            # are clear but the instrumental backing is sparser.
            vocal_energy_pref = max(0.0, 1.0 - abs(eb - 0.55) / 0.45)

            # Instrumental: raw energy (higher = better, lands on chorus/drop)
            # Vocal: moderate energy preference (cleaner separation)
            # Tempo: must match closely
            # Beat sync: rhythmic compatibility
            combined = (0.30 * ea
                        + 0.20 * vocal_energy_pref
                        + 0.30 * tempo_score
                        + 0.20 * beat_score)
            if combined > best_score:
                best_score = combined
                best = (
                    float(sa * hop / analysis_sr),
                    float(sb * hop / analysis_sr),
                    ta,
                    tb,
                    ea,
                    eb,
                )

    # --- Energy peak centering (instrumental only) ---
    # Shift instrumental start so the RMS energy peak lands ~35-40% into the clip.
    target_peak_frac = 0.37
    late_threshold = 0.60

    best_start_inst_s, best_start_vocal_s, bta, btb, bea, beb = best

    start_frame = int(best_start_inst_s * analysis_sr / hop)
    end_frame = start_frame + frames_clip
    rms_window = rms_inst[start_frame : min(end_frame, len(rms_inst))]
    if len(rms_window) >= 4:
        peak_pos = int(np.argmax(rms_window))
        peak_frac = peak_pos / len(rms_window)
        if peak_frac > late_threshold:
            shift_frames = int((peak_frac - target_peak_frac) * len(rms_window))
            new_start_frame = start_frame + shift_frames
            max_start_frame = int(len(y_inst) / analysis_sr * analysis_sr / hop) - frames_clip
            new_start_frame = max(0, min(new_start_frame, max(0, max_start_frame)))
            best_start_inst_s = float(new_start_frame * hop / analysis_sr)

    best = (best_start_inst_s, best_start_vocal_s, bta, btb, bea, beb)
    return best


def _eq_vocal(vocal: np.ndarray, sr: int) -> np.ndarray:
    """
    Vocals EQ:
    - high-pass below 120 Hz
    - slight presence boost around 3-5 kHz
    """
    from scipy.signal import butter, filtfilt

    y = high_pass_filter(vocal, sr, cutoff=120.0)
    nyq = 0.5 * sr
    lo = max(10.0, 3000.0) / max(nyq, 1e-9)
    hi = min(0.99, 5000.0 / max(nyq, 1e-9))
    if lo < hi:
        b, a = butter(2, [lo, hi], btype="band")
        band = filtfilt(b, a, y).astype(np.float32)
        y = (y + 0.12 * band).astype(np.float32)
    return y.astype(np.float32)


def _eq_instrumental(inst: np.ndarray, sr: int) -> np.ndarray:
    """
    Instrumental EQ:
    - slight cut 300-500 Hz
    - slight cut 2-4 kHz
    """
    from scipy.signal import butter, filtfilt

    y = inst.astype(np.float32)
    nyq = 0.5 * sr
    for lo_hz, hi_hz, amount in ((300.0, 500.0, 0.10), (2000.0, 4000.0, 0.10)):
        lo = max(10.0, lo_hz) / max(nyq, 1e-9)
        hi = min(0.99, hi_hz / max(nyq, 1e-9))
        if lo < hi:
            b, a = butter(2, [lo, hi], btype="band")
            band = filtfilt(b, a, y).astype(np.float32)
            y = (y - amount * band).astype(np.float32)
    return y.astype(np.float32)


def _match_rms_to_target(y: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-9)
    if rms <= 0.0 or target_rms <= 0.0:
        return y.astype(np.float32)
    gain = target_rms / rms
    return (y * gain).astype(np.float32)


def _mix_equal_db(instrumental: np.ndarray, vocal: np.ndarray) -> np.ndarray:
    """
    Mix vocal over instrumental at equal loudness.

    Uses the vocal's active (non-silent) RMS instead of overall RMS so that
    natural gaps between vocal phrases don't drag down the measured level.
    Both signals are matched to equal RMS, then normalized.
    """
    target_len = max(len(instrumental), len(vocal))
    inst = pad_or_trim_to_length(instrumental, target_len)
    voc = pad_or_trim_to_length(vocal, target_len)

    inst_rms = float(np.sqrt(np.mean(np.square(inst))) + 1e-9)

    # Compute vocal RMS from non-silent portions only (above -35dB of peak).
    voc_peak = float(np.max(np.abs(voc)) + 1e-9)
    silence_thresh = voc_peak * 0.018  # ~-35dB below peak
    active_mask = np.abs(voc) > silence_thresh
    if np.sum(active_mask) > 0:
        voc_active_rms = float(np.sqrt(np.mean(np.square(voc[active_mask]))) + 1e-9)
    else:
        voc_active_rms = float(np.sqrt(np.mean(np.square(voc))) + 1e-9)

    # Match vocal's active RMS to instrumental RMS (equal loudness).
    voc_gain = inst_rms / voc_active_rms
    voc = (voc * voc_gain).astype(np.float32)

    mixed = inst + voc
    return normalize_peak(mixed, peak=0.92)


def _onset_offset_and_score(
    master: np.ndarray,
    reference: np.ndarray,
    sr: int,
    max_offset_s: float = MAX_MANUAL_OFFSET_S,
) -> Tuple[int, float]:
    """
    Find a global offset (in samples) that maximizes beat/onset overlap.
    No time-stretching and no pitch shifting.

    Uses onset cross-correlation to find a rough offset, then snaps to the
    nearest beat-grid-aligned position using actual detected beat positions.
    This prevents half-beat-off alignment where onset correlation is high
    but beats land on the wrong phase.
    """
    hop = 512
    env_master = librosa.onset.onset_strength(y=master, sr=sr, hop_length=hop)
    env_ref = librosa.onset.onset_strength(y=reference, sr=sr, hop_length=hop)

    if len(env_master) == 0 or len(env_ref) == 0:
        return 0, 0.0

    env_master = env_master.astype(np.float32)
    env_ref = env_ref.astype(np.float32)
    env_master = (env_master - float(np.mean(env_master))) / (float(np.std(env_master)) + 1e-9)
    env_ref = (env_ref - float(np.mean(env_ref))) / (float(np.std(env_ref)) + 1e-9)

    corr = np.correlate(env_master, env_ref, mode="full")
    center = len(env_ref) - 1
    max_lag = min(int(max_offset_s * sr / hop), min(len(env_master), len(env_ref)) - 1)
    lo = max(0, center - max_lag)
    hi = min(len(corr), center + max_lag + 1)
    if hi <= lo:
        return 0, 0.0

    window = corr[lo:hi]
    best_lag = int(np.argmax(window)) + lo - center

    # --- Beat-grid snap ---
    # Detect beat positions in both signals and compute the phase offset
    # needed to align the two beat grids.  Then generate candidate lags
    # that are beat-grid-aligned (the onset-corr lag rounded to the nearest
    # position where beat phases match).
    _, beats_master = librosa.beat.beat_track(y=master, sr=sr, hop_length=hop, units="time")
    _, beats_ref = librosa.beat.beat_track(y=reference, sr=sr, hop_length=hop, units="time")

    tempo_arr = librosa.feature.tempo(onset_envelope=env_master, sr=sr, hop_length=hop, aggregate=np.median)
    tempo_bpm = float(np.ravel(tempo_arr)[0]) if np.size(tempo_arr) else 120.0
    beat_period_s = 60.0 / max(tempo_bpm, 1e-6)
    beat_frames = max(1, int(round(beat_period_s * sr / hop)))

    # Compute beat-grid phase from each signal's detected beats.
    if len(beats_master) >= 3 and len(beats_ref) >= 3:
        phase_master = float(np.median(beats_master[:10] % beat_period_s))
        phase_ref = float(np.median(beats_ref[:10] % beat_period_s))
        phase_diff_s = phase_master - phase_ref  # offset to align ref to master

        # Generate beat-grid-aligned lags near the onset-corr best lag.
        # These are offsets where ref beats would land on master beats.
        corr_offset_s = best_lag * hop / sr
        n_beats_offset = round((corr_offset_s - phase_diff_s) / beat_period_s)
        grid_candidates_s = [
            n_beats_offset * beat_period_s + phase_diff_s,
            (n_beats_offset + 1) * beat_period_s + phase_diff_s,
            (n_beats_offset - 1) * beat_period_s + phase_diff_s,
        ]
        grid_candidate_frames = [
            int(round(s * sr / hop)) for s in grid_candidates_s
        ]
    else:
        grid_candidate_frames = []

    # Build final candidate list: onset-corr best + beat-grid-snapped +
    # traditional ±beat/half-beat offsets as fallback.
    candidate_lags = [
        best_lag,
        best_lag - beat_frames,
        best_lag + beat_frames,
        best_lag - (beat_frames // 2),
        best_lag + (beat_frames // 2),
    ] + grid_candidate_frames

    def _corr_at_lag(lag: int) -> float:
        idx = center + lag
        if idx < 0 or idx >= len(corr):
            return -1e9
        return float(corr[idx])

    def _beat_alignment_score(lag: int) -> float:
        """Score how well the beat grids align at this lag."""
        if len(beats_master) < 3 or len(beats_ref) < 3:
            return 0.0
        offset_s = lag * hop / sr
        # Compute the phase relationship directly: are shifted ref beats
        # landing on the same phase as master beats?
        master_phase = float(beats_master[0]) % beat_period_s
        shifted_ref_beats = beats_ref + offset_s
        half_beat = beat_period_s / 2.0
        total_err = 0.0
        count = 0
        for rb in shifted_ref_beats:
            if rb < 0:
                continue
            # How far is this beat from the master beat grid phase?
            phase_err = (rb - master_phase) % beat_period_s
            if phase_err > half_beat:
                phase_err = beat_period_s - phase_err
            total_err += phase_err / half_beat  # 0 = on grid, 1 = half beat off
            count += 1
        if count == 0:
            return 0.0
        return 1.0 - (total_err / count)  # 1.0 = perfect, 0.0 = half beat off

    # Score candidates by weighted combination of correlation and beat alignment.
    # Beat alignment gets 60% weight to prevent half-beat-off selection.
    def _combined_score(lag: int) -> float:
        c = _corr_at_lag(lag)
        norm = (float(np.linalg.norm(env_master)) * float(np.linalg.norm(env_ref))) + 1e-9
        corr_score = max(0.0, c / norm)
        beat_score = _beat_alignment_score(lag)
        return 0.40 * corr_score + 0.60 * beat_score

    best_lag = max(candidate_lags, key=_combined_score)
    norm = (float(np.linalg.norm(env_master)) * float(np.linalg.norm(env_ref))) + 1e-9
    score = _corr_at_lag(best_lag) / norm
    offset_samples = int(best_lag * hop)
    return offset_samples, float(np.clip(score, 0.0, 1.0))


def _find_first_downbeat(y: np.ndarray, sr: int) -> int:
    """Find the sample position of the first strong beat in the audio."""
    hop = 512
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="samples")
    if len(beats) == 0:
        return 0
    return int(beats[0])


def _align_vocal_to_master(
    instrumental_master: np.ndarray,
    rhythm_reference: np.ndarray,
    vocal: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, float]:
    """
    Aligns the vocal stem to the instrumental using onset cross-correlation.
    Falls back to downbeat alignment when cross-correlation confidence is low.
    Returns (aligned_vocal, beat_lock_score).
    """
    beat_offset, beat_sync_score = _onset_offset_and_score(
        instrumental_master, rhythm_reference, sr, max_offset_s=MAX_MANUAL_OFFSET_S
    )

    # Clean the vocal stem by removing leading dead air from separator output.
    vocal, vocal_silence_offset = trim_leading_silence(vocal, sr, top_db=35.0)

    if beat_sync_score >= MIN_BEAT_SYNC_SCORE:
        # Good cross-correlation — use the computed offset.
        net_offset = beat_offset - vocal_silence_offset
    else:
        # Poor cross-correlation — align first downbeats instead.
        # This is more robust for tracks with different beat structures.
        db_master = _find_first_downbeat(instrumental_master, sr)
        db_vocal = _find_first_downbeat(vocal, sr)
        net_offset = db_master - db_vocal

    aligned_vocal = shift_audio(vocal, net_offset, target_len=len(instrumental_master))
    return aligned_vocal.astype(np.float32), float(beat_sync_score)



def _compute_pitch_shift(inst_key_pc: int, inst_mode: int,
                         vocal_key_pc: int, vocal_mode: int) -> int:
    """
    Pitch shifting is disabled. Key detection (Krumhansl-Schmuckler) is too
    unreliable for automatic correction — wrong shifts sound far worse than
    a natural key mismatch, especially on high-pitched vocals.
    """
    return 0


def _render_with_stems(
    path_a: str,
    path_b: str,
    mode: str,
    start_a: float,
    start_b: float,
    clip_duration: float,
    sr: int,
    stems_dir: Optional[str],
    config: MashupConfig,
    pitch_shift_semitones: int = 0,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    tmpdir = stems_dir or tempfile.mkdtemp(prefix="mashup_stems_")
    os.makedirs(tmpdir, exist_ok=True)
    if stems_dir:
        os.makedirs(stems_dir, exist_ok=True)

    y_a, _ = load_audio(path_a, sr=sr)
    y_b, _ = load_audio(path_b, sr=sr)
    clip_a_raw = trim_audio(y_a, sr, start_a, clip_duration)
    clip_b_raw = trim_audio(y_b, sr, start_b, clip_duration)

    vocals_a, inst_a, vocals_b, inst_b = _separate_clips(clip_a_raw, clip_b_raw, sr, tmpdir)

    if stems_dir:
        export_wav(vocals_a, sr, os.path.join(stems_dir, "track_a_vocals.wav"))
        export_wav(inst_a, sr, os.path.join(stems_dir, "track_a_instrumental.wav"))
        export_wav(vocals_b, sr, os.path.join(stems_dir, "track_b_vocals.wav"))
        export_wav(inst_b, sr, os.path.join(stems_dir, "track_b_instrumental.wav"))

    if mode == "vocals_a_inst_b":
        instrumental = inst_b
        rhythm_reference = clip_a_raw
        vocal = vocals_a
    else:
        instrumental = inst_a
        rhythm_reference = clip_b_raw
        vocal = vocals_b

    # Pitch-shift vocal to match the instrumental's key.
    if pitch_shift_semitones != 0:
        vocal = librosa.effects.pitch_shift(
            vocal, sr=sr, n_steps=float(pitch_shift_semitones),
        ).astype(np.float32)

    aligned_vocal, beat_sync_score = _align_vocal_to_master(
        instrumental_master=instrumental,
        rhythm_reference=rhythm_reference,
        vocal=vocal,
        sr=sr,
    )

    # Apply separation EQ to avoid masking/muffling.
    inst_eq = _eq_instrumental(instrumental, sr)
    vocal_eq = _eq_vocal(aligned_vocal, sr)

    # Match vocal/instrumental loudness (equal dB proxy via RMS) before summing.
    mixed = _mix_equal_db(inst_eq, vocal_eq)
    # Return rhythm_reference (original full track of vocal side) for sync
    # analysis — separated vocal stems lack rhythmic onsets.
    # Also return aligned_vocal so trim can prefer regions with vocal energy.
    return mixed, beat_sync_score, instrumental, rhythm_reference, aligned_vocal


def _trim_to_best_sync(
    instrumental: np.ndarray,
    rhythm_ref: np.ndarray,
    mixed: np.ndarray,
    sr: int,
    aligned_vocal: Optional[np.ndarray] = None,
    min_duration: float = 15.0,
    fade_in_s: float = 0.05,
    fade_out_s: float = 0.20,
) -> np.ndarray:
    """
    Scan the rendered audio for the best-synced sub-region and trim to it.

    Scores each window by a combination of:
    - Onset cross-correlation (beat sync)
    - Vocal energy (prefer regions where vocals are actually present)

    This prevents trimming to purely instrumental sections where the vocal
    has dropped off.
    """
    total_s = len(mixed) / sr
    if total_s <= min_duration + 1.0:
        return mixed

    hop = 512
    env_inst = librosa.onset.onset_strength(y=instrumental, sr=sr, hop_length=hop).astype(np.float32)
    env_ref = librosa.onset.onset_strength(y=rhythm_ref, sr=sr, hop_length=hop).astype(np.float32)

    n_frames = min(len(env_inst), len(env_ref))
    if n_frames == 0:
        return mixed

    # Compute per-frame vocal energy if vocal stem is available.
    vocal_rms = None
    if aligned_vocal is not None and len(aligned_vocal) > 0:
        voc_rms_raw = librosa.feature.rms(y=aligned_vocal, hop_length=hop)[0]
        voc_max = float(np.max(voc_rms_raw)) + 1e-9
        vocal_rms = voc_rms_raw / voc_max  # normalized 0-1

    analysis_win_s = 5.0
    win_frames = int(analysis_win_s * sr / hop)
    step_frames = max(1, int(1.0 * sr / hop))
    max_lag = int(0.15 * sr / hop)

    if win_frames >= n_frames:
        return mixed

    positions = list(range(0, n_frames - win_frames + 1, step_frames))
    scores = []
    for pos in positions:
        sync_sc = _window_onset_corr(
            env_inst[pos : pos + win_frames],
            env_ref[pos : pos + win_frames],
            max_lag,
        )
        # Vocal energy score for this window (0-1).
        voc_sc = 0.5  # neutral default if no vocal stem
        if vocal_rms is not None:
            voc_window = vocal_rms[pos : min(pos + win_frames, len(vocal_rms))]
            if len(voc_window) > 0:
                voc_sc = float(np.mean(voc_window))

        # Combined: 50% sync + 50% vocal presence.
        # This ensures we never trim to a region with no vocals.
        combined = 0.50 * sync_sc + 0.50 * voc_sc
        scores.append(combined)

    if not scores:
        return mixed

    scores_arr = np.array(scores, dtype=np.float32)
    peak_score = float(np.max(scores_arr))
    if peak_score < 1e-6:
        return mixed

    min_windows = max(1, int(min_duration))
    if len(scores_arr) <= min_windows:
        return mixed

    cumsum = np.cumsum(np.insert(scores_arr, 0, 0.0))
    rolling_avg = (cumsum[min_windows:] - cumsum[:-min_windows]) / min_windows
    best_run_start = int(np.argmax(rolling_avg))
    best_run_avg = float(rolling_avg[best_run_start])
    full_avg = float(np.mean(scores_arr))

    if best_run_avg <= full_avg * 1.02:
        return mixed

    start_sample = positions[best_run_start] * hop
    end_pos_idx = min(best_run_start + min_windows - 1, len(positions) - 1)
    end_sample = min(len(mixed), (positions[end_pos_idx] + win_frames) * hop)

    region_duration = (end_sample - start_sample) / sr
    if region_duration < min_duration:
        return mixed

    if region_duration >= total_s * 0.85:
        return mixed

    trimmed = mixed[start_sample:end_sample].copy()
    trimmed = _apply_fade(trimmed, sr, fade_in_s, fade_out_s)
    return trimmed


class MashupEngine:
    def run(self, config: MashupConfig) -> MashupResult:
        try:
            return self._run(config)
        except Exception as exc:
            import traceback

            return MashupResult(
                success=False,
                wav_out=config.wav_out,
                mp3_out=config.mp3_out,
                compatibility=CompatibilityResult(
                    compatibility_score=0.0,
                    grade="F",
                    embedding_similarity=0.0,
                    camelot_key_score=0.0,
                    tempo_similarity=0.0,
                    energy_similarity=0.0,
                    tempo_score=0.0,
                    key_score=0.0,
                    energy_score=0.0,
                    loudness_score=0.0,
                    timbre_score=0.0,
                    spectral_contrast_score=0.0,
                    tonnetz_score=0.0,
                    danceability_match_score=0.0,
                    tempo_ratio_used=1.0,
                    stretch_factor_b=1.0,
                    stretch_pct_b=0.0,
                    pitch_shift_b=0,
                    gain_db_b=0.0,
                    mashup_type="inst_a_vocals_b",
                    layerable=False,
                    reject_reasons=[str(exc)],
                    summary=str(exc),
                ),
                start_a=0.0,
                start_b=0.0,
                segment_score_a=0.0,
                segment_score_b=0.0,
                mashup_mode_used="",
                stem_separation_used=False,
                track_a_features={},
                track_b_features={},
                summary="",
                error=f"{exc}\n{traceback.format_exc()}",
            )

    def _run(self, config: MashupConfig) -> MashupResult:
        clip_duration = float(np.clip(config.clip_duration, 15.0, 60.0))
        feats_a = config.track_a_features or analyze_mp3(
            config.track_a,
            target_sr=16000,
            analysis_window_s=60.0,
            use_middle_window=True,
            fast_mode=True,
        )
        feats_b = config.track_b_features or analyze_mp3(
            config.track_b,
            target_sr=16000,
            analysis_window_s=60.0,
            use_middle_window=True,
            fast_mode=True,
        )

        comp_ab = compare_tracks(feats_a, feats_b)
        comp_ba = compare_tracks(feats_b, feats_a)

        # Choose direction BEFORE segment selection so we can pick
        # high-energy sections for the instrumental and moderate-energy
        # (vocal-forward) sections for the vocal track.
        decision = _choose_direction(comp_ab, comp_ba, config.mashup_mode)

        if config.start_a is not None and config.start_b is not None:
            start_a = float(config.start_a)
            start_b = float(config.start_b)
            seg_score_a = 1.0
            seg_score_b = 1.0
        else:
            # Pass tracks in (instrumental, vocal) order for direction-aware selection.
            if decision.mode == "inst_a_vocals_b":
                s_inst, s_vocal, _, _, seg_inst, seg_vocal = _pick_short_sync_window(
                    config.track_a, config.track_b,
                    clip_duration=clip_duration, analysis_sr=12000,
                )
                start_a, start_b = s_inst, s_vocal
                seg_score_a, seg_score_b = seg_inst, seg_vocal
            else:
                # vocals_a_inst_b: B is instrumental, A is vocal
                s_inst, s_vocal, _, _, seg_inst, seg_vocal = _pick_short_sync_window(
                    config.track_b, config.track_a,
                    clip_duration=clip_duration, analysis_sr=12000,
                )
                start_b, start_a = s_inst, s_vocal
                seg_score_b, seg_score_a = seg_inst, seg_vocal
        reject_reasons = list(decision.reject_reasons)

        if not config.use_stem_separation:
            reject_reasons.append("stem separation must be enabled")

        if reject_reasons and ENFORCE_STRICT_REJECTION:
            raise ValueError(_reject_summary(reject_reasons))

        warnings: List[str] = []
        if reject_reasons:
            warnings.append("Ignored compatibility checks: " + "; ".join(reject_reasons))

        # Compute pitch shift to align vocals to instrumental key.
        if decision.mode == "vocals_a_inst_b":
            # Instrumental from B, vocals from A → shift A vocals to B's key
            pitch_shift = _compute_pitch_shift(
                feats_b.key_pc_guess, feats_b.mode,
                feats_a.key_pc_guess, feats_a.mode,
            )
        else:
            # Instrumental from A, vocals from B → shift B vocals to A's key
            pitch_shift = _compute_pitch_shift(
                feats_a.key_pc_guess, feats_a.mode,
                feats_b.key_pc_guess, feats_b.mode,
            )

        mixed, beat_sync_score, stem_inst, stem_vocal, stem_aligned_voc = _render_with_stems(
            config.track_a,
            config.track_b,
            mode=decision.mode,
            start_a=start_a,
            start_b=start_b,
            clip_duration=clip_duration,
            sr=config.sr,
            stems_dir=config.stems_dir,
            config=config,
            pitch_shift_semitones=pitch_shift,
        )

        # Trim to the best-synced region (removes misaligned edges).
        # Pass aligned vocal so trim prefers regions with vocal presence.
        mixed = _trim_to_best_sync(
            stem_inst, stem_vocal, mixed, config.sr,
            aligned_vocal=stem_aligned_voc,
            min_duration=15.0,
            fade_in_s=config.fade_in_s,
            fade_out_s=config.fade_out_s,
        )

        if beat_sync_score < MIN_BEAT_SYNC_SCORE:
            msg = (
                f"beat grids do not lock cleanly enough "
                f"(score {beat_sync_score:.2f}, target {MIN_BEAT_SYNC_SCORE:.2f})"
            )
            if ENFORCE_MIN_BEAT_SYNC:
                raise ValueError(_reject_summary([msg]))
            warnings.append(msg)

        mixed = _apply_fade(mixed, config.sr, config.fade_in_s, config.fade_out_s)
        mixed = normalize_peak(mixed, peak=0.92)

        os.makedirs(os.path.dirname(os.path.abspath(config.wav_out)), exist_ok=True)
        export_wav(mixed, config.sr, config.wav_out)

        mp3_out = None
        if config.mp3_out:
            from audio_render import export_mp3_via_pydub

            export_mp3_via_pydub(config.wav_out, config.mp3_out)
            mp3_out = config.mp3_out

        summary = (
            f"Accepted. Mode: {decision.mode}. "
            f"Sync: short-window tempo+onset match. "
            f"Vocal pitch shift: {pitch_shift:+d} semitones. "
            f"Beat lock score: {beat_sync_score:.2f}. "
            f"Clip duration: {clip_duration:.1f}s."
        )
        if warnings:
            summary += " " + " ".join(warnings)

        return MashupResult(
            success=True,
            wav_out=config.wav_out,
            mp3_out=mp3_out,
            compatibility=decision.comp,
            start_a=start_a,
            start_b=start_b,
            segment_score_a=seg_score_a,
            segment_score_b=seg_score_b,
            mashup_mode_used=decision.mode,
            stem_separation_used=True,
            track_a_features=as_json_dict(feats_a),
            track_b_features=as_json_dict(feats_b),
            summary=summary,
        )
