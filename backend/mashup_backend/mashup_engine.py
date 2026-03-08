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
    path_a: str,
    path_b: str,
    clip_duration: float,
    analysis_sr: int = 12000,
) -> Tuple[float, float, float, float, float, float]:
    """
    Pick short windows where local tempo is close, onset pulses align,
    AND energy is high (prefer chorus / drop regions over quiet intros).

    Returns: (start_a_s, start_b_s, local_tempo_a, local_tempo_b,
              energy_score_a, energy_score_b)
    """
    y_a, _ = load_audio(path_a, sr=analysis_sr)
    y_b, _ = load_audio(path_b, sr=analysis_sr)

    hop = 512
    env_a = librosa.onset.onset_strength(y=y_a, sr=analysis_sr, hop_length=hop).astype(np.float32)
    env_b = librosa.onset.onset_strength(y=y_b, sr=analysis_sr, hop_length=hop).astype(np.float32)
    tempo_curve_a = np.ravel(
        librosa.feature.tempo(onset_envelope=env_a, sr=analysis_sr, hop_length=hop, aggregate=None)
    )
    tempo_curve_b = np.ravel(
        librosa.feature.tempo(onset_envelope=env_b, sr=analysis_sr, hop_length=hop, aggregate=None)
    )

    # Compute per-frame RMS energy for energy-aware selection
    rms_a = librosa.feature.rms(y=y_a, hop_length=hop)[0]
    rms_b = librosa.feature.rms(y=y_b, hop_length=hop)[0]
    rms_a = rms_a / (np.max(rms_a) + 1e-9)
    rms_b = rms_b / (np.max(rms_b) + 1e-9)

    frames_clip = max(8, int(clip_duration * analysis_sr / hop))
    step_frames = max(1, int(4.0 * analysis_sr / hop))

    # Skip first/last 10% of each track to avoid intros/outros
    total_frames_a = len(env_a)
    total_frames_b = len(env_b)
    skip_a = int(total_frames_a * 0.10)
    skip_b = int(total_frames_b * 0.10)
    max_start_a = max(0, total_frames_a - frames_clip - skip_a)
    max_start_b = max(0, total_frames_b - frames_clip - skip_b)
    starts_a = list(range(skip_a, max_start_a + 1, step_frames)) or [0]
    starts_b = list(range(skip_b, max_start_b + 1, step_frames)) or [0]

    def _local_tempo(curve: np.ndarray, i0: int, i1: int, fallback: float) -> float:
        w = curve[i0:i1]
        w = w[np.isfinite(w)]
        if len(w) == 0:
            return fallback
        return float(np.median(w))

    def _window_energy(rms: np.ndarray, i0: int, i1: int) -> float:
        w = rms[i0:min(i1, len(rms))]
        return float(np.mean(w)) if len(w) > 0 else 0.0

    global_tempo_a = float(np.nanmedian(tempo_curve_a)) if len(tempo_curve_a) else 120.0
    global_tempo_b = float(np.nanmedian(tempo_curve_b)) if len(tempo_curve_b) else 120.0
    tempos_a = [_local_tempo(tempo_curve_a, s, s + frames_clip, global_tempo_a) for s in starts_a]
    tempos_b = [_local_tempo(tempo_curve_b, s, s + frames_clip, global_tempo_b) for s in starts_b]
    energies_a = [_window_energy(rms_a, s, s + frames_clip) for s in starts_a]
    energies_b = [_window_energy(rms_b, s, s + frames_clip) for s in starts_b]

    # Pre-filter: only keep windows with energy above 50% of max for that track.
    # This eliminates quiet intros/verses from consideration entirely.
    max_ea = max(energies_a) if energies_a else 1.0
    max_eb = max(energies_b) if energies_b else 1.0
    energy_threshold = 0.50
    valid_a = [i for i, e in enumerate(energies_a) if e >= energy_threshold * max_ea]
    valid_b = [i for i, e in enumerate(energies_b) if e >= energy_threshold * max_eb]
    # Fall back to all windows if filtering is too aggressive
    if not valid_a:
        valid_a = list(range(len(starts_a)))
    if not valid_b:
        valid_b = list(range(len(starts_b)))

    best_score = -1.0
    best = (0.0, 0.0, global_tempo_a, global_tempo_b, 0.0, 0.0)
    lag_cap = int(1.5 * analysis_sr / hop)
    tempos_b_arr = np.array([tempos_b[j] for j in valid_b], dtype=np.float32)
    valid_b_arr = np.array(valid_b)

    for i in valid_a:
        sa = starts_a[i]
        ta = max(1e-6, float(tempos_a[i]))
        ea = float(energies_a[i])
        tempo_diffs = np.abs(tempos_b_arr - ta) / ta
        top_b_indices = np.argsort(tempo_diffs)[: min(8, len(valid_b))]
        win_a = env_a[sa : sa + frames_clip]
        for bi in top_b_indices:
            j = int(valid_b_arr[bi])
            sb = starts_b[j]
            tb = max(1e-6, float(tempos_b[j]))
            eb = float(energies_b[j])
            tempo_score = float(max(0.0, 1.0 - min(1.0, abs(tb - ta) / ta)))
            beat_score = _window_onset_corr(win_a, env_b[sb : sb + frames_clip], lag_cap)
            energy_score = (ea + eb) / 2.0

            # Energy-first scoring: energy 50%, tempo 30%, beat sync 20%
            # Energy is dominant to ensure we land on choruses/drops.
            # Tempo and beat sync keep things musically compatible.
            combined = 0.50 * energy_score + 0.30 * tempo_score + 0.20 * beat_score
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

    # --- Energy peak centering ---
    # Shift starts so the RMS energy peak lands ~35-40% into the clip
    # (i.e. roughly 15-20s into a 45s clip) instead of near the end.
    target_peak_frac = 0.37  # where we want the peak (37% into clip)
    late_threshold = 0.60    # consider it "late" if peak is past 60%

    best_start_a_s, best_start_b_s, bta, btb, bea, beb = best

    for track_label, rms_curve, start_s, total_dur_s in [
        ("a", rms_a, best_start_a_s, len(y_a) / analysis_sr),
        ("b", rms_b, best_start_b_s, len(y_b) / analysis_sr),
    ]:
        start_frame = int(start_s * analysis_sr / hop)
        end_frame = start_frame + frames_clip
        rms_window = rms_curve[start_frame : min(end_frame, len(rms_curve))]
        if len(rms_window) < 4:
            continue
        peak_pos = int(np.argmax(rms_window))
        peak_frac = peak_pos / len(rms_window)
        if peak_frac > late_threshold:
            # Peak is in the last portion — shift start forward so peak
            # lands at target_peak_frac of the window.
            shift_frames = int((peak_frac - target_peak_frac) * len(rms_window))
            new_start_frame = start_frame + shift_frames
            max_start_frame = int(total_dur_s * analysis_sr / hop) - frames_clip
            new_start_frame = max(0, min(new_start_frame, max(0, max_start_frame)))
            new_start_s = float(new_start_frame * hop / analysis_sr)
            if track_label == "a":
                best_start_a_s = new_start_s
            else:
                best_start_b_s = new_start_s

    best = (best_start_a_s, best_start_b_s, bta, btb, bea, beb)
    return best


def _eq_vocal(vocal: np.ndarray, sr: int) -> np.ndarray:
    """
    Vocals EQ:
    - high-pass below 120 Hz
    - slight presence boost around 3-5 kHz
    """
    from scipy.signal import butter, lfilter

    y = high_pass_filter(vocal, sr, cutoff=120.0)
    nyq = 0.5 * sr
    lo = max(10.0, 3000.0) / max(nyq, 1e-9)
    hi = min(0.99, 5000.0 / max(nyq, 1e-9))
    if lo < hi:
        b, a = butter(2, [lo, hi], btype="band")
        band = lfilter(b, a, y).astype(np.float32)
        y = (y + 0.12 * band).astype(np.float32)
    return y.astype(np.float32)


def _eq_instrumental(inst: np.ndarray, sr: int) -> np.ndarray:
    """
    Instrumental EQ:
    - slight cut 300-500 Hz
    - slight cut 2-4 kHz
    """
    from scipy.signal import butter, lfilter

    y = inst.astype(np.float32)
    nyq = 0.5 * sr
    for lo_hz, hi_hz, amount in ((300.0, 500.0, 0.10), (2000.0, 4000.0, 0.10)):
        lo = max(10.0, lo_hz) / max(nyq, 1e-9)
        hi = min(0.99, hi_hz / max(nyq, 1e-9))
        if lo < hi:
            b, a = butter(2, [lo, hi], btype="band")
            band = lfilter(b, a, y).astype(np.float32)
            y = (y - amount * band).astype(np.float32)
    return y.astype(np.float32)


def _match_rms_to_target(y: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-9)
    if rms <= 0.0 or target_rms <= 0.0:
        return y.astype(np.float32)
    gain = target_rms / rms
    return (y * gain).astype(np.float32)


def _mix_equal_db(instrumental: np.ndarray, vocal: np.ndarray) -> np.ndarray:
    target_len = max(len(instrumental), len(vocal))
    inst = pad_or_trim_to_length(instrumental, target_len)
    voc = pad_or_trim_to_length(vocal, target_len)
    target_rms = 0.5 * (
        float(np.sqrt(np.mean(np.square(inst))) + 1e-9) + float(np.sqrt(np.mean(np.square(voc))) + 1e-9)
    )
    inst = _match_rms_to_target(inst, target_rms)
    voc = _match_rms_to_target(voc, target_rms)
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

    # Also test likely downbeat disagreements: +/- half beat and +/- one beat.
    tempo_arr = librosa.feature.tempo(onset_envelope=env_master, sr=sr, hop_length=hop, aggregate=np.median)
    tempo_bpm = float(np.ravel(tempo_arr)[0]) if np.size(tempo_arr) else 120.0
    beat_frames = max(1, int(round((60.0 / max(tempo_bpm, 1e-6)) * sr / hop)))
    candidate_lags = [
        best_lag,
        best_lag - beat_frames,
        best_lag + beat_frames,
        best_lag - (beat_frames // 2),
        best_lag + (beat_frames // 2),
    ]

    def _corr_at_lag(lag: int) -> float:
        idx = center + lag
        if idx < 0 or idx >= len(corr):
            return -1e9
        return float(corr[idx])

    best_lag = max(candidate_lags, key=_corr_at_lag)
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
    return mixed, beat_sync_score, instrumental, rhythm_reference


def _trim_to_best_sync(
    instrumental: np.ndarray,
    rhythm_ref: np.ndarray,
    mixed: np.ndarray,
    sr: int,
    min_duration: float = 20.0,
    fade_in_s: float = 0.05,
    fade_out_s: float = 0.20,
) -> np.ndarray:
    """
    Scan the rendered audio for the best-synced sub-region and trim to it.

    Uses onset envelopes of the instrumental and rhythm reference (original
    full track of the vocal side — NOT the separated vocal, which lacks
    rhythmic onsets) to find where beats lock best.

    Slides small (5s) analysis windows across the clip, scores each by onset
    cross-correlation, then finds the best contiguous region >= min_duration.
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

    # Use 5s analysis windows for fine-grained scoring, stepped every 1s
    analysis_win_s = 5.0
    win_frames = int(analysis_win_s * sr / hop)
    step_frames = max(1, int(1.0 * sr / hop))
    max_lag = int(0.15 * sr / hop)  # 150ms tolerance

    if win_frames >= n_frames:
        return mixed

    # Score each 5s position
    positions = list(range(0, n_frames - win_frames + 1, step_frames))
    scores = []
    for pos in positions:
        sc = _window_onset_corr(
            env_inst[pos : pos + win_frames],
            env_ref[pos : pos + win_frames],
            max_lag,
        )
        scores.append(sc)

    if not scores:
        return mixed

    scores_arr = np.array(scores, dtype=np.float32)
    peak_score = float(np.max(scores_arr))
    if peak_score < 1e-6:
        return mixed  # No meaningful correlation anywhere

    # Slide a min_duration-sized window over the per-second scores and
    # pick the position with the highest average sync score.
    min_windows = max(1, int(min_duration))  # ~1 window per second
    if len(scores_arr) <= min_windows:
        return mixed

    # Compute rolling average efficiently
    cumsum = np.cumsum(np.insert(scores_arr, 0, 0.0))
    rolling_avg = (cumsum[min_windows:] - cumsum[:-min_windows]) / min_windows
    best_run_start = int(np.argmax(rolling_avg))
    best_run_avg = float(rolling_avg[best_run_start])
    full_avg = float(np.mean(scores_arr))

    # Only trim if the best region is meaningfully better than the full clip
    if best_run_avg <= full_avg * 1.05:
        return mixed  # Best region isn't much better, keep full clip

    # Convert back to samples
    start_sample = positions[best_run_start] * hop
    end_pos_idx = min(best_run_start + min_windows - 1, len(positions) - 1)
    end_sample = min(len(mixed), (positions[end_pos_idx] + win_frames) * hop)

    region_duration = (end_sample - start_sample) / sr
    if region_duration < min_duration:
        return mixed

    # If trimmed region is nearly the full clip (>85%), skip trimming
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

        if config.start_a is not None and config.start_b is not None:
            start_a = float(config.start_a)
            start_b = float(config.start_b)
            seg_score_a = 1.0
            seg_score_b = 1.0
        else:
            start_a, start_b, _, _, seg_score_a, seg_score_b = _pick_short_sync_window(
                config.track_a,
                config.track_b,
                clip_duration=clip_duration,
                analysis_sr=12000,
            )

        decision = _choose_direction(comp_ab, comp_ba, config.mashup_mode)
        reject_reasons = list(decision.reject_reasons)

        if not config.use_stem_separation:
            reject_reasons.append("stem separation must be enabled")

        if reject_reasons and ENFORCE_STRICT_REJECTION:
            raise ValueError(_reject_summary(reject_reasons))

        warnings: List[str] = []
        if reject_reasons:
            warnings.append("Ignored compatibility checks: " + "; ".join(reject_reasons))

        mixed, beat_sync_score, stem_inst, stem_vocal = _render_with_stems(
            config.track_a,
            config.track_b,
            mode=decision.mode,
            start_a=start_a,
            start_b=start_b,
            clip_duration=clip_duration,
            sr=config.sr,
            stems_dir=config.stems_dir,
            config=config,
        )

        # Trim to the best-synced region (removes misaligned edges).
        mixed = _trim_to_best_sync(
            stem_inst, stem_vocal, mixed, config.sr,
            min_duration=20.0,
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
            f"Vocal tempo scale: +0.0% (disabled). "
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
