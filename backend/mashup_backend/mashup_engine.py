"""
mashup_engine.py

High-level orchestrator for the full mashup pipeline:

  1. Analyze both tracks (BPM, key, energy, timbre)
  2. Score compatibility
  3. Find the best musical segments in each track (chorus / beat drop)
  4. Optionally separate stems (vocals / instrumental) via Demucs
  5. Render the mashup clip with beat-matching, pitch correction, gain leveling
  6. Export WAV (and optionally MP3)

Usage:
    from mashup_engine import MashupEngine, MashupConfig

    config = MashupConfig(
        track_a="song_a.mp3",
        track_b="song_b.mp3",
        clip_duration=45.0,
        mashup_mode="auto",   # auto | full_blend | vocals_a_inst_b | inst_a_vocals_b
        wav_out="mashup.wav",
        mp3_out="mashup.mp3",
    )
    result = MashupEngine().run(config)
    print(result.summary)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

from analyze_track import analyze_mp3, TrackFeatures, as_json_dict
from compatibility import compare_tracks, CompatibilityResult
from segment_finder import pick_best_aligned_segments
from audio_render import (
    load_audio,
    trim_audio,
    time_stretch_audio,
    pitch_shift_audio,
    apply_gain_db,
    overlay_audio,
    normalize_peak,
    export_wav,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MashupConfig:
    track_a: str
    track_b: str

    clip_duration: float = 45.0          # seconds; 30–60 recommended
    sr: int = 44100                      # output sample rate
    n_segment_candidates: int = 5        # how many candidate segments to evaluate

    # Override auto segment detection with manual start times
    start_a: Optional[float] = None
    start_b: Optional[float] = None

    # Mashup blend mode:
    #   "auto"               — pick based on compatibility scores
    #   "full_blend"         — both tracks mixed equally
    #   "vocals_a_inst_b"    — vocals from A + instrumental from B  (needs Demucs)
    #   "inst_a_vocals_b"    — instrumental from A + vocals from B  (needs Demucs)
    #   "acapella_over_beat" — acapella A over full B
    mashup_mode: str = "auto"
    use_stem_separation: bool = False     # set True to enable Demucs (slow but better)

    # Gain/mix controls
    gain_db_a: float = -1.0
    gain_db_b: Optional[float] = None    # None = auto-balance from loudness scores

    # Fade in/out at clip boundaries (seconds)
    fade_in_s: float = 0.5
    fade_out_s: float = 1.5

    # Output paths
    wav_out: str = "mashup.wav"
    mp3_out: Optional[str] = None
    stems_dir: Optional[str] = None      # where to cache Demucs output


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_fade(y: np.ndarray, sr: int, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    """Apply linear fade-in at start and fade-out at end."""
    y = y.copy()
    fi = int(fade_in_s * sr)
    fo = int(fade_out_s * sr)

    if fi > 0 and fi < len(y):
        y[:fi] *= np.linspace(0.0, 1.0, fi)
    if fo > 0 and fo < len(y):
        y[-fo:] *= np.linspace(1.0, 0.0, fo)

    return y


def _beat_align_offset(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> int:
    """
    Sub-beat alignment: find the sample offset for B that best aligns its
    beat grid with A's beat grid, searching within ±1 beat of zero offset.
    Returns offset in samples (positive = shift B forward).
    """
    _, beats_a = librosa.beat.beat_track(y=y_a, sr=sr, hop_length=hop_length)
    _, beats_b = librosa.beat.beat_track(y=y_b, sr=sr, hop_length=hop_length)

    if len(beats_a) == 0 or len(beats_b) == 0:
        return 0

    # Convert first beat frame to samples
    first_beat_a = int(librosa.frames_to_samples(beats_a[0], hop_length=hop_length))
    first_beat_b = int(librosa.frames_to_samples(beats_b[0], hop_length=hop_length))

    return first_beat_a - first_beat_b


def _render_full_blend(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int,
    comp: CompatibilityResult,
    gain_db_b: float,
    config: MashupConfig,
) -> np.ndarray:
    """Standard full blend: both tracks mixed together."""
    y_b = time_stretch_audio(y_b, comp.stretch_factor_b)
    y_b = pitch_shift_audio(y_b, sr, comp.pitch_shift_b)
    y_a = apply_gain_db(y_a, config.gain_db_a)
    y_b = apply_gain_db(y_b, gain_db_b)

    # Beat-align B to A
    offset = _beat_align_offset(y_a, y_b, sr)
    if offset > 0 and offset < len(y_b):
        y_b = y_b[offset:]
    elif offset < 0:
        silence = np.zeros(-offset, dtype=np.float32)
        y_b = np.concatenate([silence, y_b])

    return overlay_audio(y_a, y_b)


def _render_with_stems(
    path_a: str,
    path_b: str,
    mode: str,
    comp: CompatibilityResult,
    start_a: float,
    start_b: float,
    clip_duration: float,
    gain_db_b: float,
    sr: int,
    stems_dir: Optional[str],
    config: MashupConfig,
) -> np.ndarray:
    """Stem-separated render: vocals from one track, instrumental from other."""
    from source_separation import separate_track, blend_stems

    tmpdir = stems_dir or tempfile.mkdtemp(prefix="mashup_stems_")

    stems_a = separate_track(path_a, out_dir=os.path.join(tmpdir, "a"), sr=sr)
    stems_b = separate_track(path_b, out_dir=os.path.join(tmpdir, "b"), sr=sr)

    if mode == "vocals_a_inst_b":
        layer_a, _ = blend_stems(stems_a, ["vocals"])
        layer_b, _ = blend_stems(stems_b, ["drums", "bass", "other"])
    elif mode == "inst_a_vocals_b":
        layer_a, _ = blend_stems(stems_a, ["drums", "bass", "other"])
        layer_b, _ = blend_stems(stems_b, ["vocals"])
    else:
        # acapella_over_beat: A vocals over full B
        layer_a, _ = blend_stems(stems_a, ["vocals"])
        layer_b, _ = blend_stems(stems_b, ["vocals", "drums", "bass", "other"])

    # Clip to desired window
    def clip(y):
        return trim_audio(y, sr, 0, clip_duration)

    clip_a = clip(layer_a)
    clip_b = clip(layer_b)

    clip_b = time_stretch_audio(clip_b, comp.stretch_factor_b)
    clip_b = pitch_shift_audio(clip_b, sr, comp.pitch_shift_b)
    clip_a = apply_gain_db(clip_a, config.gain_db_a)
    clip_b = apply_gain_db(clip_b, gain_db_b)

    return overlay_audio(clip_a, clip_b)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MashupEngine:
    def run(self, config: MashupConfig) -> MashupResult:
        try:
            return self._run(config)
        except Exception as e:
            import traceback
            return MashupResult(
                success=False,
                wav_out=config.wav_out,
                mp3_out=config.mp3_out,
                compatibility=None,  # type: ignore
                start_a=0.0,
                start_b=0.0,
                segment_score_a=0.0,
                segment_score_b=0.0,
                mashup_mode_used="",
                stem_separation_used=False,
                track_a_features={},
                track_b_features={},
                summary="",
                error=f"{e}\n{traceback.format_exc()}",
            )

    def _run(self, config: MashupConfig) -> MashupResult:
        # ── Step 1: Analyze tracks ───────────────────────────────────────────
        feats_a = analyze_mp3(config.track_a, target_sr=22050)
        feats_b = analyze_mp3(config.track_b, target_sr=22050)

        # ── Step 2: Compatibility scoring ────────────────────────────────────
        comp = compare_tracks(feats_a, feats_b)

        # ── Step 3: Find best segments ───────────────────────────────────────
        if config.start_a is not None and config.start_b is not None:
            start_a, seg_score_a = config.start_a, 1.0
            start_b, seg_score_b = config.start_b, 1.0
        else:
            start_a, seg_score_a, start_b, seg_score_b = pick_best_aligned_segments(
                config.track_a,
                config.track_b,
                clip_duration=config.clip_duration,
                n_candidates=config.n_segment_candidates,
                sr=22050,
            )
            if config.start_a is not None:
                start_a, seg_score_a = config.start_a, 1.0
            if config.start_b is not None:
                start_b, seg_score_b = config.start_b, 1.0

        # ── Step 4: Decide blend mode ─────────────────────────────────────────
        mode = config.mashup_mode
        if mode == "auto":
            mode = comp.mashup_type
            # If stem separation off, fall back to full_blend
            if not config.use_stem_separation and mode in (
                "vocals_a_inst_b", "inst_a_vocals_b", "acapella_over_beat"
            ):
                mode = "full_blend"

        gain_db_b = config.gain_db_b if config.gain_db_b is not None else comp.gain_db_b

        # ── Step 5: Render ───────────────────────────────────────────────────
        if config.use_stem_separation and mode in (
            "vocals_a_inst_b", "inst_a_vocals_b", "acapella_over_beat"
        ):
            mixed = _render_with_stems(
                config.track_a, config.track_b,
                mode=mode,
                comp=comp,
                start_a=start_a,
                start_b=start_b,
                clip_duration=config.clip_duration,
                gain_db_b=gain_db_b,
                sr=config.sr,
                stems_dir=config.stems_dir,
                config=config,
            )
        else:
            # Standard full blend
            y_a, sr_a = load_audio(config.track_a, sr=config.sr)
            y_b, sr_b = load_audio(config.track_b, sr=config.sr)

            clip_a = trim_audio(y_a, config.sr, start_a, config.clip_duration)
            clip_b = trim_audio(y_b, config.sr, start_b, config.clip_duration)

            mixed = _render_full_blend(
                clip_a, clip_b,
                sr=config.sr,
                comp=comp,
                gain_db_b=gain_db_b,
                config=config,
            )

        # ── Step 6: Fade & normalize ─────────────────────────────────────────
        mixed = _apply_fade(mixed, config.sr, config.fade_in_s, config.fade_out_s)
        mixed = normalize_peak(mixed, peak=0.92)

        # ── Step 7: Export ───────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(config.wav_out)), exist_ok=True)
        export_wav(mixed, config.sr, config.wav_out)

        mp3_out = None
        if config.mp3_out:
            from audio_render import export_mp3_via_pydub
            export_mp3_via_pydub(config.wav_out, config.mp3_out)
            mp3_out = config.mp3_out

        # ── Step 8: Build result ─────────────────────────────────────────────
        summary_lines = [
            comp.summary,
            f"Clip: Track A starts at {start_a:.1f}s, Track B at {start_b:.1f}s.",
            f"Duration: {config.clip_duration:.0f}s | Mode: {mode.replace('_',' ').title()}.",
        ]
        if config.use_stem_separation:
            summary_lines.append("Stem separation was applied.")

        return MashupResult(
            success=True,
            wav_out=config.wav_out,
            mp3_out=mp3_out,
            compatibility=comp,
            start_a=start_a,
            start_b=start_b,
            segment_score_a=seg_score_a,
            segment_score_b=seg_score_b,
            mashup_mode_used=mode,
            stem_separation_used=config.use_stem_separation,
            track_a_features=as_json_dict(feats_a),
            track_b_features=as_json_dict(feats_b),
            summary=" ".join(summary_lines),
        )
