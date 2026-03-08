"""
compatibility.py

Minimal compatibility logic for mashup layering.

Only three factors are evaluated:
1) Key relation: exact key/mode match or relative major/minor.
2) Rhythmic signature scaling: tempo must match directly or via simple ratio scaling.
3) Frequency separation: tracks should occupy different spectral regions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from analyze_track import TrackFeatures


# Constraints for the simplified formula.
MAX_STRETCH_PCT = 15.0
MIN_COMPATIBILITY_SCORE = 60.0
MIN_FREQ_SEPARATION_OCT = float(os.environ.get("MASHUP_MIN_FREQ_SEP_OCT", "0.15"))


def best_tempo_alignment(bpm_a: float, bpm_b: float) -> Tuple[float, float, float]:
    """
    Returns:
      ratio_used: interpretation ratio applied to B before matching A
      stretch_factor: final stretch applied to B so it matches A
      effective_bpm_b: BPM of B after ratio_used interpretation
    """
    ratios = [1.0, 2.0, 0.5, 4.0 / 3.0, 3.0 / 4.0]
    best_ratio, best_stretch, best_diff = 1.0, 1.0, float("inf")
    best_eff_b = bpm_b
    for ratio in ratios:
        eff_b = bpm_b * ratio
        diff = abs(bpm_a - eff_b) / max(bpm_a, 1e-9)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
            best_stretch = bpm_a / max(eff_b, 1e-9)
            best_eff_b = eff_b
    return best_ratio, best_stretch, best_eff_b


def is_relative_major_minor(pc_a: int, mode_a: int, pc_b: int, mode_b: int) -> bool:
    """
    Relative major/minor relationship:
    - major tonic X with minor tonic X+9
    - minor tonic X with major tonic X+3
    """
    if mode_a == 0 and mode_b == 1:
        return pc_b == (pc_a + 9) % 12
    if mode_a == 1 and mode_b == 0:
        return pc_b == (pc_a + 3) % 12
    return False


@dataclass
class CompatibilityResult:
    compatibility_score: float
    grade: str

    embedding_similarity: float
    camelot_key_score: float
    tempo_similarity: float
    energy_similarity: float

    tempo_score: float
    key_score: float
    energy_score: float
    loudness_score: float
    timbre_score: float
    spectral_contrast_score: float
    tonnetz_score: float
    danceability_match_score: float

    tempo_ratio_used: float
    stretch_factor_b: float
    stretch_pct_b: float
    pitch_shift_b: int
    gain_db_b: float
    mashup_type: str

    layerable: bool
    reject_reasons: List[str]
    summary: str


def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 45:
        return "D"
    return "F"


def compare_tracks(track_a: TrackFeatures, track_b: TrackFeatures) -> CompatibilityResult:
    """
    Directional compatibility:
      track_a is the timing/key reference (instrumental side),
      track_b is the vocal side to align.
    """
    # 1) Key compatibility.
    same_key = (track_a.key_pc_guess == track_b.key_pc_guess) and (track_a.mode == track_b.mode)
    relative = is_relative_major_minor(track_a.key_pc_guess, track_a.mode, track_b.key_pc_guess, track_b.mode)
    if same_key:
        key_score = 1.0
        key_relation = "exact"
    elif relative:
        key_score = 0.85
        key_relation = "relative major/minor"
    else:
        key_score = 0.3
        key_relation = "different"

    # 2) Signature matching through simple tempo scaling.
    ratio_used, stretch_b, _ = best_tempo_alignment(track_a.tempo_bpm, track_b.tempo_bpm)
    stretch_pct_b = (stretch_b - 1.0) * 100.0
    signature_score = max(0.0, 1.0 - abs(stretch_pct_b) / MAX_STRETCH_PCT)

    # 3) Frequency separation.
    c_a = max(1.0, float(track_a.spectral_centroid_hz))
    c_b = max(1.0, float(track_b.spectral_centroid_hz))
    freq_sep_oct = abs(np.log2(c_a / c_b))
    freq_score = float(np.clip(freq_sep_oct / 0.5, 0.0, 1.0))

    BASELINE_BIAS = 40.0
    final_score = BASELINE_BIAS + (100.0 - BASELINE_BIAS) * (0.45 * key_score + 0.35 * signature_score + 0.20 * freq_score)

    reject_reasons: List[str] = []
    if abs(stretch_pct_b) > MAX_STRETCH_PCT:
        reject_reasons.append(
            f"signature/tempo scaling too large ({stretch_pct_b:+.1f}% needed, max ±{MAX_STRETCH_PCT:.1f}%)"
        )
    if freq_sep_oct < MIN_FREQ_SEPARATION_OCT:
        reject_reasons.append(
            f"frequency separation too small ({freq_sep_oct:.2f} octaves; need at least {MIN_FREQ_SEPARATION_OCT:.2f})"
        )
    if final_score < MIN_COMPATIBILITY_SCORE:
        reject_reasons.append(
            f"compatibility score too low ({final_score:.0f}/100, need at least {MIN_COMPATIBILITY_SCORE:.0f})"
        )

    layerable = len(reject_reasons) == 0
    if layerable:
        summary = (
            f"Layerable ({_grade(final_score)}, {final_score:.0f}/100). "
            f"Key: {key_relation}. Signature scale: {stretch_pct_b:+.1f}%. "
            f"Frequency separation: {freq_sep_oct:.2f} oct."
        )
    else:
        summary = "Not layerable: " + "; ".join(reject_reasons) + "."

    # Keep legacy response fields for frontend/backward compatibility.
    return CompatibilityResult(
        compatibility_score=float(final_score),
        grade=_grade(final_score),
        embedding_similarity=float(freq_score),
        camelot_key_score=float(key_score),
        tempo_similarity=float(signature_score),
        energy_similarity=float(freq_score),
        tempo_score=float(signature_score),
        key_score=float(key_score),
        energy_score=float(freq_score),
        loudness_score=0.0,
        timbre_score=float(freq_score),
        spectral_contrast_score=float(freq_score),
        tonnetz_score=float(key_score),
        danceability_match_score=float(signature_score),
        tempo_ratio_used=float(ratio_used),
        stretch_factor_b=float(stretch_b),
        stretch_pct_b=float(stretch_pct_b),
        pitch_shift_b=0,
        gain_db_b=0.0,
        mashup_type="inst_a_vocals_b",
        layerable=layerable,
        reject_reasons=reject_reasons,
        summary=summary,
    )
