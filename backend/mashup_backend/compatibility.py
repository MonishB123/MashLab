"""
compatibility.py  (v3)

Scores mashup compatibility using the full feature set from analyze_track.py v2.

What's new vs v2:
  - mode (major/minor) is now real data from Krumhansl-Schmuckler, not hardcoded 0
  - key_score now correctly uses actual mode for relative major/minor detection
  - spectral_contrast_score: compares tonal texture between tracks
  - tonnetz_score: compares harmonic tension/resolution patterns
  - hp_ratio_score: checks if both tracks have similar harmonic/percussive balance
  - danceability_match: rewards similar danceability (you want both drops to feel the same)
  - speechiness used to inform mashup_type: if A is vocal and B is instrumental, 
    recommend vocals_a_inst_b automatically

Scoring weights (sum to 1.0):
  - Tempo compatibility        30%
  - Key/harmonic compatibility 30%
  - Tonal texture (contrast)   10%
  - Tonnetz harmonic distance   8%
  - Energy/dynamics            10%
  - Loudness match              7%
  - Danceability match          5%
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from analyze_track import TrackFeatures


# ---------------------------------------------------------------------------
# Tempo alignment
# ---------------------------------------------------------------------------

def best_tempo_alignment(bpm_a: float, bpm_b: float) -> Tuple[float, float, float]:
    ratios = [1.0, 2.0, 0.5, 4.0 / 3.0, 3.0 / 4.0]
    best_ratio, best_stretch, best_diff = 1.0, 1.0, float("inf")
    for r in ratios:
        eff_b = bpm_b * r
        diff = abs(bpm_a - eff_b) / max(bpm_a, 1e-9)
        if diff < best_diff:
            best_diff = diff
            best_ratio = r
            best_stretch = bpm_a / max(eff_b, 1e-9)
    return best_ratio, best_stretch, best_diff


def tempo_score(diff_pct: float) -> float:
    if diff_pct <= 0.02:
        return 1.0
    if diff_pct <= 0.05:
        return 1.0 - ((diff_pct - 0.02) / 0.03) * 0.40
    if diff_pct <= 0.10:
        return 0.60 - ((diff_pct - 0.05) / 0.05) * 0.30
    return 0.0


# ---------------------------------------------------------------------------
# Key / harmonic compatibility
# ---------------------------------------------------------------------------

def pc_dist(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def is_relative_major_minor(key_a: int, mode_a: int, key_b: int, mode_b: int) -> bool:
    """True if one key is the relative major/minor of the other (e.g. C major / A minor)."""
    return (mode_a != mode_b) and (pc_dist(key_a, key_b) == 3)


def key_score(key_a: int, mode_a: int, key_b: int, mode_b: int) -> Tuple[float, int]:
    """
    Harmonic compatibility using Camelot wheel rules + all semitone shifts.
    mode_a and mode_b are now real values from Krumhansl-Schmuckler (0=major, 1=minor).
    """
    best_score, best_shift = -1.0, 0
    for shift in range(-6, 7):
        sk_b = (key_b + shift) % 12
        d = pc_dist(key_a, sk_b)

        if d == 0 and mode_a == mode_b:
            score = 1.0
        elif is_relative_major_minor(key_a, mode_a, sk_b, mode_b):
            score = 0.95
        elif d == 0:
            score = 0.90        # Same root, parallel mode (e.g. C major / C minor)
        elif d == 5 or d == 7:
            score = 0.72        # Perfect 4th / 5th — classic DJ harmony
        elif d == 1:
            score = 0.78
        elif d == 2:
            score = 0.58
        elif d == 3:
            score = 0.45
        else:
            score = max(0.0, 0.35 - d * 0.05)

        score = max(0.0, score - abs(shift) * 0.01)  # Slight penalty for large shifts
        if score > best_score:
            best_score = score
            best_shift = shift

    return best_score, best_shift


# ---------------------------------------------------------------------------
# Tonnetz harmonic distance
# Measures similarity of tonal centroid vectors — captures harmonic tension
# compatibility beyond just key matching.
# Reference: Harte et al. (2006)
# ---------------------------------------------------------------------------

def tonnetz_score(tonnetz_a: list, tonnetz_b: list) -> float:
    a = np.array(tonnetz_a)
    b = np.array(tonnetz_b)
    dist = float(np.linalg.norm(a - b))
    # Typical distance range for compatible keys: 0–0.5
    # Incompatible keys: 0.8+
    score = max(0.0, 1.0 - dist / 0.8)
    return float(score)


# ---------------------------------------------------------------------------
# Spectral contrast compatibility
# Two tracks with similar contrast profiles will blend tonally.
# ---------------------------------------------------------------------------

def spectral_contrast_score(contrast_a: list, contrast_b: list) -> float:
    a = np.array(contrast_a)
    b = np.array(contrast_b)
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float((cos_sim + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# Energy, loudness, danceability, H/P ratio
# ---------------------------------------------------------------------------

def estimate_energy_from_rms(rms_mean: float) -> float:
    return max(0.0, min(1.0, rms_mean / 0.10))


def energy_score(energy_a: float, energy_b: float) -> float:
    diff = abs(energy_a - energy_b)
    if diff <= 0.08:
        return 1.0
    if diff <= 0.20:
        return 1.0 - ((diff - 0.08) / 0.12) * 0.40
    if diff <= 0.40:
        return 0.60 - ((diff - 0.20) / 0.20) * 0.40
    return 0.0


def loudness_score(loud_a: float, loud_b: float) -> float:
    diff = abs(loud_a - loud_b)
    if diff <= 3.0:
        return 1.0
    if diff <= 6.0:
        return 1.0 - ((diff - 3.0) / 3.0) * 0.30
    if diff <= 10.0:
        return 0.70 - ((diff - 6.0) / 4.0) * 0.40
    return 0.2


def danceability_match_score(d_a: float, d_b: float) -> float:
    """
    For a mashup to feel cohesive, both tracks should have similar danceability.
    A high-energy drop over a slow groove sounds jarring.
    """
    diff = abs(d_a - d_b)
    if diff <= 0.10:
        return 1.0
    if diff <= 0.25:
        return 1.0 - ((diff - 0.10) / 0.15) * 0.50
    return max(0.0, 0.50 - (diff - 0.25) * 1.5)


def hp_ratio_score(hp_a: float, hp_b: float) -> float:
    """
    Harmonic/percussive balance match.
    Mixing a very percussion-heavy track with a very melodic one can feel off.
    (This is already partially captured by energy, but H/P is more specific.)
    """
    diff = abs(hp_a - hp_b)
    return max(0.0, 1.0 - diff * 2.0)


# ---------------------------------------------------------------------------
# Mashup type recommendation
# Now uses real speechiness/instrumentalness data from analysis
# ---------------------------------------------------------------------------

def mashup_type_recommendation(
    t_score: float,
    k_score: float,
    e_score: float,
    speech_a: float,
    speech_b: float,
    inst_a: float,
    inst_b: float,
) -> str:
    """
    Recommend the best blend mode using real vocal content data.
    """
    vocal_a = speech_a > 0.35 or inst_a < 0.50   # Track A likely has vocals
    vocal_b = speech_b > 0.35 or inst_b < 0.50   # Track B likely has vocals

    if t_score >= 0.75 and k_score >= 0.75:
        if vocal_a and not vocal_b:
            return "vocals_a_inst_b"
        if not vocal_a and vocal_b:
            return "inst_a_vocals_b"
        return "full_blend"

    if k_score >= 0.70 and t_score < 0.50:
        if vocal_a:
            return "vocals_a_inst_b"
        return "vocals_over_instrumental"

    if t_score >= 0.70 and k_score < 0.50:
        if vocal_a:
            return "acapella_over_beat"
        return "acapella_over_beat"

    if e_score >= 0.80:
        return "energy_match_blend"

    return "selective_blend"


# ---------------------------------------------------------------------------
# Result dataclass + scoring
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityResult:
    compatibility_score: float
    grade: str
    # Component scores
    tempo_score: float
    key_score: float
    tonnetz_score: float
    spectral_contrast_score: float
    energy_score: float
    loudness_score: float
    danceability_match_score: float
    # Render parameters
    tempo_ratio_used: float
    stretch_factor_b: float
    stretch_pct_b: float
    pitch_shift_b: int
    gain_db_b: float
    # Metadata
    mashup_type: str
    summary: str


def _grade(s: float) -> str:
    if s >= 85: return "A"
    if s >= 70: return "B"
    if s >= 55: return "C"
    if s >= 40: return "D"
    return "F"


def _build_summary(r: "CompatibilityResult", fa: TrackFeatures, fb: TrackFeatures) -> str:
    parts = []
    if r.compatibility_score >= 70:
        parts.append(f"Strong mashup match ({r.grade}, {r.compatibility_score:.0f}/100).")
    elif r.compatibility_score >= 50:
        parts.append(f"Moderate compatibility ({r.grade}, {r.compatibility_score:.0f}/100).")
    else:
        parts.append(f"Challenging pairing ({r.grade}, {r.compatibility_score:.0f}/100).")

    # Key info with real mode
    key_str_a = f"{fa.key_name_guess} {fa.mode_name}"
    key_str_b = f"{fb.key_name_guess} {fb.mode_name}"
    if r.key_score >= 0.80:
        parts.append(f"Keys are harmonically compatible ({key_str_a} / {key_str_b}, shift: {r.pitch_shift_b:+d} semitones).")
    else:
        parts.append(f"Key tension detected ({key_str_a} vs {key_str_b}) — pitch shifting B by {r.pitch_shift_b:+d} semitones recommended.")

    if r.tempo_score >= 0.80:
        parts.append(f"Tempos align well (stretch: {r.stretch_pct_b:+.1f}%).")
    else:
        parts.append(f"Significant tempo gap — B will be stretched {r.stretch_pct_b:+.1f}%.")

    parts.append(f"Recommended blend: {r.mashup_type.replace('_', ' ').title()}.")
    return " ".join(parts)


def compare_tracks(track_a: TrackFeatures, track_b: TrackFeatures) -> CompatibilityResult:
    # Tempo
    ratio_used, stretch_b, diff_pct = best_tempo_alignment(track_a.tempo_bpm, track_b.tempo_bpm)
    t_score = tempo_score(diff_pct)

    # Key (now uses real mode from Krumhansl-Schmuckler)
    k_score, pitch_shift_b = key_score(
        track_a.key_pc_guess, track_a.mode,
        track_b.key_pc_guess, track_b.mode,
    )

    # Tonnetz
    tn_score = tonnetz_score(track_a.tonnetz_mean, track_b.tonnetz_mean)

    # Spectral contrast
    sc_score = spectral_contrast_score(
        track_a.spectral_contrast_mean, track_b.spectral_contrast_mean
    )

    # Energy
    energy_a = estimate_energy_from_rms(track_a.rms_mean)
    energy_b = estimate_energy_from_rms(track_b.rms_mean)
    e_score = energy_score(energy_a, energy_b)

    # Loudness
    l_score = loudness_score(track_a.loudness_db_mean, track_b.loudness_db_mean)
    gain_db_b = track_a.loudness_db_mean - track_b.loudness_db_mean

    # Danceability match
    d_score = danceability_match_score(track_a.danceability, track_b.danceability)

    # Weighted composite score
    score = 100.0 * (
        0.30 * t_score
        + 0.30 * k_score
        + 0.10 * sc_score
        + 0.08 * tn_score
        + 0.10 * e_score
        + 0.07 * l_score
        + 0.05 * d_score
    )

    mashup_type = mashup_type_recommendation(
        t_score, k_score, e_score,
        track_a.speechiness, track_b.speechiness,
        track_a.instrumentalness, track_b.instrumentalness,
    )

    grade = _grade(score)

    result = CompatibilityResult(
        compatibility_score=score,
        grade=grade,
        tempo_score=t_score,
        key_score=k_score,
        tonnetz_score=tn_score,
        spectral_contrast_score=sc_score,
        energy_score=e_score,
        loudness_score=l_score,
        danceability_match_score=d_score,
        tempo_ratio_used=ratio_used,
        stretch_factor_b=stretch_b,
        stretch_pct_b=(stretch_b - 1.0) * 100.0,
        pitch_shift_b=pitch_shift_b,
        gain_db_b=gain_db_b,
        mashup_type=mashup_type,
        summary="",
    )
    result.summary = _build_summary(result, track_a, track_b)
    return result
