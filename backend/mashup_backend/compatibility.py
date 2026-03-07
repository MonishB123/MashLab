"""
compatibility.py (AI-Assisted Mashup System)

Implements the AI-Assisted Mashup Compatibility System Technical Design Document.
Replaces manual heuristic weights with a structured feature vector approach:
1. Pretrained audio embeddings (simulated via spectral feature concatenation)
2. Explicit musical features (camelot key score, tempo similarity, energy similarity)
3. Structural mashup constraints (vocal/instrumental volume reduction rules)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from analyze_track import TrackFeatures

# ---------------------------------------------------------------------------
# Camelot Key Compatibility
# ---------------------------------------------------------------------------

# Mapping of pitch class (0=C, 1=C#, etc.) to Camelot position
MAJOR_CAMELOT = {0: 8, 1: 3, 2: 10, 3: 5, 4: 12, 5: 7, 6: 2, 7: 9, 8: 4, 9: 11, 10: 6, 11: 1}
MINOR_CAMELOT = {0: 5, 1: 12, 2: 7, 3: 2, 4: 9, 5: 4, 6: 11, 7: 6, 8: 1, 9: 8, 10: 3, 11: 10}

def camelot_distance(pc1: int, mode1: int, pc2: int, mode2: int) -> int:
    """Calculates shortest distance on Camelot wheel."""
    cam1 = MAJOR_CAMELOT[pc1] if mode1 == 0 else MINOR_CAMELOT[pc1]
    cam2 = MAJOR_CAMELOT[pc2] if mode2 == 0 else MINOR_CAMELOT[pc2]
    
    # Distance around the circle (1-12)
    step_dist = min(abs(cam1 - cam2), 12 - abs(cam1 - cam2))
    
    # Mode switch counts as 1 step (A to B)
    mode_diff = 0 if mode1 == mode2 else 1
    
    return step_dist + mode_diff

def calculate_camelot_key_score(pc1: int, mode1: int, pc2: int, mode2: int) -> float:
    """
    Camelot distance mapped to score:
    same key = 1.0, adjacent key = 0.8, two steps = 0.5, otherwise = 0
    """
    dist = camelot_distance(pc1, mode1, pc2, mode2)
    if dist == 0: return 1.0
    if dist == 1: return 0.8
    if dist == 2: return 0.5
    return 0.0

# Keep this for the rendering engine which needs to know how to pitch shift
def best_pitch_shift(pc1: int, mode1: int, pc2: int, mode2: int) -> int:
    best_shift = 0
    best_dist = 999
    for shift in range(-6, 7):
        test_pc2 = (pc2 + shift) % 12
        d = camelot_distance(pc1, mode1, test_pc2, mode2)
        if d < best_dist:
            best_dist = d
            best_shift = shift
        elif d == best_dist and abs(shift) < abs(best_shift):
            best_shift = shift
    return best_shift

# ---------------------------------------------------------------------------
# Handcrafted Audio Feature Embeddings
# ---------------------------------------------------------------------------

def get_audio_embedding(track: TrackFeatures) -> np.ndarray:
    """
    Creates a handcrafted audio feature embedding (≈53 dimensions) by 
    concatenating explicit spectral features into a dense vector.
    """
    return np.concatenate([
        np.array(track.mfcc_mean),
        np.array(track.mfcc_std),
        np.array(track.spectral_contrast_mean),
        np.array(track.tonnetz_mean)
    ])

def calculate_embedding_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity of audio embeddings."""
    cos_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9)
    return float((cos_sim + 1.0) / 2.0) # Normalize to 0-1

# ---------------------------------------------------------------------------
# Tempo & Energy
# ---------------------------------------------------------------------------

def best_tempo_alignment(bpm_a: float, bpm_b: float) -> Tuple[float, float, float]:
    ratios = [1.0, 2.0, 0.5, 4.0 / 3.0, 3.0 / 4.0]
    best_ratio, best_stretch, best_diff = 1.0, 1.0, float("inf")
    best_eff_b = bpm_b
    for r in ratios:
        eff_b = bpm_b * r
        diff = abs(bpm_a - eff_b) / max(bpm_a, 1e-9)
        if diff < best_diff:
            best_diff = diff
            best_ratio = r
            best_stretch = bpm_a / max(eff_b, 1e-9)
            best_eff_b = eff_b
    return best_ratio, best_stretch, best_eff_b

def calculate_tempo_similarity(bpm_a: float, eff_bpm_b: float, max_diff: float = 40.0) -> float:
    """1 - |BPM_A - BPM_B| / max_tempo_difference"""
    diff = abs(bpm_a - eff_bpm_b)
    return max(0.0, 1.0 - diff / max_diff)

def estimate_energy_from_rms(rms_mean: float) -> float:
    return max(0.0, min(1.0, rms_mean / 0.10))

def calculate_energy_similarity(energy_a: float, energy_b: float) -> float:
    return max(0.0, 1.0 - abs(energy_a - energy_b))

# ---------------------------------------------------------------------------
# Mashup Constraints & Structural Filters
# ---------------------------------------------------------------------------

def determine_vocal_presence(speechiness: float, instrumentalness: float) -> str:
    """Classifies track into: lyrical, instrumental, or mixed."""
    if instrumentalness > 0.8:
        return "instrumental"
    elif speechiness > 0.25 or instrumentalness < 0.2:
        return "lyrical"
    return "mixed"

# ---------------------------------------------------------------------------
# Result dataclass + Engine
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityResult:
    compatibility_score: float
    grade: str
    
    # Core AI-Assisted Features
    embedding_similarity: float
    camelot_key_score: float
    tempo_similarity: float
    energy_similarity: float

    # Legacy attributes required by API/Engine
    tempo_score: float
    key_score: float
    tempo_ratio_used: float
    stretch_factor_b: float
    stretch_pct_b: float
    pitch_shift_b: int
    gain_db_b: float
    mashup_type: str
    summary: str

def _grade(s: float) -> str:
    if s >= 85: return "A"
    if s >= 70: return "B"
    if s >= 55: return "C"
    if s >= 40: return "D"
    return "F"

def compare_tracks(track_a: TrackFeatures, track_b: TrackFeatures) -> CompatibilityResult:
    # 1. Feature Extraction & Alignment
    ratio_used, stretch_b, eff_bpm_b = best_tempo_alignment(track_a.tempo_bpm, track_b.tempo_bpm)
    pitch_shift_b = best_pitch_shift(track_a.key_pc_guess, track_a.mode, track_b.key_pc_guess, track_b.mode)
    
    energy_a = estimate_energy_from_rms(track_a.rms_mean)
    energy_b = estimate_energy_from_rms(track_b.rms_mean)
    
    emb_a = get_audio_embedding(track_a)
    emb_b = get_audio_embedding(track_b)
    
    # 2. Pairwise Compatibility Engine Features
    emb_sim = calculate_embedding_similarity(emb_a, emb_b)
    
    # Key score natively without shift to see how naturally they fit
    cam_score = calculate_camelot_key_score(
        track_a.key_pc_guess, track_a.mode,
        track_b.key_pc_guess, track_b.mode
    )
    
    tempo_sim = calculate_tempo_similarity(track_a.tempo_bpm, eff_bpm_b)
    energy_sim = calculate_energy_similarity(energy_a, energy_b)
    
    # Final Compatibility Score (Equal weighting for base, tunable later via user ratings)
    final_score = 100.0 * (0.25 * emb_sim + 0.25 * cam_score + 0.25 * tempo_sim + 0.25 * energy_sim)
    
    # 3. Mashup Constraint Filter
    vocal_a = determine_vocal_presence(track_a.speechiness, track_a.instrumentalness)
    vocal_b = determine_vocal_presence(track_b.speechiness, track_b.instrumentalness)
    
    mashup_type = "vocals_a_inst_b" # Default safe mode to strictly enforce separation
    
    # Smart Leveling: Target -14 LUFS / perceived loudness balance.
    # We want the instrumental to be the 'anchor'. 
    # Usually, a vocal-only stem is ~6-10dB quieter than a full master.
    # If we match their means, the beat gets crushed.
    # We calculate the delta, but cap the reduction on the beat.
    raw_delta = track_a.loudness_db_mean - track_b.loudness_db_mean
    gain_db_b = max(-6.0, min(6.0, raw_delta)) # Don't allow more than 6dB auto-swing
    
    # Apply vocal constraints (Treat 'mixed' as 'lyrical' to strictly enforce separation)
    is_lyrical_a = vocal_a in ["lyrical", "mixed"]
    is_lyrical_b = vocal_b in ["lyrical", "mixed"]

    if is_lyrical_a and is_lyrical_b:
        mashup_type = "vocals_a_inst_b"
        # Boost the beat slightly to compensate for vocal density
        gain_db_b += 3.0 
    elif is_lyrical_a and vocal_b == "instrumental":
        mashup_type = "vocals_over_instrumental"
        gain_db_b += 3.0
    elif vocal_a == "instrumental" and is_lyrical_b:
        mashup_type = "inst_a_vocals_b"
        gain_db_b -= 3.0

    summary_parts = [
        f"AI System Compatibility ({_grade(final_score)}, {final_score:.0f}/100)."
    ]
    if cam_score < 0.8:
        summary_parts.append(f"Requires pitch shift ({pitch_shift_b:+d} st).")
    if is_lyrical_a and is_lyrical_b:
        summary_parts.append("Constraint applied: dual lyrical detected, strictly isolating A-vocals and B-beat.")
        
    return CompatibilityResult(
        compatibility_score=final_score,
        grade=_grade(final_score),
        embedding_similarity=emb_sim,
        camelot_key_score=cam_score,
        tempo_similarity=tempo_sim,
        energy_similarity=energy_sim,
        tempo_score=tempo_sim,  # Legacy alias
        key_score=cam_score,    # Legacy alias
        tempo_ratio_used=ratio_used,
        stretch_factor_b=stretch_b,
        stretch_pct_b=(stretch_b - 1.0) * 100.0,
        pitch_shift_b=pitch_shift_b,
        gain_db_b=gain_db_b,
        mashup_type=mashup_type,
        summary=" ".join(summary_parts)
    )
