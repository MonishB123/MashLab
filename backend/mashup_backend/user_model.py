"""
user_model.py

Online logistic regression for personalized mashup compatibility scoring.

Uses pairwise interaction features derived from TrackFeatures to learn
user preferences via single-step SGD updates on thumbs-up/down feedback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from analyze_track import TrackFeatures
from compatibility import best_tempo_alignment, is_relative_major_minor

MODELS_DIR = Path(__file__).resolve().parent / ".user_models"
NUM_FEATURES = 11


def _cosine_sim(a: list, b: list) -> float:
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return 0.0
    # Pad shorter to match longer
    max_len = max(len(a_arr), len(b_arr))
    if len(a_arr) < max_len:
        a_arr = np.pad(a_arr, (0, max_len - len(a_arr)))
    if len(b_arr) < max_len:
        b_arr = np.pad(b_arr, (0, max_len - len(b_arr)))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def extract_pairwise_features(feats_a: TrackFeatures, feats_b: TrackFeatures) -> np.ndarray:
    """Build an 11-element pairwise feature vector from two tracks."""
    bpm_a = max(feats_a.tempo_bpm, 1e-6)
    bpm_b = max(feats_b.tempo_bpm, 1e-6)
    bpm_diff = abs(bpm_a - bpm_b) / max(bpm_a, bpm_b)

    same_key = (feats_a.key_pc_guess == feats_b.key_pc_guess) and (feats_a.mode == feats_b.mode)
    relative = is_relative_major_minor(feats_a.key_pc_guess, feats_a.mode,
                                       feats_b.key_pc_guess, feats_b.mode)
    if same_key:
        key_match = 1.0
    elif relative:
        key_match = 0.85
    else:
        key_match = 0.0

    energy_diff = abs(feats_a.rms_mean - feats_b.rms_mean)

    spectral_contrast_sim = _cosine_sim(feats_a.spectral_contrast_mean,
                                         feats_b.spectral_contrast_mean)
    timbre_sim = _cosine_sim(feats_a.mfcc_mean, feats_b.mfcc_mean)
    tonnetz_sim = _cosine_sim(feats_a.tonnetz_mean, feats_b.tonnetz_mean)

    danceability_diff = abs(feats_a.danceability - feats_b.danceability)

    loudness_diff = abs(feats_a.loudness_db_mean - feats_b.loudness_db_mean) / 60.0  # normalize

    _, stretch_b, _ = best_tempo_alignment(bpm_a, bpm_b)
    tempo_score = max(0.0, 1.0 - abs((stretch_b - 1.0) * 100.0) / 15.0)

    c_a = max(1.0, feats_a.spectral_centroid_hz)
    c_b = max(1.0, feats_b.spectral_centroid_hz)
    freq_separation = abs(np.log2(c_a / c_b))

    bias = 1.0

    return np.array([
        bpm_diff,
        key_match,
        energy_diff,
        spectral_contrast_sim,
        timbre_sim,
        tonnetz_sim,
        danceability_diff,
        loudness_diff,
        tempo_score,
        freq_separation,
        bias,
    ], dtype=np.float64)


def predict(weights: np.ndarray, features: np.ndarray) -> float:
    """Sigmoid(w^T x), returns P(good mashup)."""
    z = float(np.dot(weights, features))
    z = np.clip(z, -500, 500)
    return float(1.0 / (1.0 + np.exp(-z)))


def update_weights(
    weights: np.ndarray,
    features: np.ndarray,
    label: int,
    lr: float = 0.05,
    l2_lambda: float = 0.01,
) -> np.ndarray:
    """Single SGD step with L2 regularization."""
    y_hat = predict(weights, features)
    weights = weights * (1 - lr * l2_lambda) + lr * (label - y_hat) * features
    return weights


def default_weights() -> np.ndarray:
    """Sensible initial weights for cold-start."""
    return np.array([
        -0.5,   # bpm_diff (negative = penalize mismatch)
        1.0,    # key_match (positive = reward match)
        -0.3,   # energy_diff (negative = penalize)
        0.3,    # spectral_contrast_sim
        0.5,    # timbre_sim
        0.3,    # tonnetz_sim
        -0.2,   # danceability_diff
        -0.3,   # loudness_diff
        0.8,    # tempo_score (positive = reward alignment)
        0.2,    # freq_separation
        0.5,    # bias
    ], dtype=np.float64)


def blend_score(base_score: float, model_score: float, n_votes: int) -> float:
    """Blend base compatibility with personalized prediction.

    With few votes, base dominates. As votes grow, model takes over.
    """
    alpha = min(n_votes / 20.0, 0.6)
    return (1.0 - alpha) * base_score + alpha * model_score * 100.0


def _model_path(user_id: str) -> Path:
    return MODELS_DIR / f"{user_id}.json"


def load_user_model(user_id: str) -> dict:
    """Load user model from disk, or return default."""
    path = _model_path(user_id)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            data["weights"] = np.array(data["weights"], dtype=np.float64)
            return data
        except Exception:
            pass
    return {
        "weights": default_weights(),
        "n_votes": 0,
        "version": 1,
    }


def save_user_model(user_id: str, model: dict) -> None:
    """Persist user model to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(user_id)
    data = {
        "weights": model["weights"].tolist() if isinstance(model["weights"], np.ndarray) else model["weights"],
        "n_votes": model["n_votes"],
        "version": model.get("version", 1),
    }
    path.write_text(json.dumps(data, indent=2))


def get_personalized_score(
    user_id: str,
    feats_a: TrackFeatures,
    feats_b: TrackFeatures,
    base_score: float,
) -> Optional[float]:
    """Compute blended personalized score if user has a model."""
    model = load_user_model(user_id)
    if model["n_votes"] == 0:
        return None
    features = extract_pairwise_features(feats_a, feats_b)
    model_score = predict(model["weights"], features)
    return blend_score(base_score, model_score, model["n_votes"])
