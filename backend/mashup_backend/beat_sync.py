"""
beat_sync.py

Beat alignment helpers for strict mashup layering.
"""

from __future__ import annotations

from typing import Optional, Tuple

import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Beat tracking
# ---------------------------------------------------------------------------


def _beat_times_madmom(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        import collections
        import collections.abc

        for name in ("MutableMapping", "MutableSequence", "MutableSet"):
            if not hasattr(collections, name):
                setattr(collections, name, getattr(collections.abc, name))

        import madmom

        if sr != 44100:
            y = librosa.resample(y, orig_sr=sr, target_sr=44100)

        proc = madmom.features.RNNBeatProcessor()
        beat_act = proc(y.astype(np.float32))
        dbn = madmom.features.DBNBeatTrackingProcessor(fps=100)
        beat_times = dbn(beat_act)
        return np.array(beat_times, dtype=float)
    except (ImportError, Exception):
        return _beat_times_librosa(y, sr)



def _beat_times_librosa(y: np.ndarray, sr: int) -> np.ndarray:
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        trim=False,
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    correction_s = 0.030
    return np.maximum(0.0, beat_times - correction_s)



def get_beat_times(y: np.ndarray, sr: int, use_madmom: bool = True) -> np.ndarray:
    if use_madmom:
        return _beat_times_madmom(y, sr)
    return _beat_times_librosa(y, sr)


# ---------------------------------------------------------------------------
# BPM / alignment helpers
# ---------------------------------------------------------------------------


def median_ibi_bpm(beat_times: np.ndarray) -> float:
    if len(beat_times) < 4:
        return 120.0
    ibis = np.diff(beat_times)
    median_ibi = float(np.median(ibis))
    ibis_clean = ibis[(ibis > median_ibi * 0.5) & (ibis < median_ibi * 2.0)]
    if len(ibis_clean) == 0:
        ibis_clean = ibis
    return float(60.0 / np.median(ibis_clean))



def _beat_grid_signal(beat_times: np.ndarray, sr: int, length_samples: int) -> np.ndarray:
    signal = np.zeros(length_samples, dtype=np.float32)
    for t in beat_times:
        idx = int(round(t * sr))
        if 0 <= idx < length_samples:
            signal[idx] = 1.0
    return signal



def find_phase_offset_samples(
    beat_times_a: np.ndarray,
    beat_times_b: np.ndarray,
    sr: int,
    length_samples: int,
    search_window_s: float = 4.0,
) -> int:
    if len(beat_times_a) == 0 or len(beat_times_b) == 0:
        return 0

    from scipy.ndimage import gaussian_filter1d

    grid_a = _beat_grid_signal(beat_times_a, sr, length_samples)
    grid_b = _beat_grid_signal(beat_times_b, sr, length_samples)
    sigma = max(1, int(0.005 * sr))
    grid_a = gaussian_filter1d(grid_a, sigma=sigma)
    grid_b = gaussian_filter1d(grid_b, sigma=sigma)

    corr = np.correlate(grid_a, grid_b, mode="full")
    center = len(grid_b) - 1
    max_offset = min(int(search_window_s * sr), len(grid_a) // 4)

    lo = max(0, center - max_offset)
    hi = min(len(corr), center + max_offset + 1)
    window = corr[lo:hi]
    if len(window) == 0:
        return 0

    best_idx = int(np.argmax(window))
    lag = (lo + best_idx) - center
    return int(lag)



def compute_beat_sync_score(y_a: np.ndarray, y_b: np.ndarray, sr: int) -> float:
    bt_a = get_beat_times(y_a, sr)
    bt_b = get_beat_times(y_b, sr)
    if len(bt_a) < 4 or len(bt_b) < 4:
        return 0.5

    length = min(len(y_a), len(y_b))
    grid_a = _beat_grid_signal(bt_a, sr, length)
    grid_b = _beat_grid_signal(bt_b, sr, length)

    from scipy.ndimage import gaussian_filter1d

    sigma = max(1, int(0.005 * sr))
    grid_a = gaussian_filter1d(grid_a, sigma=sigma)
    grid_b = gaussian_filter1d(grid_b, sigma=sigma)

    corr = np.correlate(grid_a, grid_b, mode="full")
    norm = (np.linalg.norm(grid_a) * np.linalg.norm(grid_b)) + 1e-9
    score = float(np.max(corr) / norm)
    return float(np.clip(score, 0.0, 1.0))



def align_beats(y_a: np.ndarray, y_b: np.ndarray, sr: int, stretch_factor: float = 1.0) -> np.ndarray:
    """
    Align y_b to y_a using one global stretch and one global phase offset.
    Returns a signal the same length as y_a.
    """
    bt_a = get_beat_times(y_a, sr)
    bt_b = get_beat_times(y_b, sr)

    bpm_a = median_ibi_bpm(bt_a)
    bpm_b_orig = median_ibi_bpm(bt_b)

    if bpm_b_orig > 0 and abs(bpm_a / bpm_b_orig - 1.0) < 0.10:
        precise_stretch = bpm_a / bpm_b_orig
    else:
        precise_stretch = stretch_factor

    precise_stretch = float(np.clip(precise_stretch, 0.95, 1.05))

    try:
        import pyrubberband as rb

        y_b_stretched = rb.time_stretch(y_b, sr, precise_stretch).astype(np.float32)
    except ImportError:
        # librosa uses the same convention: rate > 1.0 speeds up.
        y_b_stretched = librosa.effects.time_stretch(y_b, rate=precise_stretch).astype(np.float32)

    bt_b_stretched = bt_b / precise_stretch if precise_stretch != 0 else bt_b
    length_samples = len(y_a)
    offset = find_phase_offset_samples(bt_a, bt_b_stretched, sr, length_samples)

    if offset > 0:
        y_b_aligned = np.concatenate([np.zeros(offset, dtype=np.float32), y_b_stretched])
    elif offset < 0:
        trim = min(-offset, max(0, len(y_b_stretched) - 1))
        y_b_aligned = y_b_stretched[trim:]
    else:
        y_b_aligned = y_b_stretched

    if len(y_b_aligned) > length_samples:
        y_b_aligned = y_b_aligned[:length_samples]
    elif len(y_b_aligned) < length_samples:
        y_b_aligned = np.concatenate(
            [y_b_aligned, np.zeros(length_samples - len(y_b_aligned), dtype=np.float32)]
        )

    return y_b_aligned.astype(np.float32)
