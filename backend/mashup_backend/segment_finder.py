"""
segment_finder.py

Finds the best clip segment(s) in a song by locating high-energy, musically
interesting moments — ideally a chorus or beat drop rather than intros/outros.

Strategy:
  1. Compute a per-frame energy curve (RMS).
  2. Compute onset strength to find rhythmically dense regions.
  3. Score each candidate window using energy + onset density + beat alignment.
  4. Return the top-N start times.

Usage:
  from segment_finder import find_best_segments
  segments = find_best_segments("song.mp3", clip_duration=30.0, n_candidates=3)
  # returns list of (start_s, score) sorted best-first
"""

from __future__ import annotations

from typing import List, Tuple

import librosa
import numpy as np


def _energy_curve(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Normalized RMS energy per frame."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_max = np.max(rms) + 1e-9
    return rms / rms_max


def _onset_density(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Normalized onset strength envelope per frame."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_max = np.max(onset_env) + 1e-9
    return onset_env / onset_max


def _spectral_flux(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Spectral flux: measures how rapidly the spectrum changes frame-to-frame.
    High flux = energetic / dynamic sections.
    """
    stft = np.abs(librosa.stft(y, hop_length=hop_length))
    flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
    flux = np.concatenate([[0], flux])
    flux_max = np.max(flux) + 1e-9
    return flux / flux_max


def score_windows(
    energy: np.ndarray,
    onset: np.ndarray,
    flux: np.ndarray,
    frames_per_clip: int,
    hop_length: int,
    sr: int,
    total_frames: int,
    skip_start_frac: float = 0.10,
    skip_end_frac: float = 0.10,
) -> np.ndarray:
    """
    Slide a window of `frames_per_clip` frames across the track and compute a
    composite musical-interest score for each start position.

    We skip the first/last `skip_*_frac` of the track to avoid intros/outros.
    """
    skip_start = int(total_frames * skip_start_frac)
    skip_end = int(total_frames * skip_end_frac)
    max_start = total_frames - frames_per_clip - skip_end

    if max_start <= skip_start:
        # Track is too short to skip; use full range
        skip_start = 0
        max_start = max(1, total_frames - frames_per_clip)

    scores = np.full(total_frames, -np.inf)

    for i in range(skip_start, max_start):
        window_energy = energy[i : i + frames_per_clip]
        window_onset = onset[i : i + frames_per_clip]
        window_flux = flux[i : i + frames_per_clip]

        # Weighted combination: energy 45%, onset density 25%, flux 30%
        # Flux is weighted higher to find dynamic 'drops'
        s = (
            0.45 * np.mean(window_energy)
            + 0.25 * np.mean(window_onset)
            + 0.30 * np.mean(window_flux)
        )
        scores[i] = s

    return scores


def find_best_segments(
    audio_path: str,
    clip_duration: float = 45.0,
    n_candidates: int = 3,
    sr: int = 22050,
    hop_length: int = 512,
    min_gap_s: float = 20.0,
    analysis_window_s: float = 90.0,
    use_middle_crop: bool = True,
) -> List[Tuple[float, float]]:
    """
    Analyze an audio file and return the `n_candidates` best start positions
    for a clip of `clip_duration` seconds.

    Returns:
        List of (start_seconds, score) tuples, sorted best-first.
        Candidates are spaced at least `min_gap_s` apart to ensure variety.
    """
    full_duration_s = float(librosa.get_duration(path=audio_path))
    offset_s = 0.0
    load_duration_s = None
    if use_middle_crop and analysis_window_s > 0 and full_duration_s > analysis_window_s:
        offset_s = max(0.0, (full_duration_s - analysis_window_s) * 0.5)
        load_duration_s = analysis_window_s

    y, sr = librosa.load(audio_path, sr=sr, mono=True, offset=offset_s, duration=load_duration_s)
    duration_s = len(y) / sr

    if clip_duration >= duration_s:
        # If analyzed window is too short, return absolute offset.
        return [(float(offset_s), 1.0)]

    energy = _energy_curve(y, sr, hop_length)
    onset = _onset_density(y, sr, hop_length)
    flux = _spectral_flux(y, sr, hop_length)

    total_frames = len(energy)
    frames_per_clip = int(clip_duration * sr / hop_length)
    min_gap_frames = int(min_gap_s * sr / hop_length)

    scores = score_windows(
        energy, onset, flux,
        frames_per_clip=frames_per_clip,
        hop_length=hop_length,
        sr=sr,
        total_frames=total_frames,
    )

    # Greedy selection with minimum-gap suppression
    results: List[Tuple[float, float]] = []
    temp_scores = scores.copy()

    for _ in range(n_candidates):
        best_frame = int(np.argmax(temp_scores))
        if temp_scores[best_frame] == -np.inf:
            break
        best_time = librosa.frames_to_time(best_frame, sr=sr, hop_length=hop_length) + offset_s
        results.append((float(best_time), float(temp_scores[best_frame])))

        # Suppress nearby frames
        lo = max(0, best_frame - min_gap_frames)
        hi = min(len(temp_scores), best_frame + min_gap_frames)
        temp_scores[lo:hi] = -np.inf

    return results


def find_peak_energy_start(
    audio_path: str,
    clip_duration: float = 45.0,
    sr: int = 22050,
    hop_length: int = 512,
) -> Tuple[float, float]:
    """
    Find the highest-energy segment in a track — the region most likely to
    contain a chorus or beat drop.

    Uses the proven `find_best_segments` scoring (energy + onset density +
    spectral flux) to locate the single best clip-length window across the
    whole track, then returns its start time.

    Returns:
        (start_seconds, score)
    """
    segments = find_best_segments(
        audio_path,
        clip_duration=clip_duration,
        n_candidates=1,
        sr=sr,
        hop_length=hop_length,
        use_middle_crop=False,  # search the whole track, not just the middle
    )
    return segments[0]


def pick_best_aligned_segments(
    path_a: str,
    path_b: str,
    clip_duration: float = 45.0,
    n_candidates: int = 5,
    sr: int = 22050,
    analysis_window_s: float = 90.0,
) -> Tuple[float, float, float, float]:
    """
    Find start positions in both tracks that maximize combined energy.
    We prioritize 'high energy' pairs above all else.

    Returns:
        (start_a, score_a, start_b, score_b)
    """
    cands_a = find_best_segments(
        path_a,
        clip_duration=clip_duration,
        n_candidates=n_candidates,
        sr=sr,
        analysis_window_s=analysis_window_s,
        use_middle_crop=True,
    )
    cands_b = find_best_segments(
        path_b,
        clip_duration=clip_duration,
        n_candidates=n_candidates,
        sr=sr,
        analysis_window_s=analysis_window_s,
        use_middle_crop=True,
    )

    # Pick pair that maximizes combined score
    best_score = -1.0
    best_a, best_sa = cands_a[0]
    best_b, best_sb = cands_b[0]

    for (ta, sa) in cands_a:
        for (tb, sb) in cands_b:
            # We want both to be high energy. 
            # Simple average works well when the individual scores are already optimized for energy.
            combined = (sa + sb) / 2.0
            if combined > best_score:
                best_score = combined
                best_a, best_sa = ta, sa
                best_b, best_sb = tb, sb

    return best_a, best_sa, best_b, best_sb
