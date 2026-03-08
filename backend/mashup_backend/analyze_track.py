"""
analyze_track.py  (v2)

Extracts a rich MIR feature bundle from an audio file using librosa.

New in v2 vs v1:
  - Major/minor mode detection via Krumhansl-Schmuckler key profiles
    (replaces hardcoded mode=0 in compatibility.py)
  - Danceability proxy: beat strength regularity + tempo stability
  - Speechiness proxy: zero-crossing rate + MFCC variance (high ZCR + high
    variance → likely vocal/speech content)
  - Instrumentalness proxy: inverse of speechiness signal + harmonic ratio
  - Spectral contrast: tonal vs. percussive frequency content per subband
    (a well-researched complement to MFCC for genre-matching)
  - Tonnetz: tonal centroid features for richer harmonic description
  - Harmonic/percussive ratio: how drum-heavy vs. melodic a track is

All features use established MIR methods from the librosa paper
(McFee et al., SciPy 2015) and published MIR research.
No black-box external APIs are used.

Usage:
  python analyze_track.py path/to/song.mp3 --out features.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa


PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles (1990)
# These are empirically derived probe-tone ratings that correlate well with
# perceived key in music psychology research.
# Major profile: C major tonal hierarchy
# Minor profile: C minor tonal hierarchy
# ---------------------------------------------------------------------------
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    arr = np.asarray(x)
    if arr.size == 0:
        return float("nan")
    if arr.ndim == 0:
        return float(arr)
    return float(np.mean(arr))


def safe_list(x: np.ndarray, max_len: Optional[int] = None) -> List[float]:
    if max_len is not None:
        x = x[:max_len]
    return [safe_float(v) for v in x.tolist()]


def rms_to_db(rms: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    rms = np.maximum(rms, amin)
    return 20.0 * np.log10(rms / ref)


# ---------------------------------------------------------------------------
# Key + Mode detection via Krumhansl-Schmuckler profiles
# ---------------------------------------------------------------------------

def estimate_key_and_mode(chroma_mean: np.ndarray) -> Tuple[int, str, int, str]:
    """
    Estimate musical key and mode (major/minor) using the
    Krumhansl-Schmuckler key-finding algorithm (1990).

    Computes Pearson correlation between the chroma profile and all 24
    major/minor key profiles, picks the best match.

    Returns:
        (key_pc, key_name, mode, mode_name)
        mode: 0 = major, 1 = minor
    """
    chroma = np.array(chroma_mean)
    chroma = chroma / (np.sum(chroma) + 1e-9)

    best_corr = -2.0
    best_key = 0
    best_mode = 0

    for tonic in range(12):
        # Rotate profiles to this tonic
        major_prof = np.roll(KS_MAJOR, tonic)
        minor_prof = np.roll(KS_MINOR, tonic)

        corr_major = float(np.corrcoef(chroma, major_prof)[0, 1])
        corr_minor = float(np.corrcoef(chroma, minor_prof)[0, 1])

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = tonic
            best_mode = 0
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = tonic
            best_mode = 1

    mode_name = "major" if best_mode == 0 else "minor"
    return best_key, PITCH_CLASS_NAMES[best_key], best_mode, mode_name


# ---------------------------------------------------------------------------
# Danceability proxy
# Based on: beat strength regularity + tempo consistency
# Reference: Spotify defines danceability via "tempo, rhythm stability, beat
# strength, and overall regularity" — all computable from librosa beat data.
# ---------------------------------------------------------------------------

def compute_danceability(
    y: np.ndarray,
    sr: int,
    tempo: float,
    beat_frames: np.ndarray,
    hop_length: int = 512,
) -> float:
    """
    Proxy for danceability (0–1) based on:
      1. Beat regularity: how consistent the inter-beat intervals are
         (coefficient of variation of IBIs — lower CoV = more regular = more danceable)
      2. Beat strength: average onset strength at beat locations
      3. Tempo favorability: tracks near 90–140 BPM score higher (dance music range)

    Returns a float in [0, 1].
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Beat regularity via IBI coefficient of variation
    if len(beat_frames) >= 4:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        ibis = np.diff(beat_times)
        ibi_cv = float(np.std(ibis) / (np.mean(ibis) + 1e-9))
        # CoV near 0 = perfect metronome; CoV > 0.3 = irregular
        regularity = max(0.0, 1.0 - ibi_cv / 0.30)
    else:
        regularity = 0.3

    # Beat strength
    if len(beat_frames) > 0:
        valid_frames = beat_frames[beat_frames < len(onset_env)]
        if len(valid_frames) > 0:
            beat_strength = float(np.mean(onset_env[valid_frames]))
            beat_strength_norm = min(1.0, beat_strength / (np.mean(onset_env) * 2.0 + 1e-9))
        else:
            beat_strength_norm = 0.3
    else:
        beat_strength_norm = 0.3

    # Tempo favorability — 90–140 BPM is peak dance territory
    if 90 <= tempo <= 140:
        tempo_fav = 1.0
    elif 70 <= tempo < 90 or 140 < tempo <= 160:
        tempo_fav = 0.75
    elif 60 <= tempo < 70 or 160 < tempo <= 180:
        tempo_fav = 0.50
    else:
        tempo_fav = 0.25

    danceability = 0.45 * regularity + 0.35 * beat_strength_norm + 0.20 * tempo_fav
    return float(np.clip(danceability, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Speechiness proxy
# High zero-crossing rate (ZCR) + high MFCC variance → vocal/speech content
# Reference: ZCR is widely used as a voiced/unvoiced discriminator in speech
# processing. MFCC variance captures the rapid timbral change of speech.
# ---------------------------------------------------------------------------

def compute_speechiness(
    y: np.ndarray,
    sr: int,
    mfcc: np.ndarray,
    hop_length: int = 512,
) -> float:
    """
    Proxy for spoken-word content (0–1).
    Instruments have low ZCR variance; speech has characteristically high
    and rapidly varying ZCR. Combined with MFCC delta variance.
    """
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    # Normalize ZCR mean — 0.1+ is typical for speech, < 0.05 for instruments
    zcr_mean = float(np.mean(zcr))
    zcr_score = min(1.0, zcr_mean / 0.12)

    # MFCC delta variance: speech changes rapidly, instruments more smoothly
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_var = float(np.mean(np.var(mfcc_delta, axis=1)))
    # Normalize empirically — typical range 0–500
    mfcc_var_score = min(1.0, mfcc_delta_var / 300.0)

    speechiness = 0.55 * zcr_score + 0.45 * mfcc_var_score
    return float(np.clip(speechiness, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Instrumentalness proxy
# Harmonic-to-percussive energy ratio — more harmonic = more instrumental feel
# Reference: librosa.effects.hpss separates harmonic/percussive components
# ---------------------------------------------------------------------------

def compute_instrumentalness(
    y: np.ndarray,
    speechiness: float,
) -> float:
    """
    Proxy for absence of vocals (0–1, higher = more instrumental).
    Uses harmonic/percussive separation + inverse speechiness.
    """
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    harm_energy = float(np.mean(y_harmonic ** 2))
    perc_energy = float(np.mean(y_percussive ** 2))
    total_energy = harm_energy + perc_energy + 1e-12

    # High harmonic ratio alone doesn't mean instrumental (vocals are harmonic too),
    # but combined with low speechiness it's a reasonable proxy.
    harmonic_ratio = harm_energy / total_energy

    # Invert speechiness: low speech → likely instrumental
    instrumentalness = 0.50 * harmonic_ratio + 0.50 * (1.0 - speechiness)
    return float(np.clip(instrumentalness, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Spectral contrast
# Measures the difference between spectral peaks and valleys in frequency
# subbands — captures tonal vs. noisy texture differences between tracks.
# Reference: Jiang et al. (2002) "Music type classification by spectral contrast feature"
# ---------------------------------------------------------------------------

def compute_spectral_contrast(
    y: np.ndarray,
    sr: int,
    n_bands: int = 6,
    hop_length: int = 512,
) -> List[float]:
    """
    Returns mean spectral contrast across n_bands+1 subbands.
    High contrast = strong tonal peaks (pitched instruments, clean production).
    Low contrast = dense/noisy spectrum (distorted guitars, heavy compression).
    """
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_bands=n_bands, hop_length=hop_length
    )
    return safe_list(np.mean(contrast, axis=1))


# ---------------------------------------------------------------------------
# Tonnetz (tonal centroid features)
# 6-dimensional representation of harmonic relations on the Tonnetz grid.
# Captures tonal tension/resolution patterns well beyond chroma.
# Reference: Harte et al. (2006) "Detecting Harmonic Change in Musical Audio"
# ---------------------------------------------------------------------------

def compute_tonnetz(y: np.ndarray, sr: int) -> List[float]:
    """
    Returns mean Tonnetz feature vector (6 dimensions).
    Useful for detecting harmonic compatibility beyond simple key matching.
    """
    y_harm, _ = librosa.effects.hpss(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    return safe_list(np.mean(tonnetz, axis=1))


# ---------------------------------------------------------------------------
# Harmonic/percussive ratio
# ---------------------------------------------------------------------------

def compute_hp_ratio(y: np.ndarray) -> float:
    """
    Returns harmonic energy / total energy.
    Useful for matching tracks by feel: both tracks should have similar
    harmonic-to-percussive balance for a cohesive mashup.
    """
    y_h, y_p = librosa.effects.hpss(y)
    e_h = float(np.mean(y_h ** 2))
    e_p = float(np.mean(y_p ** 2))
    return float(e_h / (e_h + e_p + 1e-12))


# ---------------------------------------------------------------------------
# Main feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackFeatures:
    path: str
    sr: int
    duration_s: float

    # Rhythm
    tempo_bpm: float
    beat_times_s: List[float]
    danceability: float

    # Harmony
    chroma_mean: List[float]
    key_pc_guess: int
    key_name_guess: str
    mode: int            # 0 = major, 1 = minor  (Krumhansl-Schmuckler)
    mode_name: str       # "major" or "minor"
    tonnetz_mean: List[float]

    # Timbre
    mfcc_mean: List[float]
    mfcc_std: List[float]
    spectral_contrast_mean: List[float]
    spectral_centroid_hz: float

    # Energy / loudness
    rms_mean: float
    rms_std: float
    loudness_db_mean: float
    loudness_db_std: float
    rms_curve: List[float]
    loudness_db_curve: List[float]

    # Vocal content
    speechiness: float
    instrumentalness: float

    # Texture
    hp_ratio: float     # harmonic / (harmonic + percussive)


def analyze_mp3(
    path: str,
    target_sr: int = 22050,
    mono: bool = True,
    hop_length: int = 512,
    n_mfcc: int = 20,
    curve_max_points: int = 600,
    analysis_window_s: float = 90.0,
    use_middle_window: bool = True,
    fast_mode: bool = True,
) -> TrackFeatures:
    full_duration_s = float(librosa.get_duration(path=path))

    offset_s = 0.0
    load_duration_s: Optional[float] = None
    if use_middle_window and analysis_window_s > 0 and full_duration_s > analysis_window_s:
        offset_s = max(0.0, (full_duration_s - analysis_window_s) * 0.5)
        load_duration_s = analysis_window_s

    y, sr = librosa.load(
        path,
        sr=target_sr,
        mono=mono,
        offset=offset_s,
        duration=load_duration_s,
    )
    duration_s = full_duration_s

    # Rhythm
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo_bpm = safe_float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length) + offset_s

    # Harmony
    if fast_mode:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=2048)
    else:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-9)

    key_pc, key_name, mode, mode_name = estimate_key_and_mode(chroma_norm)
    if fast_mode:
        tonnetz_mean = [0.0] * 6
        mfcc_mean = np.zeros(n_mfcc, dtype=float)
        mfcc_std = np.zeros(n_mfcc, dtype=float)
        spectral_contrast_mean = [0.0] * 7
        spectral_centroid_hz = float(
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0])
        )
    else:
        tonnetz_mean = compute_tonnetz(y, sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        spectral_contrast_mean = compute_spectral_contrast(y, sr, hop_length=hop_length)
        spectral_centroid_hz = float(
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0])
        )

    # Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = rms_to_db(rms)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    loud_db_mean = float(np.mean(rms_db))
    loud_db_std = float(np.std(rms_db))

    def downsample(arr, max_pts):
        if len(arr) <= max_pts:
            return arr
        idx = np.linspace(0, len(arr) - 1, max_pts).astype(int)
        return arr[idx]

    # Vocal / texture features
    if fast_mode:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        beat_strength = 0.0
        if len(beat_frames) > 0 and len(onset_env) > 0:
            valid_frames = beat_frames[beat_frames < len(onset_env)]
            if len(valid_frames) > 0:
                beat_strength = float(np.mean(onset_env[valid_frames]) / (np.mean(onset_env) + 1e-9))
        danceability = float(np.clip(beat_strength, 0.0, 1.0))
        speechiness = 0.0
        instrumentalness = 0.0
        hp_ratio = 0.0
    else:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
        speechiness = compute_speechiness(y, sr, mfcc, hop_length)
        instrumentalness = compute_instrumentalness(y, speechiness)
        hp_ratio = compute_hp_ratio(y)
        danceability = compute_danceability(y, sr, tempo_bpm, beat_frames, hop_length)

    return TrackFeatures(
        path=path,
        sr=sr,
        duration_s=duration_s,
        tempo_bpm=tempo_bpm,
        beat_times_s=safe_list(beat_times),
        danceability=danceability,
        chroma_mean=safe_list(chroma_norm),
        key_pc_guess=key_pc,
        key_name_guess=key_name,
        mode=mode,
        mode_name=mode_name,
        tonnetz_mean=tonnetz_mean,
        mfcc_mean=safe_list(mfcc_mean),
        mfcc_std=safe_list(mfcc_std),
        spectral_contrast_mean=spectral_contrast_mean,
        spectral_centroid_hz=spectral_centroid_hz,
        rms_mean=rms_mean,
        rms_std=rms_std,
        loudness_db_mean=loud_db_mean,
        loudness_db_std=loud_db_std,
        rms_curve=safe_list(downsample(rms, curve_max_points)),
        loudness_db_curve=safe_list(downsample(rms_db, curve_max_points)),
        speechiness=speechiness,
        instrumentalness=instrumentalness,
        hp_ratio=hp_ratio,
    )


def as_json_dict(features: TrackFeatures) -> Dict[str, Any]:
    return {
        "path": features.path,
        "sr": features.sr,
        "duration_s": features.duration_s,
        "rhythm": {
            "tempo_bpm": features.tempo_bpm,
            "beat_times_s": features.beat_times_s,
            "danceability": features.danceability,
        },
        "harmony": {
            "chroma_mean": features.chroma_mean,
            "key_pc_guess": features.key_pc_guess,
            "key_name_guess": features.key_name_guess,
            "mode": features.mode,
            "mode_name": features.mode_name,
            "tonnetz_mean": features.tonnetz_mean,
        },
        "timbre": {
            "mfcc_mean": features.mfcc_mean,
            "mfcc_std": features.mfcc_std,
            "spectral_contrast_mean": features.spectral_contrast_mean,
            "spectral_centroid_hz": features.spectral_centroid_hz,
        },
        "energy": {
            "rms_mean": features.rms_mean,
            "rms_std": features.rms_std,
            "loudness_db_mean": features.loudness_db_mean,
            "loudness_db_std": features.loudness_db_std,
            "rms_curve": features.rms_curve,
            "loudness_db_curve": features.loudness_db_curve,
        },
        "vocal_content": {
            "speechiness": features.speechiness,
            "instrumentalness": features.instrumentalness,
        },
        "texture": {
            "hp_ratio": features.hp_ratio,
        },
    }


def from_json_dict(payload: Dict[str, Any]) -> TrackFeatures:
    rhythm = payload.get("rhythm", {})
    harmony = payload.get("harmony", {})
    timbre = payload.get("timbre", {})
    energy = payload.get("energy", {})
    vocal = payload.get("vocal_content", {})
    texture = payload.get("texture", {})
    return TrackFeatures(
        path=str(payload.get("path", "")),
        sr=int(payload.get("sr", 22050)),
        duration_s=float(payload.get("duration_s", 0.0)),
        tempo_bpm=safe_float(rhythm.get("tempo_bpm")),
        beat_times_s=[safe_float(v) for v in rhythm.get("beat_times_s", [])],
        danceability=safe_float(rhythm.get("danceability")),
        chroma_mean=[safe_float(v) for v in harmony.get("chroma_mean", [])],
        key_pc_guess=int(harmony.get("key_pc_guess", 0)),
        key_name_guess=str(harmony.get("key_name_guess", "C")),
        mode=int(harmony.get("mode", 0)),
        mode_name=str(harmony.get("mode_name", "major")),
        tonnetz_mean=[safe_float(v) for v in harmony.get("tonnetz_mean", [])],
        mfcc_mean=[safe_float(v) for v in timbre.get("mfcc_mean", [])],
        mfcc_std=[safe_float(v) for v in timbre.get("mfcc_std", [])],
        spectral_contrast_mean=[safe_float(v) for v in timbre.get("spectral_contrast_mean", [])],
        spectral_centroid_hz=safe_float(timbre.get("spectral_centroid_hz", 0.0)),
        rms_mean=safe_float(energy.get("rms_mean")),
        rms_std=safe_float(energy.get("rms_std")),
        loudness_db_mean=safe_float(energy.get("loudness_db_mean")),
        loudness_db_std=safe_float(energy.get("loudness_db_std")),
        rms_curve=[safe_float(v) for v in energy.get("rms_curve", [])],
        loudness_db_curve=[safe_float(v) for v in energy.get("loudness_db_curve", [])],
        speechiness=safe_float(vocal.get("speechiness")),
        instrumentalness=safe_float(vocal.get("instrumentalness")),
        hp_ratio=safe_float(texture.get("hp_ratio")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an audio file and export MIR features to JSON.")
    parser.add_argument("mp3_path", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--mfcc", type=int, default=20)
    args = parser.parse_args()

    feats = analyze_mp3(args.mp3_path, target_sr=args.sr, hop_length=args.hop, n_mfcc=args.mfcc)
    payload = as_json_dict(feats)
    text = json.dumps(payload, indent=2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote JSON features to: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
