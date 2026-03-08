from __future__ import annotations

from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


DEFAULT_SR = 44100


def load_audio(path: str, sr: int = DEFAULT_SR, mono: bool = True) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return y.astype(np.float32), sr


def trim_audio(y: np.ndarray, sr: int, start_s: float, duration_s: float) -> np.ndarray:
    start_idx = int(start_s * sr)
    end_idx = int((start_s + duration_s) * sr)
    start_idx = max(0, start_idx)
    end_idx = min(len(y), end_idx)
    return y[start_idx:end_idx].astype(np.float32)


def time_stretch_audio(y: np.ndarray, stretch_factor: float) -> np.ndarray:
    if len(y) == 0 or abs(stretch_factor - 1.0) < 1e-6:
        return y.astype(np.float32)
    return librosa.effects.time_stretch(y, rate=stretch_factor).astype(np.float32)


def pitch_shift_audio(y: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    if len(y) == 0 or semitones == 0:
        return y.astype(np.float32)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32)


def apply_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    gain_linear = 10 ** (gain_db / 20.0)
    return (y * gain_linear).astype(np.float32)


def pad_or_trim_to_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len:
        return y.astype(np.float32)
    if len(y) > target_len:
        return y[:target_len].astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(y)] = y
    return out


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_val = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    if max_val < 1e-9:
        return y.astype(np.float32)
    return (y * (peak / max_val)).astype(np.float32)


def trim_leading_silence(y: np.ndarray, sr: int, top_db: float = 35.0) -> Tuple[np.ndarray, int]:
    """
    Returns (trimmed_audio, offset_samples).
    """
    if len(y) == 0:
        return y.astype(np.float32), 0
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y.astype(np.float32), 0
    start = int(intervals[0][0])
    return y[start:].astype(np.float32), start


def shift_audio(y: np.ndarray, offset_samples: int, target_len: Optional[int] = None) -> np.ndarray:
    """
    Positive offset -> delay by inserting silence.
    Negative offset -> trim from the front.
    """
    if offset_samples > 0:
        y = np.concatenate([np.zeros(offset_samples, dtype=np.float32), y.astype(np.float32)])
    elif offset_samples < 0:
        trim = min(-offset_samples, max(0, len(y) - 1))
        y = y[trim:].astype(np.float32)
    else:
        y = y.astype(np.float32)

    if target_len is not None:
        y = pad_or_trim_to_length(y, target_len)
    return y.astype(np.float32)


def soft_limit(y: np.ndarray, drive: float = 1.1) -> np.ndarray:
    if len(y) == 0:
        return y.astype(np.float32)
    return np.tanh(y * drive).astype(np.float32)


def high_pass_filter(y: np.ndarray, sr: int, cutoff: float = 140.0) -> np.ndarray:
    """Light cleanup for vocal stems; removes sub-bass bleed."""
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * sr
    normal_cutoff = min(0.99, cutoff / max(nyq, 1e-9))
    b, a = butter(2, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, y).astype(np.float32)


def overlay_audio(
    y_a: np.ndarray,
    y_b: np.ndarray,
    gain_db_a: float = -6.0,
    gain_db_b: float = -6.0,
    peak: float = 0.92,
) -> np.ndarray:
    """
    Conservative summing for mashups:
    - pad to same length
    - apply fixed gains before summing
    - soft-limit the result
    - then peak-normalize once at the end
    """
    target_len = max(len(y_a), len(y_b))
    y_a = apply_gain_db(pad_or_trim_to_length(y_a, target_len), gain_db_a)
    y_b = apply_gain_db(pad_or_trim_to_length(y_b, target_len), gain_db_b)
    mixed = soft_limit(y_a + y_b)
    return normalize_peak(mixed, peak=peak)


def export_wav(y: np.ndarray, sr: int, out_path: str) -> None:
    sf.write(out_path, y, sr)


def export_mp3_via_pydub(wav_path: str, mp3_path: str, bitrate: str = "192k") -> None:
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3", bitrate=bitrate)


def render_mashup_preview(
    file_a: str,
    file_b: str,
    start_a: float,
    start_b: float,
    duration_s: float,
    stretch_factor_b: float,
    pitch_shift_b: int,
    gain_db_a: float,
    gain_db_b: float,
    sr: int = DEFAULT_SR,
) -> Tuple[np.ndarray, int]:
    y_a, sr_a = load_audio(file_a, sr=sr)
    y_b, sr_b = load_audio(file_b, sr=sr)

    if sr_a != sr_b:
        raise ValueError("Sample rates do not match after loading.")

    clip_a = trim_audio(y_a, sr, start_a, duration_s)
    clip_b = trim_audio(y_b, sr, start_b, duration_s)
    clip_b = time_stretch_audio(clip_b, stretch_factor_b)
    clip_b = pitch_shift_audio(clip_b, sr, pitch_shift_b)

    mixed = overlay_audio(clip_a, clip_b, gain_db_a=gain_db_a, gain_db_b=gain_db_b)
    return mixed, sr


def save_mashup(y: np.ndarray, sr: int, wav_out: str, mp3_out: Optional[str] = None) -> None:
    export_wav(y, sr, wav_out)
    if mp3_out is not None:
        export_mp3_via_pydub(wav_out, mp3_out)
