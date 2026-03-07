from __future__ import annotations

from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str, sr: int = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return y, sr


def trim_audio(y: np.ndarray, sr: int, start_s: float, duration_s: float) -> np.ndarray:
    start_idx = int(start_s * sr)
    end_idx = int((start_s + duration_s) * sr)
    start_idx = max(0, start_idx)
    end_idx = min(len(y), end_idx)
    return y[start_idx:end_idx]


def time_stretch_audio(y: np.ndarray, stretch_factor: float) -> np.ndarray:
    if abs(stretch_factor - 1.0) < 1e-6:
        return y
    return librosa.effects.time_stretch(y, rate=stretch_factor)


def pitch_shift_audio(y: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    if semitones == 0:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def apply_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    gain_linear = 10 ** (gain_db / 20.0)
    return y * gain_linear


def pad_or_trim_to_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        return y[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(y)] = y
    return out


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_val = np.max(np.abs(y)) if len(y) > 0 else 0.0
    if max_val < 1e-9:
        return y
    return y * (peak / max_val)


def overlay_audio(y_a: np.ndarray, y_b: np.ndarray) -> np.ndarray:
    target_len = max(len(y_a), len(y_b))
    y_a = pad_or_trim_to_length(y_a, target_len)
    y_b = pad_or_trim_to_length(y_b, target_len)
    mixed = y_a + y_b
    return normalize_peak(mixed)


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
    sr: int = 22050,
) -> Tuple[np.ndarray, int]:
    y_a, sr_a = load_audio(file_a, sr=sr)
    y_b, sr_b = load_audio(file_b, sr=sr)

    if sr_a != sr_b:
        raise ValueError("Sample rates do not match after loading.")

    clip_a = trim_audio(y_a, sr, start_a, duration_s)
    clip_b = trim_audio(y_b, sr, start_b, duration_s)

    clip_b = time_stretch_audio(clip_b, stretch_factor_b)
    clip_b = pitch_shift_audio(clip_b, sr, pitch_shift_b)

    clip_a = apply_gain_db(clip_a, gain_db_a)
    clip_b = apply_gain_db(clip_b, gain_db_b)

    mixed = overlay_audio(clip_a, clip_b)
    return mixed, sr


def save_mashup(y: np.ndarray, sr: int, wav_out: str, mp3_out: Optional[str] = None) -> None:
    export_wav(y, sr, wav_out)
    if mp3_out is not None:
        export_mp3_via_pydub(wav_out, mp3_out)
