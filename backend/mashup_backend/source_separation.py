"""
source_separation.py

Separates a track into stems (vocals, drums, bass, other/melody) using
Facebook's Demucs model. This enables proper mashup combinations like:
  - Vocals from Song A + Instrumental (drums+bass+other) from Song B
  - Both instrumentals layered under Song A's vocals
  - Custom stem blends

Demucs docs: https://github.com/facebookresearch/demucs

Install:
    pip install demucs

Usage:
    from source_separation import separate_track, StemBundle, blend_stems
    stems = separate_track("song.mp3", out_dir="stems/song/")
    # stems.vocals, stems.drums, stems.bass, stems.other → np.ndarray

    instrumental = blend_stems(stems, include=["drums", "bass", "other"])
    vocals_only  = blend_stems(stems, include=["vocals"])
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

_DEMUCS_STACK_OK: Optional[bool] = None


def _demucs_env() -> dict:
    """Return an env dict that ensures ffmpeg is on PATH for Demucs subprocesses."""
    env = os.environ.copy()
    # WinGet installs ffmpeg here; the system PATH may not be refreshed yet.
    extra = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links")
    if os.path.isdir(extra) and extra not in env.get("PATH", ""):
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
    return env


@dataclass
class StemBundle:
    """Holds separated stems for a single track (all same sr)."""
    vocals: np.ndarray
    drums: np.ndarray
    bass: np.ndarray
    other: np.ndarray
    sr: int
    source_path: str

    def get(self, name: str) -> np.ndarray:
        return getattr(self, name)


_demucs_model_cache = {}


def _get_demucs_model(model_name: str, device: str):
    """Load and cache the Demucs model."""
    key = (model_name, device)
    if key not in _demucs_model_cache:
        from demucs.pretrained import get_model

        m = get_model(model_name)
        m.to(device)
        _demucs_model_cache[key] = m
    return _demucs_model_cache[key]


def _run_demucs_inprocess(
    audio_path: str,
    out_dir: str,
    model: str = "htdemucs",
    device: str = "cpu",
    two_stem: Optional[str] = None,
) -> Path:
    """
    Run Demucs separation in-process using the Python API.
    Saves stems with soundfile to avoid torchaudio.save / torchcodec issues.
    """
    import librosa as _lr

    demucs_model = _get_demucs_model(model, device)

    # Load audio at the model's expected sample rate
    y, _ = _lr.load(audio_path, sr=demucs_model.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])  # mono → stereo
    # Demucs expects (batch, channels, samples)
    wav = torch.from_numpy(y).float().unsqueeze(0).to(device)

    from demucs.apply import apply_model

    sources = apply_model(demucs_model, wav, device=device)
    # sources shape: (batch, n_sources, channels, samples)

    # apply_model may pad the output; trim back to the original length
    original_len = y.shape[-1]
    sources = sources[..., :original_len]

    track_name = Path(audio_path).stem
    stem_dir = Path(out_dir) / model / track_name
    os.makedirs(stem_dir, exist_ok=True)

    source_names = demucs_model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
    for i, name in enumerate(source_names):
        stem_audio = sources[0, i].cpu().numpy()  # (channels, samples)
        # Save as mono WAV using soundfile
        mono = np.mean(stem_audio, axis=0)
        sf.write(str(stem_dir / f"{name}.wav"), mono, demucs_model.samplerate)

    return stem_dir


def _run_demucs(
    audio_path: str,
    out_dir: str,
    model: str = "htdemucs",
    device: str = "cpu",
) -> Path:
    """Run Demucs in 2-stem mode (vocals / no_vocals)."""
    return _run_demucs_inprocess(audio_path, out_dir, model, device, two_stem="vocals")


def _run_demucs_4stem(
    audio_path: str,
    out_dir: str,
    model: str = "htdemucs",
    device: str = "cpu",
) -> Path:
    """Run Demucs in full 4-stem mode: vocals, drums, bass, other."""
    return _run_demucs_inprocess(audio_path, out_dir, model, device)


def _load_stem(stem_path: Path, name: str, sr: int) -> np.ndarray:
    """Load a stem file (wav/mp3/flac); return mono float32 array."""
    import librosa
    for ext in ("wav", "mp3", "flac"):
        stem_file = stem_path / f"{name}.{ext}"
        if stem_file.exists():
            y, _ = librosa.load(str(stem_file), sr=sr, mono=True)
            return y
    return np.zeros(sr, dtype=np.float32)


def _fast_fallback_split(audio_path: str, sr: int) -> StemBundle:
    """
    Fallback when Demucs cannot run in the local environment.
    This is not true source separation, but keeps preview generation working.
    """
    import librosa

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    if len(y) == 0:
        z = np.zeros(sr, dtype=np.float32)
        return StemBundle(vocals=z, drums=z, bass=z, other=z, sr=sr, source_path=audio_path)

    harmonic, percussive = librosa.effects.hpss(y)
    # This is a very poor approximation of vocal separation.
    vocals = harmonic.astype(np.float32)
    no_vocals = (0.80 * percussive + 0.20 * (y - vocals)).astype(np.float32)
    bass = np.zeros_like(no_vocals)
    other = np.zeros_like(no_vocals)
    return StemBundle(
        vocals=vocals,
        drums=no_vocals,
        bass=bass,
        other=other,
        sr=sr,
        source_path=audio_path,
    )


def _demucs_stack_ok() -> bool:
    """
    Fast preflight for environments where Demucs fails at save time
    because torchaudio runtime libraries are missing.
    """
    global _DEMUCS_STACK_OK
    if _DEMUCS_STACK_OK is not None:
        return _DEMUCS_STACK_OK
    try:
        import demucs  # noqa: F401
        import torchaudio  # noqa: F401
        _DEMUCS_STACK_OK = True
    except Exception:
        _DEMUCS_STACK_OK = False
    return _DEMUCS_STACK_OK


def separate_track(
    audio_path: str,
    out_dir: Optional[str] = None,
    model: str = "htdemucs",
    device: Optional[str] = None,
    sr: int = 44100,
    four_stem: bool = True,
    allow_fallback: bool = True,
) -> StemBundle:
    """
    Separate `audio_path` into stems using Demucs.

    Args:
        audio_path:  Input MP3/WAV path.
        out_dir:     Where to write separated WAVs. Uses a temp dir if None.
        model:       Demucs model name. htdemucs is recommended.
        device:      'cpu' or 'cuda'. If None, uses cuda if available.
        sr:          Output sample rate.
        four_stem:   If True, separate into vocals/drums/bass/other.
                     If False, use fast 2-stem (vocals / no_vocals).
        allow_fallback: If True, falls back to fast HPSS split when Demucs fails.

    Returns:
        StemBundle with numpy arrays for each stem.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cleanup = False
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="mashup_stems_")
        cleanup = True

    if allow_fallback and not _demucs_stack_ok():
        warnings.warn(
            "Demucs runtime stack is unavailable (torchaudio issue). "
            "Using fast fallback split.",
            RuntimeWarning,
        )
        return _fast_fallback_split(audio_path, sr)

    try:
        if four_stem:
            stem_dir = _run_demucs_4stem(audio_path, out_dir, model, device)
        else:
            stem_dir = _run_demucs(audio_path, out_dir, model, device)

        vocals = _load_stem(stem_dir, "vocals", sr)
        drums  = _load_stem(stem_dir, "drums", sr)
        bass   = _load_stem(stem_dir, "bass", sr)
        other  = _load_stem(stem_dir, "other", sr)

        # In 2-stem mode, no_vocals is stored as "no_vocals"
        if not four_stem:
            no_vocals_file = None
            for ext in ("wav", "mp3", "flac"):
                p = stem_dir / f"no_vocals.{ext}"
                if p.exists():
                    no_vocals_file = p
                    break
            if no_vocals_file is not None:
                import librosa
                nv, _ = librosa.load(str(no_vocals_file), sr=sr, mono=True)
                drums = nv
                bass = np.zeros_like(nv)
                other = np.zeros_like(nv)

        return StemBundle(
            vocals=vocals,
            drums=drums,
            bass=bass,
            other=other,
            sr=sr,
            source_path=audio_path,
        )
    except Exception as e:
        if allow_fallback:
            warnings.warn(
                f"Demucs unavailable/failed for {audio_path}. "
                f"Falling back to fast split. Original error: {e}",
                RuntimeWarning,
            )
            return _fast_fallback_split(audio_path, sr)
        raise RuntimeError(f"Stem separation failed for {audio_path}: {e}") from e


def blend_stems(
    bundle: StemBundle,
    include: List[str],
    weights: Optional[dict] = None,
    normalize: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Mix selected stems from a StemBundle into one audio array.

    Args:
        bundle:   StemBundle from separate_track().
        include:  Which stems to include: any of ["vocals","drums","bass","other"].
        weights:  Optional per-stem gain multipliers, e.g. {"vocals": 1.2}.  
        normalize: Whether to normalize the resulting mix to 0.95 peak.

    Returns:
        (mixed_audio, sr)
    """
    arrays = []
    for name in include:
        stem = bundle.get(name).copy()
        if weights and name in weights:
            stem *= weights[name]
        arrays.append(stem)

    if not arrays:
        return np.zeros(bundle.sr, dtype=np.float32), bundle.sr

    # Pad all to same length
    max_len = max(len(a) for a in arrays)
    padded = []
    for a in arrays:
        if len(a) < max_len:
            a = np.concatenate([a, np.zeros(max_len - len(a), dtype=np.float32)])
        padded.append(a)

    mixed = np.sum(padded, axis=0)

    if normalize:
        peak = np.max(np.abs(mixed))
        if peak > 1e-9:
            mixed = mixed * (0.95 / peak)

    return mixed.astype(np.float32), bundle.sr

def is_demucs_available() -> bool:
    """Check whether demucs is installed and runnable."""
    result = subprocess.run(
        [sys.executable, "-m", "demucs", "--help"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0
