"""
api.py

FastAPI backend for the mashup pipeline.

Endpoints:
  POST /upload            — upload two audio files, get session_id back
  POST /analyze/{sid}     — run compatibility analysis, return scores
  POST /preview/{sid}     — render the mashup clip, return audio URL
  GET  /audio/{sid}       — stream/download the rendered mashup WAV
  GET  /download/{sid}    — download as MP3
  DELETE /session/{sid}   — clean up session files

Run:
    uvicorn api:app --reload --port 8000

Frontend calls:
    1. POST /upload  with form-data files track_a, track_b  → {session_id}
    2. POST /analyze/{session_id}                           → {compatibility, segments, ...}
    3. POST /preview/{session_id} with optional JSON body   → {preview_url, ...}
    4. GET  /audio/{session_id}                             → WAV stream
    5. GET  /download/{session_id}                          → MP3 download
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from analyze_track import analyze_mp3, as_json_dict
from compatibility import compare_tracks
from segment_finder import pick_best_aligned_segments, find_best_segments
from mashup_engine import MashupEngine, MashupConfig


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Mashup API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_SESSIONS_DIR = Path(__file__).resolve().parent / ".sessions"
SESSIONS_DIR = Path(os.environ.get("MASHUP_SESSIONS_DIR", str(DEFAULT_SESSIONS_DIR)))
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    # Keep backend errors visible to frontend during development.
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    clip_duration: float = 45.0
    start_a: Optional[float] = None
    start_b: Optional[float] = None
    mashup_mode: str = "auto"           # auto | full_blend | vocals_a_inst_b | inst_a_vocals_b
    use_stem_separation: bool = False
    gain_db_a: float = -1.0
    gain_db_b: Optional[float] = None


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def session_dir(sid: str) -> Path:
    d = SESSIONS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def require_session(sid: str) -> Path:
    d = SESSIONS_DIR / sid
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found.")
    return d


def session_file(sid: str, name: str) -> Path:
    return require_session(sid) / name


def save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/upload")
@app.post("/api/upload")
async def upload_tracks(
    track_a: UploadFile = File(..., description="First track (MP3/WAV)"),
    track_b: UploadFile = File(..., description="Second track (MP3/WAV)"),
) -> Dict[str, Any]:
    """
    Upload two audio tracks and create a session.
    Returns a session_id used by all subsequent calls.
    """
    sid = str(uuid.uuid4())
    sdir = session_dir(sid)

    ext_a = Path(track_a.filename or "track_a.mp3").suffix or ".mp3"
    ext_b = Path(track_b.filename or "track_b.mp3").suffix or ".mp3"

    path_a = sdir / f"track_a{ext_a}"
    path_b = sdir / f"track_b{ext_b}"

    save_upload(track_a, path_a)
    save_upload(track_b, path_b)

    # Save metadata
    meta = {
        "session_id": sid,
        "track_a": str(path_a),
        "track_b": str(path_b),
        "track_a_name": track_a.filename,
        "track_b_name": track_b.filename,
    }
    import json
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "session_id": sid,
        "track_a": track_a.filename,
        "track_b": track_b.filename,
        "message": "Tracks uploaded. Call POST /analyze/{session_id} next.",
    }


@app.post("/analyze/{session_id}")
@app.post("/api/analyze/{session_id}")
async def analyze_session(session_id: str) -> Dict[str, Any]:
    """
    Run full analysis on both uploaded tracks:
      - BPM, key, energy, timbre extraction
      - Compatibility scoring with grade and summary
      - Top candidate clip segments for each track
    """
    import json
    sdir = require_session(session_id)
    meta = json.loads((sdir / "meta.json").read_text())

    path_a = meta["track_a"]
    path_b = meta["track_b"]

    feats_a = analyze_mp3(path_a)
    feats_b = analyze_mp3(path_b)
    comp = compare_tracks(feats_a, feats_b)

    # Find top 3 candidate start times for each track
    cands_a = find_best_segments(path_a, clip_duration=45.0, n_candidates=3)
    cands_b = find_best_segments(path_b, clip_duration=45.0, n_candidates=3)

    def score_value(name: str, default: float = 0.0) -> float:
        value = getattr(comp, name, default)
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return round(float(default), 3)

    spectral_score = score_value("spectral_contrast_score")
    timbre_score = score_value("timbre_score", spectral_score)

    result = {
        "session_id": session_id,
        "compatibility": {
            "score": round(comp.compatibility_score, 1),
            "grade": comp.grade,
            "summary": comp.summary,
            "mashup_type": comp.mashup_type,
            "details": {
                "tempo_score": score_value("tempo_score"),
                "key_score": score_value("key_score"),
                "energy_score": score_value("energy_score"),
                "loudness_score": score_value("loudness_score"),
                "timbre_score": timbre_score,
                "spectral_contrast_score": spectral_score,
                "tonnetz_score": score_value("tonnetz_score"),
                "danceability_match_score": score_value("danceability_match_score"),
            },
            "adjustments": {
                "stretch_factor_b": round(comp.stretch_factor_b, 4),
                "stretch_pct_b": round(comp.stretch_pct_b, 2),
                "pitch_shift_b_semitones": comp.pitch_shift_b,
                "gain_db_b": round(comp.gain_db_b, 2),
            },
        },
        "track_a": {
            "name": meta.get("track_a_name", "track_a"),
            "features": as_json_dict(feats_a),
            "candidate_segments": [
                {"start_s": round(t, 2), "quality_score": round(s, 3)}
                for t, s in cands_a
            ],
        },
        "track_b": {
            "name": meta.get("track_b_name", "track_b"),
            "features": as_json_dict(feats_b),
            "candidate_segments": [
                {"start_s": round(t, 2), "quality_score": round(s, 3)}
                for t, s in cands_b
            ],
        },
    }

    # Cache analysis
    (sdir / "analysis.json").write_text(json.dumps(result, indent=2))

    return result


@app.post("/preview/{session_id}")
@app.post("/api/preview/{session_id}")
async def render_preview(
    session_id: str,
    req: PreviewRequest = PreviewRequest(),
) -> Dict[str, Any]:
    """
    Render a mashup preview clip using the analyzed parameters.
    Optionally override start times and blend mode.

    Returns preview metadata and audio URLs.
    """
    import json
    sdir = require_session(session_id)
    meta = json.loads((sdir / "meta.json").read_text())

    wav_out = str(sdir / "preview.wav")
    # MP3 export needs ffmpeg. If not available, we still render WAV successfully.
    mp3_out = str(sdir / "preview.mp3") if shutil.which("ffmpeg") else None

    config = MashupConfig(
        track_a=meta["track_a"],
        track_b=meta["track_b"],
        clip_duration=req.clip_duration,
        start_a=req.start_a,
        start_b=req.start_b,
        mashup_mode=req.mashup_mode,
        use_stem_separation=req.use_stem_separation,
        gain_db_a=req.gain_db_a,
        gain_db_b=req.gain_db_b,
        wav_out=wav_out,
        mp3_out=mp3_out,
        stems_dir=str(sdir / "stems"),
    )

    engine = MashupEngine()
    result = engine.run(config)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    return {
        "session_id": session_id,
        "success": True,
        "summary": result.summary,
        "mashup_mode_used": result.mashup_mode_used,
        "clip": {
            "start_a_s": round(result.start_a, 2),
            "start_b_s": round(result.start_b, 2),
            "duration_s": req.clip_duration,
            "segment_quality_a": round(result.segment_score_a, 3),
            "segment_quality_b": round(result.segment_score_b, 3),
        },
        "compatibility": {
            "score": round(result.compatibility.compatibility_score, 1),
            "grade": result.compatibility.grade,
        },
        "audio_url": f"/api/audio/{session_id}",
        "download_url": f"/api/download/{session_id}",
    }


@app.get("/audio/{session_id}")
@app.get("/api/audio/{session_id}")
async def stream_audio(session_id: str):
    """Stream the rendered preview WAV for in-browser playback."""
    sdir = require_session(session_id)
    wav = sdir / "preview.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail="Preview not rendered yet. Call POST /preview first.")
    return FileResponse(str(wav), media_type="audio/wav", filename="mashup_preview.wav")


@app.get("/download/{session_id}")
@app.get("/api/download/{session_id}")
async def download_mp3(session_id: str):
    """Download the rendered preview as MP3."""
    sdir = require_session(session_id)
    mp3 = sdir / "preview.mp3"
    if not mp3.exists():
        # Fall back to WAV
        wav = sdir / "preview.wav"
        if not wav.exists():
            raise HTTPException(status_code=404, detail="Preview not rendered yet. Call POST /preview first.")
        return FileResponse(str(wav), media_type="audio/wav", filename="mashup_preview.wav")
    return FileResponse(str(mp3), media_type="audio/mpeg", filename="mashup_preview.mp3")


@app.delete("/session/{session_id}")
@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Remove all files associated with a session."""
    sdir = SESSIONS_DIR / session_id
    if sdir.exists():
        shutil.rmtree(sdir)
    return {"session_id": session_id, "deleted": True}


@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0"}
