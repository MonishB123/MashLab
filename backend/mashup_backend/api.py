"""
api.py

FastAPI backend for the strict mashup pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from analyze_track import analyze_mp3, as_json_dict, from_json_dict
from compatibility import compare_tracks
from mashup_engine import MashupConfig, MashupEngine
from user_model import (
    extract_pairwise_features,
    get_personalized_score,
    load_user_model,
    predict,
    save_user_model,
    update_weights,
    blend_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mashup API", version="3.0")
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
    logger.exception("Unhandled API exception:")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


class PreviewRequest(BaseModel):
    clip_duration: float = 45.0
    start_a: Optional[float] = None
    start_b: Optional[float] = None
    mashup_mode: str = "auto"  # auto | inst_a_vocals_b | vocals_a_inst_b
    use_stem_separation: bool = True
    gain_db_a: float = 0.0
    gain_db_b: Optional[float] = -3.0


class FeedbackRequest(BaseModel):
    rating: str  # "up" or "down"
    user_id: str



def session_dir(sid: str) -> Path:
    d = SESSIONS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    return d



def require_session(sid: str) -> Path:
    d = SESSIONS_DIR / sid
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found.")
    return d



def save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


@app.post("/upload")
@app.post("/api/upload")
async def upload_tracks(
    track_a: UploadFile = File(..., description="First track (MP3/WAV)"),
    track_b: UploadFile = File(..., description="Second track (MP3/WAV)"),
) -> Dict[str, Any]:
    sid = str(uuid.uuid4())
    sdir = session_dir(sid)

    ext_a = Path(track_a.filename or "track_a.mp3").suffix or ".mp3"
    ext_b = Path(track_b.filename or "track_b.mp3").suffix or ".mp3"

    path_a = sdir / f"track_a{ext_a}"
    path_b = sdir / f"track_b{ext_b}"
    save_upload(track_a, path_a)
    save_upload(track_b, path_b)

    meta = {
        "session_id": sid,
        "track_a": str(path_a),
        "track_b": str(path_b),
        "track_a_name": track_a.filename,
        "track_b_name": track_b.filename,
    }
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "session_id": sid,
        "track_a": track_a.filename,
        "track_b": track_b.filename,
        "message": "Tracks uploaded. Call POST /analyze/{session_id} next.",
    }


@app.post("/analyze/{session_id}")
@app.post("/api/analyze/{session_id}")
async def analyze_session(session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    sdir = require_session(session_id)
    meta = json.loads((sdir / "meta.json").read_text())
    path_a = meta["track_a"]
    path_b = meta["track_b"]

    try:
        feats_a = analyze_mp3(
            path_a,
            target_sr=16000,
            analysis_window_s=90.0,
            use_middle_window=True,
            fast_mode=True,
        )
        feats_b = analyze_mp3(
            path_b,
            target_sr=16000,
            analysis_window_s=90.0,
            use_middle_window=True,
            fast_mode=True,
        )
        comp_ab = compare_tracks(feats_a, feats_b)
        comp_ba = compare_tracks(feats_b, feats_a)
        cands_a = [(0.0, 1.0)]
        cands_b = [(0.0, 1.0)]
    except Exception as exc:
        logger.exception("Error during analysis for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    def pack_comp(comp):
        return {
            "score": round(comp.compatibility_score, 1),
            "grade": comp.grade,
            "layerable": comp.layerable,
            "summary": comp.summary,
            "mashup_type": comp.mashup_type,
            "reject_reasons": comp.reject_reasons,
            "details": {
                "tempo_score": round(comp.tempo_score, 3),
                "key_score": round(comp.key_score, 3),
                "energy_score": round(comp.energy_score, 3),
                "loudness_score": round(comp.loudness_score, 3),
                "timbre_score": round(comp.timbre_score, 3),
                "spectral_contrast_score": round(comp.spectral_contrast_score, 3),
                "tonnetz_score": round(comp.tonnetz_score, 3),
                "danceability_match_score": round(comp.danceability_match_score, 3),
            },
            "adjustments": {
                "stretch_factor_b": round(comp.stretch_factor_b, 4),
                "stretch_pct_b": round(comp.stretch_pct_b, 2),
                "pitch_shift_b_semitones": comp.pitch_shift_b,
                "gain_db_b": round(comp.gain_db_b, 2),
            },
        }

    result = {
        "session_id": session_id,
        "compatibility": {
            "inst_a_vocals_b": pack_comp(comp_ab),
            "vocals_a_inst_b": pack_comp(comp_ba),
        },
        "track_a": {
            "name": meta.get("track_a_name", "track_a"),
            "features": as_json_dict(feats_a),
            "candidate_segments": [
                {"start_s": round(t, 2), "quality_score": round(s, 3)} for t, s in cands_a
            ],
        },
        "track_b": {
            "name": meta.get("track_b_name", "track_b"),
            "features": as_json_dict(feats_b),
            "candidate_segments": [
                {"start_s": round(t, 2), "quality_score": round(s, 3)} for t, s in cands_b
            ],
        },
    }

    # Add personalized score if user model exists.
    if user_id:
        best_base = max(comp_ab.compatibility_score, comp_ba.compatibility_score)
        p_score = get_personalized_score(user_id, feats_a, feats_b, best_base)
        if p_score is not None:
            result["personalized_score"] = round(p_score, 1)

    # Persist per-track fast artifacts so preview can reuse them.
    (sdir / "track_a_analysis.json").write_text(json.dumps(as_json_dict(feats_a), indent=2))
    (sdir / "track_b_analysis.json").write_text(json.dumps(as_json_dict(feats_b), indent=2))
    (sdir / "track_a_candidates.json").write_text(json.dumps(cands_a, indent=2))
    (sdir / "track_b_candidates.json").write_text(json.dumps(cands_b, indent=2))
    (sdir / "analysis.json").write_text(json.dumps(result, indent=2))
    return result


@app.post("/preview/{session_id}")
@app.post("/api/preview/{session_id}")
async def render_preview(session_id: str, req: PreviewRequest = PreviewRequest()) -> Dict[str, Any]:
    sdir = require_session(session_id)
    meta = json.loads((sdir / "meta.json").read_text())

    wav_out = str(sdir / "preview.wav")
    mp3_out = str(sdir / "preview.mp3") if shutil.which("ffmpeg") else None

    forced_mode = req.mashup_mode if req.mashup_mode in {"auto", "inst_a_vocals_b", "vocals_a_inst_b"} else "auto"
    clip_duration = float(max(15.0, min(60.0, req.clip_duration)))

    # Reuse cached analysis/candidate artifacts if present.
    feats_a = None
    feats_b = None
    seg_score_a = None
    seg_score_b = None
    start_a = req.start_a
    start_b = req.start_b
    try:
        a_analysis = sdir / "track_a_analysis.json"
        b_analysis = sdir / "track_b_analysis.json"
        if a_analysis.exists() and b_analysis.exists():
            feats_a = from_json_dict(json.loads(a_analysis.read_text()))
            feats_b = from_json_dict(json.loads(b_analysis.read_text()))
    except Exception:
        logger.warning("Could not load cached analysis artifacts for session %s; falling back to on-demand.", session_id)

    if seg_score_a is None:
        seg_score_a = 1.0
    if seg_score_b is None:
        seg_score_b = 1.0

    config = MashupConfig(
        track_a=meta["track_a"],
        track_b=meta["track_b"],
        clip_duration=clip_duration,
        start_a=start_a,
        start_b=start_b,
        mashup_mode=forced_mode,
        use_stem_separation=True,
        gain_db_a=req.gain_db_a,
        gain_db_b=req.gain_db_b,
        wav_out=wav_out,
        mp3_out=mp3_out,
        stems_dir=str(sdir / "stems"),
        track_a_features=feats_a,
        track_b_features=feats_b,
        segment_score_a=seg_score_a,
        segment_score_b=seg_score_b,
    )

    result = MashupEngine().run(config)
    if not result.success:
        detail = result.error or result.summary or "Rendering failed"
        status = 422 if "Mashup rejected:" in detail else 500
        raise HTTPException(status_code=status, detail=detail)

    return {
        "session_id": session_id,
        "success": True,
        "summary": result.summary,
        "mashup_mode_used": result.mashup_mode_used,
        "clip": {
            "start_a_s": round(result.start_a, 2),
            "start_b_s": round(result.start_b, 2),
            "duration_s": clip_duration,
            "segment_quality_a": round(result.segment_score_a, 3),
            "segment_quality_b": round(result.segment_score_b, 3),
        },
        "compatibility": {
            "score": round(result.compatibility.compatibility_score, 1),
            "grade": result.compatibility.grade,
            "layerable": result.compatibility.layerable,
            "reject_reasons": result.compatibility.reject_reasons,
        },
        "audio_url": f"/api/audio/{session_id}",
        "download_url": f"/api/download/{session_id}",
    }


@app.get("/audio/{session_id}")
@app.get("/api/audio/{session_id}")
async def stream_audio(session_id: str):
    sdir = require_session(session_id)
    wav = sdir / "preview.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail="Preview audio not found.")
    return FileResponse(str(wav), media_type="audio/wav", filename="mashup_preview.wav")


@app.get("/download/{session_id}")
@app.get("/api/download/{session_id}")
async def download_audio(session_id: str):
    sdir = require_session(session_id)
    mp3 = sdir / "preview.mp3"
    wav = sdir / "preview.wav"
    if mp3.exists():
        return FileResponse(str(mp3), media_type="audio/mpeg", filename="mashup_preview.mp3")
    if wav.exists():
        return FileResponse(str(wav), media_type="audio/wav", filename="mashup_preview.wav")
    raise HTTPException(status_code=404, detail="Rendered audio not found.")


@app.post("/feedback/{session_id}")
@app.post("/api/feedback/{session_id}")
async def submit_feedback(session_id: str, req: FeedbackRequest) -> Dict[str, Any]:
    sdir = require_session(session_id)

    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    # Load track features from saved analysis artifacts.
    try:
        feats_a = from_json_dict(json.loads((sdir / "track_a_analysis.json").read_text()))
        feats_b = from_json_dict(json.loads((sdir / "track_b_analysis.json").read_text()))
    except Exception:
        raise HTTPException(status_code=400, detail="Analysis artifacts not found for this session. Run analyze first.")

    features = extract_pairwise_features(feats_a, feats_b)
    label = 1 if req.rating == "up" else 0

    model = load_user_model(req.user_id)
    model["weights"] = update_weights(model["weights"], features, label)
    model["n_votes"] += 1
    save_user_model(req.user_id, model)

    # Compute updated blended score.
    best_base = 0.0
    try:
        analysis = json.loads((sdir / "analysis.json").read_text())
        best_base = max(
            analysis.get("compatibility", {}).get("inst_a_vocals_b", {}).get("score", 0),
            analysis.get("compatibility", {}).get("vocals_a_inst_b", {}).get("score", 0),
        )
    except Exception:
        pass

    model_score = predict(model["weights"], features)
    updated_score = blend_score(best_base, model_score, model["n_votes"])

    return {
        "success": True,
        "updated_score": round(updated_score, 1),
        "n_votes": model["n_votes"],
    }


@app.delete("/session/{session_id}")
@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> Dict[str, Any]:
    sdir = require_session(session_id)
    shutil.rmtree(sdir, ignore_errors=True)
    return {"session_id": session_id, "deleted": True}
