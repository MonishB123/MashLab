# Mashup Backend v2

A Python backend that turns two songs into a TikTok-ready mashup clip — automatically finding the most energetic moments (chorus / beat drop) in each track and blending them with beat-matching and key correction.

## What it does

1. **Analyzes** both tracks: BPM, musical key, energy, timbre
2. **Scores compatibility** (0–100) across tempo, harmony, energy, loudness, and timbre
3. **Finds the best segment** in each song (beat drops / chorus sections, not intros/outros)
4. **Renders a 30–60 second clip** with:
   - Automatic tempo alignment (time-stretching)
   - Key correction (pitch shifting)
   - Gain leveling
   - Beat-grid alignment
   - Fade-in/out
5. **Optional stem separation** via Demucs — puts Song A's vocals over Song B's instrumental (or vice versa)
6. Exports WAV + MP3

## Files

| File | Purpose |
|------|---------|
| `analyze_track.py` | MIR feature extraction (BPM, key, MFCC, energy) |
| `compatibility.py` | Compatibility scoring with Camelot-wheel harmonic logic |
| `segment_finder.py` | Finds high-energy chorus/drop segments using energy + onset + spectral flux |
| `source_separation.py` | Demucs stem separation (vocals / drums / bass / other) |
| `audio_render.py` | Low-level audio ops: trim, stretch, pitch-shift, mix, export |
| `mashup_engine.py` | Full pipeline orchestrator |
| `api.py` | FastAPI REST API for the frontend |
| `render_mashup.py` | Original CLI tool (still works) |

## Install

```bash
pip install -r requirements.txt

# Also install ffmpeg for MP3 export:
# macOS:  brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

## Quick CLI

```bash
python render_mashup.py song_a.mp3 song_b.mp3 \
  --duration 45 \
  --wav-out mashup.wav \
  --mp3-out mashup.mp3
```

## API Server

```bash
uvicorn api:app --reload --port 8000
```

### Frontend flow

```
1. POST /upload
   Form-data: track_a=<file>, track_b=<file>
   → { session_id: "abc-123" }

2. POST /analyze/abc-123
   → { compatibility: { score, grade, summary, ... }, candidate_segments, ... }

3. POST /preview/abc-123
   Body (optional): { clip_duration: 45, mashup_mode: "auto" }
   → { audio_url: "/audio/abc-123", download_url: "/download/abc-123", ... }

4. GET /audio/abc-123      → streams WAV for browser playback
   GET /download/abc-123   → downloads MP3
```

## Blend modes

| Mode | Description |
|------|-------------|
| `auto` | Chosen based on compatibility scores |
| `full_blend` | Both full tracks mixed |
| `vocals_a_inst_b` | Vocals from A + drums/bass/other from B *(needs Demucs)* |
| `inst_a_vocals_b` | Instrumental from A + vocals from B *(needs Demucs)* |
| `acapella_over_beat` | Acapella A over full B *(needs Demucs)* |

## Compatibility score guide

| Score | Grade | Meaning |
|-------|-------|---------|
| 85–100 | A | Excellent — will sound natural |
| 70–84 | B | Good — minor adjustments may help |
| 55–69 | C | Moderate — noticeable but workable |
| 40–54 | D | Difficult — only specific segments will work |
| <40 | F | Poor match |
