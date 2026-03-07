# MashLab

Repo layout:

- `frontend/`: React + Vite UI
- `backend/`: Python FastAPI mashup backend (from `mashup_backend_v3`)

## Prerequisites

- Node.js (for frontend)
- Python 3.10+ (backend tested with Python 3.14 without Demucs)
- FFmpeg on PATH (recommended for MP3 export)

## Setup

From repo root:

```powershell
npm.cmd run setup:frontend
npm.cmd run setup:backend
```

Create `frontend/.env.local` (copy from `frontend/.env.example`):

```env
VITE_API_BASE_URL=http://localhost:8787
```

## Run

Terminal 1:

```powershell
npm.cmd run dev:backend
```

Terminal 2:

```powershell
npm.cmd run dev:frontend
```

## API health

```powershell
Invoke-RestMethod http://localhost:8787/api/health
```

## Notes

- No external API keys are required for this backend.
- Stem separation via Demucs is optional and currently excluded from default install because dependency support is limited on Python 3.14.
