# MashLab Project Layout

This repo is now split into:

- `frontend/`: Vite + React UI
- `backend/`: Local API server for song analysis

## Install

From the repo root:

```powershell
npm install --prefix frontend
npm install --prefix backend
```

## Environment

Copy `frontend/.env.example` to `frontend/.env.local` and set values:

```env
VITE_API_BASE_URL=http://localhost:8787
VITE_BASE44_APP_ID=your_app_id
VITE_BASE44_APP_BASE_URL=your_backend_url
```

`VITE_BASE44_*` values are optional for the current mocked base44 client, but safe to keep.

## Run Locally

Terminal 1:

```powershell
npm run dev:backend
```

Terminal 2:

```powershell
npm run dev:frontend
```

Frontend defaults to `http://localhost:5173` and backend defaults to `http://localhost:8787`.

## Health Check

```powershell
Invoke-RestMethod http://localhost:8787/api/health
```
