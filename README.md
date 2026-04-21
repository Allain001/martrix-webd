# MatrixVis Public Web App

`one_page_demo` is now being upgraded from a local demo shell into a public-facing MatrixVis website.
The target is a browser-openable product site, not a Streamlit-only presentation.

## Current Architecture

- `frontend/`: React + Vite, responsible for the GeoGebra-inspired workspace UI
- `backend/`: FastAPI, responsible for matrix analysis APIs and static-site hosting
- Frontend assets and `/api/*` endpoints are served from the same FastAPI site

## What Is Already Done

- The backend no longer depends on legacy matrix code outside `one_page_demo`
- The frontend has been reshaped into a three-column math workspace
- Matrix, vector, and operation state can now be restored from the URL
- Users can copy a shareable link to open the same workspace state elsewhere

## Local Run

Double-click:

```bat
start_fullstack.bat
```

The startup script now checks both backend and frontend dependencies before launching.

Or run manually:

```bash
cd one_page_demo/frontend
npm install
npm run build
```

```bash
cd one_page_demo/backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Smoke Test

Run the full local verification flow:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke_test.ps1
```

This will:

- build the frontend
- start the FastAPI site temporarily
- check `/api/health`
- check the homepage and defense-mode route
- call the main analysis API
- verify public assets such as `favicon.svg` and `site.webmanifest`

See [DEPLOY_CHECKLIST.md](./DEPLOY_CHECKLIST.md) for the publish checklist.

## Continuous Integration

If this workspace is pushed to GitHub as a repository root, the workflow at:

```text
.github/workflows/matrixvis-ci.yml
```

will automatically:

- install backend and frontend dependencies
- build the frontend
- compile the backend
- run the MatrixVis smoke test

## Production Direction

This project is now organized for same-origin deployment:

- the frontend is built into static files
- FastAPI serves both the UI and APIs
- the site can be packaged and deployed as a single container

## Docker Deployment

Build the image from `one_page_demo/`:

```bash
docker build -t matrixvis-web .
```

Run it:

```bash
docker run -p 8000:8000 matrixvis-web
```

Then open:

```text
http://127.0.0.1:8000
```

## Render Deployment

This folder now also includes `render.yaml`.
If you push this directory to GitHub, Render can create the web service from the Docker setup directly.

## Next Product Steps

- add richer case-library narratives and onboarding copy
- add a clearer public homepage / help / about flow
- deploy to a public host with a real domain
- polish mobile experience and answer-page mode for defense/demo usage
