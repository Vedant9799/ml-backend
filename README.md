# ml-backend (FastAPI on Railway)

## Local dev
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://127.0.0.1:8000/docs
```

## Deploy (Railway)
- Create a new Railway project → Deploy from GitHub (this repo).
- Railway sets `PORT` automatically.
- (Optional) Set env var `NETLIFY_ORIGIN=https://your-site.netlify.app` to lock CORS.

The process is defined in `Procfile`:
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
