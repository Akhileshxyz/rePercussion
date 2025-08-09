import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spotipy

from app.auth import router as auth_router


load_dotenv()

app = FastAPI(title="rePercussion API", version="0.1.0")


# Allow a configured frontend origin (e.g., Vercel) to call the API
allowed_origins = set()
frontend_url_env = os.getenv("FRONTEND_URL")
if frontend_url_env:
    allowed_origins.add(frontend_url_env)
else:
    # Fallback: allow any origin for development on Vercel preview
    allowed_origins = {"*"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)


def get_spotify_client_from_bearer(authorization: str | None = Header(default=None)) -> spotipy.Spotify:
    """Create a Spotipy client from a Bearer Authorization header."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")
    return spotipy.Spotify(auth=token)


@app.get("/api/me")
def read_me(sp_client: spotipy.Spotify = Depends(get_spotify_client_from_bearer)):
    """Return the authenticated user's Spotify profile as a smoke test for auth."""
    try:
        return sp_client.me()
    except spotipy.exceptions.SpotifyException as exc:
        status = getattr(exc, "http_status", 400) or 400
        raise HTTPException(status_code=status, detail=str(exc))


