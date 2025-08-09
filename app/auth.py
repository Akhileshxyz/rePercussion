import os
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from spotipy.oauth2 import SpotifyOAuth


load_dotenv()


def get_spotify_oauth() -> SpotifyOAuth:
    """Create and return a configured SpotifyOAuth instance.

    Uses environment variables:
    - SPOTIPY_CLIENT_ID
    - SPOTIPY_CLIENT_SECRET
    - SPOTIPY_REDIRECT_URI (defaults to http://localhost:8000/callback)
    """
    scope = (
        "user-read-email user-read-private playlist-read-private user-library-read"
    )
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        # Set SPOTIPY_REDIRECT_URI to your deployed URL, e.g. on Heroku:
        # https://<your-heroku-app>.herokuapp.com/callback
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=scope,
        cache_handler=None,
        show_dialog=True,
    )


router = APIRouter()


@router.get("/login")
async def login() -> RedirectResponse:
    """Redirect the user to Spotify's authorization URL."""
    oauth = get_spotify_oauth()
    auth_url = oauth.get_authorize_url()
    return RedirectResponse(auth_url)


@router.get("/callback")
async def callback(request: Request) -> RedirectResponse:
    """Handle Spotify redirect, exchange code for token, and forward token to frontend.

    For this prototype, we redirect the user back to the Streamlit app with the
    access token embedded in the URL query string. A production app should use
    secure server-side sessions or JWTs instead.
    """
    params = dict(request.query_params)
    error = params.get("error")
    if error:
        raise HTTPException(status_code=400, detail=error)

    code = params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    oauth = get_spotify_oauth()
    try:
        token_info = oauth.get_access_token(code)
    except Exception as exc:  # spotipy may raise generic Exception here
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {exc}")

    access_token = token_info.get("access_token")
    refresh_token = token_info.get("refresh_token")
    expires_at = token_info.get("expires_at")

    if not access_token:
        raise HTTPException(status_code=400, detail="No access token returned from Spotify")

    frontend_url = os.getenv("FRONTEND_URL")
    if frontend_url:
        query = urlencode(
            {
                "access_token": access_token or "",
                "refresh_token": refresh_token or "",
                "expires_at": str(expires_at or ""),
            }
        )
        return RedirectResponse(f"{frontend_url}?{query}")

    # If no FRONTEND_URL is configured, return the token info as JSON
    return JSONResponse(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
        }
    )


