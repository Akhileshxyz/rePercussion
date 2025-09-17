from __future__ import annotations

import os
import time
from typing import Optional

from flask import session, flash
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth


def get_oauth() -> SpotifyOAuth:
    scope = "user-read-email user-read-private playlist-read-private user-library-read"
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=scope,
        cache_handler=None,
        show_dialog=True,
    )


def _ensure_token_valid(token_info: dict) -> dict:
    # Refresh if expired
    expires_at = token_info.get("expires_at")
    if expires_at and int(expires_at) - int(time.time()) < 60:
        oauth = get_oauth()
        refresh_token = token_info.get("refresh_token")
        if refresh_token:
            refreshed = oauth.refresh_access_token(refresh_token)
            token_info.update(refreshed)
            session["token_info"] = token_info
    return token_info


def get_spotify_client() -> Optional[Spotify]:
    token_info = session.get("token_info")
    if not token_info:
        return None
    token_info = _ensure_token_valid(token_info)
    access_token = token_info.get("access_token")
    if not access_token:
        return None
    return Spotify(auth=access_token)


def require_spotify() -> Spotify | None:
    sp = get_spotify_client()
    if sp is None:
        flash("Please log in with Spotify first.")
        return None
    return sp
