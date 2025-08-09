from __future__ import annotations

import spotipy


def create_spotify_client(access_token: str) -> spotipy.Spotify:
    """Create a Spotipy client using a raw access token."""
    return spotipy.Spotify(auth=access_token)


def get_current_user_profile(access_token: str) -> dict:
    """Fetch the current user's profile using the given token."""
    client = create_spotify_client(access_token)
    return client.me()


