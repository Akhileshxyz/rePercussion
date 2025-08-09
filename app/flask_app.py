from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth


load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"), static_folder=os.path.join(os.getcwd(), "static"))
    app.secret_key = os.getenv("APP_SECRET_KEY", os.urandom(24))
    app.config["SESSION_COOKIE_NAME"] = "repercussion_session"

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

    @app.route("/")
    def index():
        sp = get_spotify_client()
        profile = None
        if sp is not None:
            try:
                profile = sp.me()
            except Exception:
                profile = None
        return render_template("index.html", profile=profile)

    @app.route("/login")
    def login():
        oauth = get_oauth()
        auth_url = oauth.get_authorize_url()
        return redirect(auth_url)

    @app.route("/callback")
    def callback():
        error = request.args.get("error")
        if error:
            return render_template("index.html", error=error)
        code = request.args.get("code")
        if not code:
            return render_template("index.html", error="Missing authorization code")
        oauth = get_oauth()
        try:
            token_info = oauth.get_access_token(code)
        except Exception as exc:
            return render_template("index.html", error=f"Token exchange failed: {exc}")
        if not token_info or not token_info.get("access_token"):
            return render_template("index.html", error="No access token returned from Spotify")
        session["token_info"] = token_info
        frontend_url = os.getenv("FRONTEND_URL")
        if frontend_url:
            return redirect(frontend_url)
        return redirect(url_for("index"))

    @app.route("/logout")
    def logout():
        session.pop("token_info", None)
        return redirect(url_for("index"))

    @app.route("/api/me")
    def api_me():
        sp = get_spotify_client()
        if sp is None:
            return jsonify({"error": "unauthorized"}), 401
        try:
            return jsonify(sp.me())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


app = create_app()


