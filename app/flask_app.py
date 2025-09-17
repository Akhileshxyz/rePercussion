from __future__ import annotations

import os
import time
from typing import Any, Dict

import json
from dotenv import find_dotenv, load_dotenv
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   session, url_for)

from app.services.analysis_service import (RateLimitError, analyze_playlist,
                                           analyze_user_music,
                                           fallback_analysis,
                                           generate_llm_analysis,
                                           generate_llm_summary,
                                           get_personalized_analysis)
from app.services.spotify_service import (get_oauth, get_spotify_client,
                                          require_spotify)

# Ensure the .env at project root is loaded and overrides any pre-set environment values
load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=True)

# Simple in-memory cache for LLM analyses
ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
ANALYSIS_CACHE_TTL_SECONDS = 3600


def create_app() -> Flask:
    app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"), static_folder=os.path.join(os.getcwd(), "static"))
    app.secret_key = os.getenv("APP_SECRET_KEY", os.urandom(24))
    app.config["SESSION_COOKIE_NAME"] = "repercussion_session"

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

    # ---------- App feature routes ----------

    @app.route("/liked-songs")
    def liked_songs():
        sp = require_spotify()
        if sp is None:
            return redirect(url_for("index"))
        try:
            page = int(request.args.get("page", "0"))
        except ValueError:
            page = 0
        limit = 20
        offset = page * limit
        results = sp.current_user_saved_tracks(limit=limit, offset=offset)
        tracks = []
        for item in results.get("items", []):
            track = item.get("track", {})
            artists = ", ".join(a.get("name", "") for a in track.get("artists", []))
            images = (track.get("album", {}) or {}).get("images", [])
            cover = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
            tracks.append(
                {
                    "name": track.get("name"),
                    "artists": artists,
                    "album": (track.get("album", {}) or {}).get("name"),
                    "cover": cover,
                    "id": track.get("id"),
                    "uri": track.get("uri"),
                }
            )
        total = results.get("total", 0)
        has_next = offset + limit < total
        has_prev = page > 0
        return render_template("liked_songs.html", tracks=tracks, page=page, has_next=has_next, has_prev=has_prev)

    @app.route("/playlist", methods=["GET", "POST"])
    def playlist():
        sp = require_spotify()
        if sp is None:
            return redirect(url_for("index"))
        playlist_data = None
        error = None
        analysis: Dict[str, Any] | None = None
        if request.method == "POST":
            user_input = request.form.get("playlist", "").strip()
            playlist_id = None
            if "open.spotify.com/playlist/" in user_input:
                try:
                    playlist_id = user_input.split("playlist/")[1].split("?")[0]
                except Exception:
                    playlist_id = None
            else:
                playlist_id = user_input or None
            if not playlist_id:
                error = "Please provide a valid Spotify playlist URL or ID."
            else:
                try:
                    pl = sp.playlist(playlist_id)
                    playlist_data = {
                        "name": pl.get("name"),
                        "owner": (pl.get("owner") or {}).get("display_name"),
                        "tracks_total": (pl.get("tracks") or {}).get("total"),
                        "images": pl.get("images", []),
                        "external_url": (pl.get("external_urls") or {}).get("spotify"),
                    }
                    # Analyze this playlist for audio features and sense
                    analysis = analyze_playlist(sp, playlist_id)
                except Exception as exc:
                    error = f"Could not fetch playlist: {exc}"
        return render_template(
            "playlist.html",
            playlist=playlist_data,
            error=error,
            summary=(analysis or {}).get("audio_summary", {}),
            ratings=(analysis or {}).get("ratings", {}),
            profile=analysis,
        )

    @app.route("/sense")
    def musical_sense():
        sp = require_spotify()
        if sp is None:
            return redirect(url_for("index"))
        analysis = analyze_user_music(sp)
        personalized_analysis = get_personalized_analysis(analysis)
        return render_template("sense.html", summary=analysis.get("audio_summary", {}), profile=analysis, personalized_analysis=personalized_analysis)

    @app.route("/recommendations")
    def recommendations():
        sp = require_spotify()
        if sp is None:
            return redirect(url_for("index"))

        # 1) Gather a representative sample of liked songs (up to 200)
        liked_ids: list[str] = []
        liked_tracks_meta: list[dict] = []
        try:
            limit = 50
            offset = 0
            while offset < 200:
                page = sp.current_user_saved_tracks(limit=limit, offset=offset) or {}
                items = page.get("items", []) or []
                if not items:
                    break
                for it in items:
                    tr = (it.get("track") or {})
                    if tr and tr.get("id"):
                        liked_ids.append(tr["id"])
                        liked_tracks_meta.append(tr)
                offset += limit
                if offset >= (page.get("total") or offset):
                    break
        except Exception:
            pass

        # Fallback: if no likes retrieved, use top tracks/artists
        if not liked_ids:
            try:
                top_tracks = sp.current_user_top_tracks(limit=20).get("items", [])
                liked_ids = [t.get("id") for t in top_tracks if t.get("id")]
                liked_tracks_meta = top_tracks
            except Exception:
                liked_ids, liked_tracks_meta = [], []

        # 2) Compute audio feature averages from likes for targeted recs
        features: list[dict] = []
        for start in range(0, len(liked_ids), 100):
            chunk = liked_ids[start : start + 100]
            try:
                features.extend(sp.audio_features(chunk) or [])
            except Exception:
                pass

        def _avg(key: str) -> float | None:
            vals = [f.get(key) for f in features if f and f.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        targets = {
            "target_danceability": _avg("danceability"),
            "target_energy": _avg("energy"),
            "target_valence": _avg("valence"),
            "target_tempo": (_avg("tempo") or 115),
            "target_acousticness": _avg("acousticness"),
            "target_instrumentalness": _avg("instrumentalness"),
            "target_speechiness": _avg("speechiness"),
            "target_liveness": _avg("liveness"),
        }
        # Drop None values as Spotify rejects them
        targets = {k: v for k, v in targets.items() if v is not None}

        # 3) Derive seeds from liked songs: top artists and tracks by popularity
        from collections import Counter
        artist_counter: Counter[str] = Counter()
        seed_tracks: list[str] = []
        for tr in liked_tracks_meta:
            for a in tr.get("artists", []) or []:
                if a.get("id"):
                    artist_counter[a["id"]] += 1
        seed_artists = [aid for aid, _ in artist_counter.most_common(3)][:3]
        # Pick up to 2 highly popular liked tracks as seeds
        liked_sorted = sorted([t for t in liked_tracks_meta if t.get("id")], key=lambda t: (t.get("popularity") or 0), reverse=True)
        seed_tracks = [t.get("id") for t in liked_sorted[:2]]

        # Ensure at least one seed exists
        if not seed_tracks and liked_ids:
            seed_tracks = liked_ids[:1]

        # 4) Call recommendations, filter out already-liked and dedupe, cap 10
        try:
            recs = sp.recommendations(
                seed_tracks=seed_tracks[:2],
                seed_artists=seed_artists[:3],
                limit=50,
                **targets,
            )
        except Exception:
            recs = {"tracks": []}

        seen_ids = set()
        liked_set = set(liked_ids)
        unique: list[dict] = []
        for tr in recs.get("tracks", []):
            tid = tr.get("id")
            if not tid or tid in liked_set or tid in seen_ids:
                continue
            seen_ids.add(tid)
            artists = ", ".join(a.get("name", "") for a in tr.get("artists", []))
            images = (tr.get("album", {}) or {}).get("images", [])
            cover = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
            unique.append(
                {
                    "name": tr.get("name"),
                    "artists": artists,
                    "album": (tr.get("album", {}) or {}).get("name"),
                    "cover": cover,
                    "preview_url": tr.get("preview_url"),
                    "external_url": (tr.get("external_urls") or {}).get("spotify"),
                }
            )
            if len(unique) >= 10:
                break

        # If underfilled, try a second pass with only artist seeds
        if len(unique) < 5 and seed_artists:
            try:
                recs2 = sp.recommendations(seed_artists=seed_artists[:5], limit=50, **targets)
            except Exception:
                recs2 = {"tracks": []}
            for tr in recs2.get("tracks", []):
                tid = tr.get("id")
                if not tid or tid in liked_set or tid in seen_ids:
                    continue
                seen_ids.add(tid)
                artists = ", ".join(a.get("name", "") for a in tr.get("artists", []))
                images = (tr.get("album", {}) or {}).get("images", [])
                cover = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
                unique.append(
                    {
                        "name": tr.get("name"),
                        "artists": artists,
                        "album": (tr.get("album", {}) or {}).get("name"),
                        "cover": cover,
                        "preview_url": tr.get("preview_url"),
                        "external_url": (tr.get("external_urls") or {}).get("spotify"),
                    }
                )
                if len(unique) >= 10:
                    break

        return render_template("recommendations.html", tracks=unique)

    @app.route("/api/llm-analyze", methods=["POST"])
    def llm_analyze():
        """LLM-driven analysis with structured JSON output and caching.

        Body JSON: {
          "tracks": [{"name": str, "artists": [str], "id": str|null}],
          "audio_summary": { ... Spotify/librosa averages ... },
          "genres": [str],
          "artists": [str]
        }
        """
        try:
            payload = request.get_json(force=True) or {}
        except Exception:
            return jsonify({"error": "invalid JSON"}), 400

        # Build a stable cache key
        try:
            cache_key = json.dumps(payload, sort_keys=True)[:4096]
        except Exception:
            cache_key = str(hash(str(payload)))

        # Serve from cache if present and fresh
        cached = ANALYSIS_CACHE.get(cache_key)
        if cached and (time.time() - cached.get("ts", 0) < ANALYSIS_CACHE_TTL_SECONDS):
            return jsonify(cached["data"])  # already structured JSON

        try:
            data = generate_llm_analysis(payload)
            ANALYSIS_CACHE[cache_key] = {"ts": time.time(), "data": data}
            return jsonify(data)
        except RateLimitError as rl:
            # soft fallback with cache miss
            fallback = fallback_analysis(payload)
            return jsonify({"_fallback": True, **fallback}), 429
        except Exception:
            fallback = fallback_analysis(payload)
            return jsonify({"_fallback": True, **fallback}), 200

    @app.route("/api/llm-summary", methods=["POST"])
    def llm_summary():
        """Generate an LLM-driven personalized summary from feature ratings and librosa stats.

        Expects JSON body with keys: ratings (dict), audio_summary (dict), favorite_genre (str|None).
        """
        try:
            payload = request.get_json(force=True) or {}
            ratings = payload.get("ratings") or {}
            audio_summary = payload.get("audio_summary") or {}
            favorite_genre = payload.get("favorite_genre")
        except Exception:
            return jsonify({"error": "invalid JSON"}), 400

        try:
            summary = generate_llm_summary(ratings, audio_summary, favorite_genre)
            return jsonify({"summary": summary})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
