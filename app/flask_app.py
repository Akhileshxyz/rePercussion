from __future__ import annotations

import os
import time
from typing import Optional, List, Dict, Any
import json

from dotenv import load_dotenv, find_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for, flash
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
try:
    from openai import OpenAI
    class RateLimitError(Exception):
        pass
except Exception:
    OpenAI = None  # type: ignore
    class RateLimitError(Exception):
        pass


# Ensure the .env at project root is loaded and overrides any pre-set environment values
load_dotenv(dotenv_path=find_dotenv(usecwd=True), override=True)

# Simple in-memory cache for LLM analyses
ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
ANALYSIS_CACHE_TTL_SECONDS = 3600


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

    # ---------- App feature routes ----------

    def require_spotify() -> Spotify | None:
        sp = get_spotify_client()
        if sp is None:
            flash("Please log in with Spotify first.")
            return None
        return sp

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

    def get_personalized_analysis(analysis: Dict[str, Any]) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""
        Analyze the following music taste profile and generate a fun, exciting, and personalized summary for a music lover.
        The user's favorite genre is {analysis.get('favorite_genre', 'not available')}.
        Their top artist is {analysis.get('top_artist', 'not available')}.
        They primarily listen to music in {analysis.get('primary_language', 'not available')}.
        Their music has the following characteristics:
        - Danceability: {analysis.get('audio_summary', {}).get('danceability', 'not available')}
        - Energy: {analysis.get('audio_summary', {}).get('energy', 'not available')}
        - Valence (Positivity): {analysis.get('audio_summary', {}).get('valence', 'not available')}
        - Tempo: {analysis.get('audio_summary', {}).get('tempo', 'not available')} BPM
        - Acousticness: {analysis.get('audio_summary', {}).get('acousticness', 'not available')}
        - Instrumentalness: {analysis.get('audio_summary', {}).get('instrumentalness', 'not available')}
        - Librosa Tempo: {analysis.get('audio_summary', {}).get('librosa_tempo', 'not available')}
        - Spectral Centroid Mean: {analysis.get('audio_summary', {}).get('spectral_centroid_mean', 'not available')}
        - Dominant Pitch Class: {analysis.get('audio_summary', {}).get('dominant_pitch_class', 'not available')}

        Give them a cool title for their musical taste and a paragraph or two that sounds exciting and insightful.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a witty and insightful music analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Sorry, I couldn't generate a personalized analysis at the moment. Error: {e}"

    @app.route("/recommendations")
    def recommendations():
        sp = require_spotify()
        if sp is None:
            return redirect(url_for("index"))
        # Seed with user's top tracks/artists for simple recs
        try:
            top_tracks = sp.current_user_top_tracks(limit=5).get("items", [])
            top_artists = sp.current_user_top_artists(limit=5).get("items", [])
        except Exception:
            top_tracks, top_artists = [], []
        seed_tracks = [t.get("id") for t in top_tracks[:2] if t.get("id")]
        seed_artists = [a.get("id") for a in top_artists[:2] if a.get("id")]
        try:
            recs = sp.recommendations(seed_tracks=seed_tracks, seed_artists=seed_artists, limit=20)
        except Exception:
            recs = {"tracks": []}
        tracks = []
        for tr in recs.get("tracks", []):
            artists = ", ".join(a.get("name", "") for a in tr.get("artists", []))
            images = (tr.get("album", {}) or {}).get("images", [])
            cover = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
            tracks.append(
                {
                    "name": tr.get("name"),
                    "artists": artists,
                    "album": (tr.get("album", {}) or {}).get("name"),
                    "cover": cover,
                    "preview_url": tr.get("preview_url"),
                    "external_url": (tr.get("external_urls") or {}).get("spotify"),
                }
            )
        return render_template("recommendations.html", tracks=tracks)

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


def analyze_user_music(sp: Spotify) -> Dict[str, Any]:
    """Analyze user's music taste using Spotipy data, librosa heuristics, and sentence embeddings.

    Returns a dict with:
    - favorite_genre: str | None
    - top_artist: str | None
    - primary_language: str | None
    - common_instruments: List[str]
    - musical_sense: str
    - audio_summary: Dict[str, float]
    """
    from collections import Counter
    from langdetect import detect, LangDetectException

    # 1) Gather data: top tracks and artists
    top_tracks = (sp.current_user_top_tracks(limit=50, time_range="medium_term") or {}).get("items", [])
    top_artists = (sp.current_user_top_artists(limit=50, time_range="medium_term") or {}).get("items", [])

    # Favorite Genre and Top Artist
    genre_counter: Counter[str] = Counter()
    for artist in top_artists:
        for g in artist.get("genres", []) or []:
            if g:
                genre_counter[g.lower()] += 1
    favorite_genre = (genre_counter.most_common(1)[0][0] if genre_counter else None)
    top_artist = (top_artists[0].get("name") if top_artists else None)

    # Primary language from titles and artist names
    texts: List[str] = []
    for t in top_tracks:
        name = t.get("name")
        if name:
            texts.append(name)
        for a in t.get("artists", []) or []:
            n = a.get("name")
            if n:
                texts.append(n)
    lang_counter: Counter[str] = Counter()
    for txt in texts[:200]:
        try:
            lang = detect(txt)
            if lang:
                lang_counter[lang] += 1
        except LangDetectException:
            continue
    primary_language = (lang_counter.most_common(1)[0][0] if lang_counter else None)

    # Audio features aggregates for musical sense
    track_ids = [t.get("id") for t in top_tracks if t.get("id")]
    features: List[Dict[str, Any]] = []
    if track_ids:
        for chunk_start in range(0, len(track_ids), 100):
            chunk = track_ids[chunk_start : chunk_start + 100]
            try:
                features.extend(sp.audio_features(chunk) or [])
            except Exception:
                pass

    def _avg(key: str) -> float | None:
        vals = [f.get(key) for f in features if f and f.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    audio_summary: Dict[str, float | None] = {
        "danceability": _avg("danceability"),
        "energy": _avg("energy"),
        "valence": _avg("valence"),
        "tempo": _avg("tempo"),
        "acousticness": _avg("acousticness"),
        "instrumentalness": _avg("instrumentalness"),
        "speechiness": _avg("speechiness"),
        "liveness": _avg("liveness"),
    }

    def _rate(key: str, value: float | None) -> str | None:
        if value is None:
            return None
        if key == "tempo":
            if value < 90:
                return "Slow"
            if value < 120:
                return "Moderate"
            return "Fast"
        # For 0..1 features
        if value < 0.33:
            return "Low"
        if value < 0.66:
            return "Medium"
        return "High"

    ratings: Dict[str, str | None] = {k: _rate(k, v) for k, v in audio_summary.items()}

    # Optional: deepen analysis via librosa on preview clips when available
    # This enriches tempo and timbre estimates using signal processing
    # Librosa-based analysis to enrich/fallback ratings
    try:
        import tempfile, os as _os
        import requests as _rq
        import numpy as _np
        import librosa as _librosa

        preview_urls = [t.get("preview_url") for t in top_tracks if t.get("preview_url")]
        preview_urls = preview_urls[:5]
        tempos: list[float] = []
        spectral_centroids: list[float] = []
        rms_vals: list[float] = []
        zcr_vals: list[float] = []
        flatness_vals: list[float] = []
        rolloff_vals: list[float] = []
        onset_vals: list[float] = []
        chroma_profiles: list[list[float]] = []
        for url in preview_urls:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    resp = _rq.get(url, timeout=10)
                    resp.raise_for_status()
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                y, sr = _librosa.load(tmp_path, mono=True)
                tempo, _ = _librosa.beat.beat_track(y=y, sr=sr)
                tempos.append(float(tempo))
                sc = _librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_centroids.append(float(_np.mean(sc)))
                rms = _librosa.feature.rms(y=y)
                rms_vals.append(float(_np.mean(rms)))
                zcr = _librosa.feature.zero_crossing_rate(y)
                zcr_vals.append(float(_np.mean(zcr)))
                flat = _librosa.feature.spectral_flatness(y=y)
                flatness_vals.append(float(_np.mean(flat)))
                roll = _librosa.feature.spectral_rolloff(y=y, sr=sr)
                rolloff_vals.append(float(_np.mean(roll)))
                onset_strength = _librosa.onset.onset_strength(y=y, sr=sr)
                onset_vals.append(float(_np.mean(onset_strength)))
                chroma = _librosa.feature.chroma_cqt(y=y, sr=sr)
                chroma_profiles.append(_np.mean(chroma, axis=1).tolist())
            finally:
                try:
                    _os.unlink(tmp_path)
                except Exception:
                    pass
        if tempos:
            audio_summary["librosa_tempo"] = round(sum(tempos) / len(tempos), 2)
        if spectral_centroids:
            audio_summary["spectral_centroid_mean"] = round(sum(spectral_centroids) / len(spectral_centroids), 2)
        if rms_vals:
            audio_summary["rms_mean"] = round(sum(rms_vals) / len(rms_vals), 6)
        if zcr_vals:
            audio_summary["zcr_mean"] = round(sum(zcr_vals) / len(zcr_vals), 6)
        if flatness_vals:
            audio_summary["flatness_mean"] = round(sum(flatness_vals) / len(flatness_vals), 6)
        if rolloff_vals:
            audio_summary["rolloff_mean"] = round(sum(rolloff_vals) / len(rolloff_vals), 3)
        if onset_vals:
            audio_summary["onset_mean"] = round(sum(onset_vals) / len(onset_vals), 6)
        if chroma_profiles:
            avg_chroma = _np.mean(_np.array(chroma_profiles), axis=0)
            key_index = int(_np.argmax(avg_chroma))
            pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            audio_summary["dominant_pitch_class"] = pitch_classes[key_index]

        # Fill ratings using librosa proxies where missing
        def _fill_if_missing(k: str, value: str):
            if ratings.get(k) is None:
                ratings[k] = value

        if audio_summary.get("tempo") is None and tempos:
            t = sum(tempos) / len(tempos)
            _fill_if_missing("tempo", "Slow" if t < 90 else ("Moderate" if t < 120 else "Fast"))

        # Energy via spectral centroid with robust normalization
        if ratings.get("energy") is None and spectral_centroids:
            sc = sum(spectral_centroids) / len(spectral_centroids)
            # Normalize by Nyquist; on common 22k-44k sr, centroid ranges ~1k-6k
            if sc < 1700:
                _fill_if_missing("energy", "Low")
            elif sc < 3200:
                _fill_if_missing("energy", "Medium")
            else:
                _fill_if_missing("energy", "High")

        # Danceability: tempo in 90-130 and strong onsets
        if ratings.get("danceability") is None and (tempos or onset_vals):
            t = (sum(tempos) / len(tempos)) if tempos else 110.0
            onset = (sum(onset_vals) / len(onset_vals)) if onset_vals else 0.0
            if 95 <= t <= 130 and onset > 0.1:
                _fill_if_missing("danceability", "High")
            elif 85 <= t <= 140:
                _fill_if_missing("danceability", "Medium")
            else:
                _fill_if_missing("danceability", "Low")

        # Valence proxy: combine brightness and rolloff if available
        if ratings.get("valence") is None and spectral_centroids:
            sc = sum(spectral_centroids) / len(spectral_centroids)
            # slightly higher thresholds to avoid all-low
            if sc < 1600:
                _fill_if_missing("valence", "Low")
            elif sc < 2900:
                _fill_if_missing("valence", "Medium")
            else:
                _fill_if_missing("valence", "High")

        # Acousticness proxy: low flatness + low zcr => more acoustic
        if ratings.get("acousticness") is None and (flatness_vals or zcr_vals):
            flat = (sum(flatness_vals) / len(flatness_vals)) if flatness_vals else 0.3
            zcrv = (sum(zcr_vals) / len(zcr_vals)) if zcr_vals else 0.1
            if flat < 0.2 and zcrv < 0.08:
                _fill_if_missing("acousticness", "High")
            elif flat < 0.35 and zcrv < 0.12:
                _fill_if_missing("acousticness", "Medium")
            else:
                _fill_if_missing("acousticness", "Low")

        # Speechiness proxy: high zcr + high flatness
        if ratings.get("speechiness") is None and (flatness_vals or zcr_vals):
            flat = (sum(flatness_vals) / len(flatness_vals)) if flatness_vals else 0.2
            zcrv = (sum(zcr_vals) / len(zcr_vals)) if zcr_vals else 0.08
            if zcrv > 0.18 and flat > 0.35:
                _fill_if_missing("speechiness", "High")
            elif zcrv > 0.12 or flat > 0.25:
                _fill_if_missing("speechiness", "Medium")
            else:
                _fill_if_missing("speechiness", "Low")

        # Instrumentalness proxy: low speechiness -> higher instrumentalness
        if ratings.get("instrumentalness") is None:
            spch = ratings.get("speechiness")
            if spch == "Low":
                _fill_if_missing("instrumentalness", "High")
            elif spch == "Medium":
                _fill_if_missing("instrumentalness", "Medium")
            else:
                _fill_if_missing("instrumentalness", "Low")

        # Liveness proxy: higher onset and rms variation => higher liveness
        if ratings.get("liveness") is None and (onset_vals or rms_vals):
            onset = (sum(onset_vals) / len(onset_vals)) if onset_vals else 0.0
            rmsm = (sum(rms_vals) / len(rms_vals)) if rms_vals else 0.0
            if onset > 0.2 and rmsm > 0.02:
                _fill_if_missing("liveness", "High")
            elif onset > 0.1:
                _fill_if_missing("liveness", "Medium")
            else:
                _fill_if_missing("liveness", "Low")
    except Exception:
        # Librosa is optional at runtime; if unavailable or preview fetch fails, proceed silently
        pass

    # Optionally let the LLM refine ratings from librosa/Spotify inputs
    llm_ratings = generate_llm_ratings(audio_summary)
    if llm_ratings:
        ratings.update({k: llm_ratings.get(k) or ratings.get(k) for k in ratings.keys()})

    # Ensure every metric has a rating to avoid confusing placeholders in UI
    for k, v in list(audio_summary.items()):
        if ratings.get(k) is None:
            # If we have a numeric value, rate from it; otherwise choose neutral
            ratings[k] = _rate(k, v) or ("Moderate" if k == "tempo" else "Medium")

    # Seed instruments from genre and feature hints
    common_instruments: List[str] = []
    def add(instr: str):
        if instr not in common_instruments:
            common_instruments.append(instr)
    g = favorite_genre or ""
    if any(k in g for k in ["rock", "metal", "punk"]):
        add("electric guitar"); add("drums"); add("bass guitar")
    if any(k in g for k in ["pop", "dance", "edm", "house"]):
        add("synthesizer"); add("drum machine")
    if any(k in g for k in ["jazz", "blues"]):
        add("saxophone"); add("piano"); add("double bass")
    if any(k in g for k in ["classical", "orchestral"]):
        add("violin"); add("piano"); add("cello")
    if (audio_summary.get("acousticness") or 0) > 0.5:
        add("acoustic guitar")
    if (audio_summary.get("instrumentalness") or 0) > 0.5:
        add("instrumental leads")
    if (audio_summary.get("energy") or 0) > 0.6 and (audio_summary.get("danceability") or 0) > 0.6:
        add("percussion")

    # Ensure we always return at least three instruments
    if len(common_instruments) < 3:
        # Try LLM inference first
        inferred = generate_llm_instruments([favorite_genre] if favorite_genre else [], ratings, audio_summary) or []
        for instr in inferred:
            if instr and instr not in common_instruments:
                common_instruments.append(instr)
        # Heuristic complements
        if ratings.get("energy") == "High":
            for i in ["drums", "electric guitar", "bass guitar"]:
                if i not in common_instruments:
                    common_instruments.append(i)
        if ratings.get("acousticness") == "High":
            for i in ["acoustic guitar", "piano", "strings"]:
                if i not in common_instruments:
                    common_instruments.append(i)
        if ratings.get("speechiness") == "High":
            for i in ["vocals", "spoken word", "rap vocals"]:
                if i not in common_instruments:
                    common_instruments.append(i)
        if ratings.get("instrumentalness") == "High":
            for i in ["synthesizer", "strings", "pads"]:
                if i not in common_instruments:
                    common_instruments.append(i)
        common_instruments = common_instruments[:5]

    # Build a dynamic, personalized summary instead of generic archetypes
    musical_sense = _describe_personality(ratings, audio_summary, favorite_genre)

    # Optionally enhance with LLM-crafted summary
    llm_summary = generate_llm_summary(ratings, audio_summary, favorite_genre)

    return {
        "favorite_genre": favorite_genre,
        "top_artist": top_artist,
        "primary_language": primary_language,
        "common_instruments": common_instruments,
        "musical_sense": musical_sense,
        "audio_summary": audio_summary,
        "ratings": ratings,
        "llm_summary": llm_summary,
    }


def analyze_playlist(sp: Spotify, playlist_id: str) -> Dict[str, Any]:
    """Analyze a specific playlist similarly to analyze_user_music.

    Returns the same structure as analyze_user_music including ratings.
    """
    from collections import Counter
    from langdetect import detect, LangDetectException

    # Fetch all tracks in the playlist
    items: list[dict] = []
    next_url: str | None = None
    limit = 100
    offset = 0
    while True:
        try:
            page = sp.playlist_items(playlist_id, limit=limit, offset=offset, additional_types=["track"]) or {}
        except Exception:
            break
        items.extend(page.get("items", []) or [])
        total = (page.get("total") or 0)
        offset += limit
        if offset >= total:
            break

    tracks_meta: list[dict] = []
    track_ids: list[str] = []
    artist_ids: list[str] = []
    texts: list[str] = []
    preview_urls: list[str] = []
    for it in items:
        tr = (it.get("track") or {})
        if not tr:
            continue
        tracks_meta.append(tr)
        if tr.get("id"):
            track_ids.append(tr["id"])
        if tr.get("name"):
            texts.append(tr["name"]) 
        for a in (tr.get("artists") or []):
            aid = a.get("id")
            if aid:
                artist_ids.append(aid)
            n = a.get("name")
            if n:
                texts.append(n)
        if tr.get("preview_url"):
            preview_urls.append(tr["preview_url"])  

    # Genres via artists endpoint
    genre_counter: Counter[str] = Counter()
    for start in range(0, len(artist_ids), 50):
        chunk = artist_ids[start : start + 50]
        try:
            arts = sp.artists(chunk).get("artists", [])
        except Exception:
            arts = []
        for a in arts:
            for g in a.get("genres", []) or []:
                if g:
                    genre_counter[g.lower()] += 1
    favorite_genre = (genre_counter.most_common(1)[0][0] if genre_counter else None)

    # Primary language
    lang_counter: Counter[str] = Counter()
    for txt in texts[:200]:
        try:
            lang = detect(txt)
            if lang:
                lang_counter[lang] += 1
        except LangDetectException:
            continue
    primary_language = (lang_counter.most_common(1)[0][0] if lang_counter else None)

    # Audio features aggregates
    features: list[Dict[str, Any]] = []
    for start in range(0, len(track_ids), 100):
        chunk = track_ids[start : start + 100]
        try:
            features.extend(sp.audio_features(chunk) or [])
        except Exception:
            pass

    def _avg(key: str) -> float | None:
        vals = [f.get(key) for f in features if f and f.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    audio_summary: Dict[str, float | None] = {
        "danceability": _avg("danceability"),
        "energy": _avg("energy"),
        "valence": _avg("valence"),
        "tempo": _avg("tempo"),
        "acousticness": _avg("acousticness"),
        "instrumentalness": _avg("instrumentalness"),
        "speechiness": _avg("speechiness"),
        "liveness": _avg("liveness"),
    }

    def _rate(key: str, value: float | None) -> str | None:
        if value is None:
            return None
        if key == "tempo":
            if value < 90:
                return "Slow"
            if value < 120:
                return "Moderate"
            return "Fast"
        if value < 0.33:
            return "Low"
        if value < 0.66:
            return "Medium"
        return "High"

    ratings: Dict[str, str | None] = {k: _rate(k, v) for k, v in audio_summary.items()}

    # Librosa-based enrichment/fallback ratings on playlist previews
    try:
        import tempfile, os as _os
        import requests as _rq
        import numpy as _np
        import librosa as _librosa

        tempos: list[float] = []
        spectral_centroids: list[float] = []
        rms_vals: list[float] = []
        zcr_vals: list[float] = []
        flatness_vals: list[float] = []
        onset_vals: list[float] = []

        for url in preview_urls[:5]:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    resp = _rq.get(url, timeout=10)
                    resp.raise_for_status()
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                y, sr = _librosa.load(tmp_path, mono=True)
                tempo, _ = _librosa.beat.beat_track(y=y, sr=sr)
                tempos.append(float(tempo))
                sc = _librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_centroids.append(float(_np.mean(sc)))
                rms = _librosa.feature.rms(y=y)
                rms_vals.append(float(_np.mean(rms)))
                zcr = _librosa.feature.zero_crossing_rate(y)
                zcr_vals.append(float(_np.mean(zcr)))
                flat = _librosa.feature.spectral_flatness(y=y)
                flatness_vals.append(float(_np.mean(flat)))
                onset_strength = _librosa.onset.onset_strength(y=y, sr=sr)
                onset_vals.append(float(_np.mean(onset_strength)))
            finally:
                try:
                    _os.unlink(tmp_path)
                except Exception:
                    pass

        def _fill_if_missing(k: str, value: str):
            if ratings.get(k) is None:
                ratings[k] = value

        if audio_summary.get("tempo") is None and tempos:
            t = sum(tempos) / len(tempos)
            _fill_if_missing("tempo", "Slow" if t < 90 else ("Moderate" if t < 120 else "Fast"))

        if ratings.get("energy") is None and spectral_centroids:
            sc = sum(spectral_centroids) / len(spectral_centroids)
            _fill_if_missing("energy", "Low" if sc < 2000 else ("Medium" if sc < 3500 else "High"))

        if ratings.get("danceability") is None and (tempos or onset_vals):
            t = (sum(tempos) / len(tempos)) if tempos else 110.0
            onset = (sum(onset_vals) / len(onset_vals)) if onset_vals else 0.0
            if 95 <= t <= 130 and onset > 0.1:
                _fill_if_missing("danceability", "High")
            elif 85 <= t <= 140:
                _fill_if_missing("danceability", "Medium")
            else:
                _fill_if_missing("danceability", "Low")

        if ratings.get("valence") is None and spectral_centroids:
            sc = sum(spectral_centroids) / len(spectral_centroids)
            _fill_if_missing("valence", "Low" if sc < 1800 else ("Medium" if sc < 3000 else "High"))

        if ratings.get("acousticness") is None and (flatness_vals or zcr_vals):
            flat = (sum(flatness_vals) / len(flatness_vals)) if flatness_vals else 0.3
            zcrv = (sum(zcr_vals) / len(zcr_vals)) if zcr_vals else 0.1
            if flat < 0.2 and zcrv < 0.08:
                _fill_if_missing("acousticness", "High")
            elif flat < 0.35 and zcrv < 0.12:
                _fill_if_missing("acousticness", "Medium")
            else:
                _fill_if_missing("acousticness", "Low")

        if ratings.get("speechiness") is None and (flatness_vals or zcr_vals):
            flat = (sum(flatness_vals) / len(flatness_vals)) if flatness_vals else 0.2
            zcrv = (sum(zcr_vals) / len(zcr_vals)) if zcr_vals else 0.08
            if zcrv > 0.18 and flat > 0.35:
                _fill_if_missing("speechiness", "High")
            elif zcrv > 0.12 or flat > 0.25:
                _fill_if_missing("speechiness", "Medium")
            else:
                _fill_if_missing("speechiness", "Low")

        if ratings.get("instrumentalness") is None:
            spch = ratings.get("speechiness")
            if spch == "Low":
                _fill_if_missing("instrumentalness", "High")
            elif spch == "Medium":
                _fill_if_missing("instrumentalness", "Medium")
            else:
                _fill_if_missing("instrumentalness", "Low")

        if ratings.get("liveness") is None and (onset_vals or rms_vals):
            onset = (sum(onset_vals) / len(onset_vals)) if onset_vals else 0.0
            rmsm = (sum(rms_vals) / len(rms_vals)) if rms_vals else 0.0
            if onset > 0.2 and rmsm > 0.02:
                _fill_if_missing("liveness", "High")
            elif onset > 0.1:
                _fill_if_missing("liveness", "Medium")
            else:
                _fill_if_missing("liveness", "Low")
    except Exception:
        pass

    # Optionally let the LLM refine ratings from librosa/Spotify inputs
    llm_ratings = generate_llm_ratings(audio_summary)
    if llm_ratings:
        ratings.update({k: llm_ratings.get(k) or ratings.get(k) for k in ratings.keys()})

    # Ensure complete ratings coverage
    for k, v in list(audio_summary.items()):
        if ratings.get(k) is None:
            ratings[k] = _rate(k, v) or ("Moderate" if k == "tempo" else "Medium")

    # Personality: dynamic description based on features
    musical_sense = _describe_personality(ratings, audio_summary, favorite_genre)

    common_instruments: List[str] = []
    def add(instr: str):
        if instr not in common_instruments:
            common_instruments.append(instr)
    g = favorite_genre or ""
    if any(k in g for k in ["rock", "metal", "punk"]):
        add("electric guitar"); add("drums"); add("bass guitar")
    if any(k in g for k in ["pop", "dance", "edm", "house"]):
        add("synthesizer"); add("drum machine")
    if any(k in g for k in ["jazz", "blues"]):
        add("saxophone"); add("piano"); add("double bass")
    if any(k in g for k in ["classical", "orchestral"]):
        add("violin"); add("piano"); add("cello")
    if (audio_summary.get("acousticness") or 0) > 0.5:
        add("acoustic guitar")
    if (audio_summary.get("instrumentalness") or 0) > 0.5:
        add("instrumental leads")
    if (audio_summary.get("energy") or 0) > 0.6 and (audio_summary.get("danceability") or 0) > 0.6:
        add("percussion")

    llm_summary = generate_llm_summary(ratings, audio_summary, favorite_genre)
    # Ensure instruments are populated for playlist context too
    if len(common_instruments) < 3:
        inferred = generate_llm_instruments([favorite_genre] if favorite_genre else [], ratings, audio_summary) or []
        for instr in inferred:
            if instr and instr not in common_instruments:
                common_instruments.append(instr)
        if ratings.get("energy") == "High":
            for i in ["drums", "electric guitar", "bass guitar"]:
                if i not in common_instruments:
                    common_instruments.append(i)
        if ratings.get("acousticness") == "High":
            for i in ["acoustic guitar", "piano", "strings"]:
                if i not in common_instruments:
                    common_instruments.append(i)
        if ratings.get("instrumentalness") == "High":
            for i in ["synthesizer", "strings", "pads"]:
                if i not in common_instruments:
                    common_instruments.append(i)
        common_instruments = common_instruments[:5]

    return {
        "favorite_genre": favorite_genre,
        "top_artist": None,
        "primary_language": primary_language,
        "common_instruments": common_instruments,
        "musical_sense": musical_sense,
        "audio_summary": audio_summary,
        "ratings": ratings,
        "llm_summary": llm_summary,
    }


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)


# ---------- Helpers ----------
def _describe_personality(ratings: Dict[str, str | None], summary: Dict[str, Any], favorite_genre: str | None) -> str:
    """Create a short, varied description from ratings and summary.

    This avoids generic one-liners by mixing multiple traits.
    """
    def pick(keys: list[str], target: str) -> bool:
        return any((ratings.get(k) or "").lower() == target.lower() for k in keys)

    phrases: list[str] = []

    # Core vibe
    if pick(["energy", "danceability"], "High"):
        phrases.append("high‑energy and rhythm‑forward")
    elif pick(["energy"], "Medium") and pick(["danceability"], "Medium"):
        phrases.append("balanced and groove‑friendly")
    elif pick(["acousticness"], "High"):
        phrases.append("warm, acoustic textures")
    else:
        phrases.append("mood‑driven selections")

    # Emotional tone
    if pick(["valence"], "High"):
        phrases.append("generally uplifting")
    elif pick(["valence"], "Low"):
        phrases.append("introspective and moody")
    else:
        phrases.append("tonally versatile")

    # Tempo
    tempo_label = ratings.get("tempo") or "Moderate"
    phrases.append(f"with {tempo_label.lower()} tempos")

    # Extra color
    if pick(["speechiness"], "High"):
        phrases.append("story‑forward vocals")
    if pick(["instrumentalness"], "High"):
        phrases.append("instrumental leads")
    if pick(["liveness"], "High"):
        phrases.append("live‑performance feel")

    # Genre hint
    if favorite_genre:
        phrases.append(f"rooted in {favorite_genre}")

    desc = "; ".join(phrases[:-1]) + ("; " if len(phrases) > 1 else "") + phrases[-1]

    # Add a concise closing tag of standout traits
    standout = [k for k, v in ratings.items() if v == "High" and k in ("energy", "danceability", "acousticness", "valence")]
    if standout:
        tag = ", ".join(standout)
        desc += f". Standout: {tag}."
    return desc


def generate_llm_summary(ratings: Dict[str, Any], audio_summary: Dict[str, Any], favorite_genre: str | None) -> str:
    """Call OpenAI (or compatible) LLM to craft a personalized summary using our DSP analysis.

    Requires OPENAI_API_KEY in environment. Falls back to deterministic description if unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _describe_personality(ratings, audio_summary, favorite_genre)

    try:
        client = OpenAI(api_key=api_key)

        # Build a compact prompt with ratings and key DSP stats only
        lines = [
            "You are a concise, vivid music curator.",
            "Write a 2-3 sentence personalized description of the listener's taste.",
            "Avoid generic platitudes; reference concrete traits.",
        ]
        if favorite_genre:
            lines.append(f"Favorite genre hint: {favorite_genre}.")
        lines.append(f"Ratings: {ratings}.")
        keep = {k: audio_summary.get(k) for k in ("tempo","spectral_centroid_mean","librosa_tempo","dominant_pitch_class") if audio_summary.get(k) is not None}
        lines.append(f"DSP: {keep}.")
        lines.append("Return only the description.")
        prompt = "\n".join(lines)

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You craft engaging yet precise music taste summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _describe_personality(ratings, audio_summary, favorite_genre)


def generate_llm_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return fallback_analysis(payload)
    client = OpenAI(api_key=api_key)
    tracks = payload.get("tracks") or []
    audio_summary = payload.get("audio_summary") or {}
    genres = payload.get("genres") or []
    artists = payload.get("artists") or []
    top_tracks_text = "; ".join([f"{t.get('name','')} – {', '.join(t.get('artists',[]))}" for t in tracks[:20]])
    prompt = (
        "Analyze this music data and provide insights about the user's musical preferences:\n\n"
        f"User's Top Tracks: [{top_tracks_text}]\n"
        f"Audio Features: {json.dumps({k: audio_summary.get(k) for k in ['energy','danceability','valence','acousticness','tempo']}, ensure_ascii=False)}\n"
        f"Top Genres: {genres}\n"
        f"Top Artists: {artists}\n\n"
        "Please provide analysis in this exact JSON format:\n"
        "{\n  \"favoriteGenre\": \"Primary genre with explanation\",\n  \"topArtist\": \"Most significant artist with reason\",\n  \"primaryLanguage\": \"Detected language\",\n  \"commonInstruments\": [\"instrument1\", \"instrument2\", \"instrument3\"],\n  \"musicalSense\": \"3-4 sentence personality description of their musical taste\",\n  \"recommendations\": {\n    \"reason\": \"Why these recommendations fit\",\n    \"searchTerms\": [\"term1\", \"term2\", \"term3\"]\n  }\n}\n\n"
        "Be insightful and personal in the analysis."
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a JSON-only music analyst. Always return valid JSON and nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=400,
        )
        text = resp.choices[0].message.content.strip()
        # Guard: ensure valid JSON
        data = json.loads(text)
        return data
    except Exception as exc:
        # Map rate-limit text to RateLimitError where possible
        err = str(exc).lower()
        if "rate limit" in err or "429" in err:
            raise RateLimitError(str(exc))
        return fallback_analysis(payload)


def generate_llm_ratings(audio_summary: Dict[str, Any]) -> Dict[str, str] | None:
    """Ask the LLM to map numeric features to human labels.

    Returns a dict like {energy: High, danceability: Medium, ...}. If LLM fails or not configured, returns None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    try:
        payload = {k: audio_summary.get(k) for k in ["danceability","energy","valence","tempo","acousticness","instrumentalness","speechiness","liveness","spectral_centroid_mean","librosa_tempo"]}
        prompt = (
            "Given these audio features from Spotify/librosa, assign human-friendly ratings.\n"
            "Return JSON only with keys: danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness, liveness.\n"
            "Rules: use 'Low'/'Medium'/'High' for all except 'tempo' which must be 'Slow'/'Moderate'/'Fast'.\n"
            f"Features: {json.dumps(payload, ensure_ascii=False)}"
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a JSON-only music rater."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)
        # Validate keys
        out: Dict[str, str] = {}
        for k in ["danceability","energy","valence","tempo","acousticness","instrumentalness","speechiness","liveness"]:
            v = data.get(k)
            if isinstance(v, str) and v:
                out[k] = v
        return out
    except Exception:
        return None


def generate_llm_instruments(genres: List[str], ratings: Dict[str, Any], audio_summary: Dict[str, Any]) -> List[str] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    try:
        prompt = (
            "Infer likely instruments present in the user's music from genres, ratings and audio features.\n"
            "Return a JSON array of 3-5 concise instrument names (lowercase).\n"
            f"Genres: {genres}\nRatings: {ratings}\nFeatures: {json.dumps(audio_summary, ensure_ascii=False)}"
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Return only a JSON array of instruments."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        arr = json.loads(resp.choices[0].message.content.strip())
        return [str(x).strip().lower() for x in arr if isinstance(x, (str,))][:5]
    except Exception:
        return None


def fallback_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    audio_summary = payload.get("audio_summary") or {}
    genres = payload.get("genres") or []
    artists = payload.get("artists") or []
    favorite_genre = genres[0] if genres else None
    ratings = {}
    def classify(k):
        v = audio_summary.get(k)
        if v is None:
            return "Medium"
        if k == "tempo":
            return "Slow" if v < 90 else ("Moderate" if v < 120 else "Fast")
        return "Low" if v < 0.33 else ("Medium" if v < 0.66 else "High")
    for k in ["energy","danceability","valence","acousticness","tempo"]:
        ratings[k] = classify(k)
    desc = _describe_personality(ratings, audio_summary, favorite_genre)
    common_instruments = [i for i in ["synthesizer","drums","acoustic guitar","piano"]]
    return {
        "favoriteGenre": favorite_genre or "mixed",
        "topArtist": artists[0] if artists else "various",
        "primaryLanguage": None,
        "commonInstruments": common_instruments,
        "musicalSense": desc,
        "recommendations": {
            "reason": "Based on energy/valence/tempo balance",
            "searchTerms": [favorite_genre or "indie", ratings.get("tempo","Moderate"), "similar artists"]
        }
    }


