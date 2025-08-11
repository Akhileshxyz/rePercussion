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

        # Extract librosa features
        audio_summary = analysis.get('audio_summary', {})
        librosa_tempo = audio_summary.get('librosa_tempo', 'not available')
        spectral_centroid_mean = audio_summary.get('spectral_centroid_mean', 'not available')
        spectral_bandwidth_mean = audio_summary.get('spectral_bandwidth_mean', 'not available')
        spectral_rolloff_mean = audio_summary.get('spectral_rolloff_mean', 'not available')
        rms_mean = audio_summary.get('rms_mean', 'not available')
        zero_crossing_rate_mean = audio_summary.get('zero_crossing_rate_mean', 'not available')
        dominant_pitch_class = audio_summary.get('dominant_pitch_class', 'not available')
        
        # Prepare librosa feature explanations
        librosa_explanations = f"""
        Advanced Audio Features (from librosa analysis):
        - Librosa Tempo: {librosa_tempo} BPM - The estimated beats per minute using librosa's beat tracking algorithm
        - Spectral Centroid Mean: {spectral_centroid_mean} - Represents the "center of mass" of the spectrum; higher values indicate brighter sound
        - Spectral Bandwidth Mean: {spectral_bandwidth_mean} - Indicates the width of the spectrum; higher values suggest more range in frequencies
        - Spectral Rolloff Mean: {spectral_rolloff_mean} - Frequency below which most of the energy is concentrated; indicates tonal range
        - RMS Mean: {rms_mean} - Root Mean Square energy; indicates overall loudness and dynamic range
        - Zero Crossing Rate Mean: {zero_crossing_rate_mean} - Rate of sign changes; higher values suggest more noise/percussion
        - Dominant Pitch Class: {dominant_pitch_class} - The most common pitch class (musical note) in the audio
        """

        prompt = f"""
        Analyze the following music taste profile and generate a fun, exciting, and personalized summary for a music lover.
        The user's favorite genre is {analysis.get('favorite_genre', 'not available')}.
        Their top artist is {analysis.get('top_artist', 'not available')}.
        They primarily listen to music in {analysis.get('primary_language', 'not available')}.
        
        Their music has the following characteristics:
        - Danceability: {audio_summary.get('danceability', 'not available')}
        - Energy: {audio_summary.get('energy', 'not available')}
        - Valence (Positivity): {audio_summary.get('valence', 'not available')}
        - Tempo: {audio_summary.get('tempo', 'not available')} BPM
        - Acousticness: {audio_summary.get('acousticness', 'not available')}
        - Instrumentalness: {audio_summary.get('instrumentalness', 'not available')}
        
        {librosa_explanations}

        Give them a cool title for their musical taste and a paragraph or two that sounds exciting and insightful. Include specific insights about what the advanced audio features reveal about their music preferences in terms of production style and sonic qualities.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a witty and insightful music expert who can analyze both standard and advanced audio features. You understand signal processing concepts and can translate technical measurements into meaningful musical insights. Your analysis should be fun, personalized, and technically accurate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
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
    """Call OpenAI (or compatible) LLM to craft a personalized summary using our DSP analysis and librosa features.

    Requires OPENAI_API_KEY in environment. Falls back to deterministic description if unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _describe_personality(ratings, audio_summary, favorite_genre)

    try:
        client = OpenAI(api_key=api_key)
        
        # Extract librosa-specific features with enhanced detail
        librosa_features = {k: audio_summary.get(k) for k in [
            "librosa_tempo", "spectral_centroid_mean", "rms_mean", "zcr_mean", 
            "flatness_mean", "rolloff_mean", "onset_mean", "dominant_pitch_class",
            "spectral_bandwidth_mean", "spectral_contrast_mean", "chroma_stft_mean",
            "mfcc_mean", "harmonic_mean", "percussive_mean"
        ] if audio_summary.get(k) is not None}
        
        # Enhanced feature explanations with production implications
        feature_explanations = {
            "librosa_tempo": "Beats per minute detected through signal processing (affects energy perception and groove feel)",
            "spectral_centroid_mean": "Average frequency distribution center - higher values (>3000) indicate brighter/sharper production with emphasis on high frequencies; lower values (<1500) suggest warmer, bass-focused mixes",
            "rms_mean": "Root mean square energy - indicates loudness, dynamic range, and compression level; higher values suggest more aggressive mastering",
            "zcr_mean": "Zero crossing rate - higher values indicate more percussive/noisy content with transient-rich production; lower values suggest smoother, more tonal production",
            "flatness_mean": "Spectral flatness - higher values indicate noise-like textures (distortion, complex timbres); lower values suggest more tonal/harmonic production",
            "rolloff_mean": "Frequency below which 85% of energy is contained - indicates brightness/darkness of the mix and high-frequency content",
            "onset_mean": "Strength of note onsets/attacks - indicates rhythmic clarity, articulation, and transient preservation in production",
            "dominant_pitch_class": "Most common musical note/key - indicates tonal center and harmonic foundation",
            "spectral_bandwidth_mean": "Width of the spectrum - indicates frequency range and richness; wider bandwidth suggests fuller production",
            "spectral_contrast_mean": "Contrast between peaks and valleys in spectrum - higher values indicate clarity and separation in mix; lower values suggest a more blended sound",
            "chroma_stft_mean": "Distribution of energy across pitch classes - indicates harmonic content and complexity",
            "mfcc_mean": "Mel-frequency cepstral coefficients - captures timbral characteristics and instrument/vocal qualities",
            "harmonic_mean": "Energy in harmonic components - indicates melodic content and tonal elements",
            "percussive_mean": "Energy in percussive components - indicates rhythmic emphasis and beat prominence"
        }

        # Production style implications based on feature combinations
        production_implications = []
        if librosa_features.get("spectral_centroid_mean") and librosa_features.get("rms_mean"):
            sc = librosa_features["spectral_centroid_mean"]
            rms = librosa_features["rms_mean"]
            if sc > 3000 and rms > 0.2:
                production_implications.append("Bright, loud production with emphasis on clarity and presence")
            elif sc < 1500 and rms > 0.15:
                production_implications.append("Warm, bass-heavy production with substantial power")
            elif sc > 2500 and rms < 0.1:
                production_implications.append("Bright but dynamic production with preserved transients")
        
        if librosa_features.get("zcr_mean") and librosa_features.get("flatness_mean"):
            zcr = librosa_features["zcr_mean"]
            flat = librosa_features["flatness_mean"]
            if zcr > 0.1 and flat > 0.3:
                production_implications.append("Textured production with complex timbres and noise elements")
            elif zcr < 0.05 and flat < 0.2:
                production_implications.append("Smooth, harmonically rich production with minimal distortion")
        
        # Build a comprehensive prompt with ratings, detailed DSP stats, and production implications
        lines = [
            "You are a concise, vivid music curator with expertise in audio signal processing and music production.",
            "Write a 2-3 sentence personalized description of the listener's taste that includes specific insights about production style and sonic qualities.",
            "Avoid generic platitudes; reference concrete traits, production characteristics, and sonic textures based on the advanced audio features.",
        ]
        if favorite_genre:
            lines.append(f"Favorite genre hint: {favorite_genre}.")
        lines.append(f"Ratings: {ratings}.")
        
        # Include standard audio features
        standard_features = {k: audio_summary.get(k) for k in ["danceability", "energy", "valence", "tempo", "acousticness", "instrumentalness"] if audio_summary.get(k) is not None}
        lines.append(f"Standard Features: {standard_features}.")
        
        # Include librosa features with detailed explanations
        if librosa_features:
            librosa_desc = "Advanced Audio Features (use these to infer production style and sonic qualities):\n"
            for k, v in librosa_features.items():
                if k in feature_explanations:
                    librosa_desc += f"- {k}: {v} ({feature_explanations[k]})\n"
            lines.append(librosa_desc)
        
        # Include production style implications
        if production_implications:
            lines.append(f"Production Style Implications: {', '.join(production_implications)}")
        
        lines.append("Use both standard and advanced features to create a detailed, personalized description that specifically addresses production style (e.g., warm/bright, clean/distorted, sparse/dense) and sonic qualities (e.g., textural elements, frequency balance, dynamic range).")
        lines.append("Return only the description.")
        prompt = "\n".join(lines)

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a music expert and audio engineer who crafts engaging and precise summaries of music taste. You have deep knowledge of audio signal processing, music production techniques, and sonic characteristics. You can analyze both standard streaming metrics and advanced DSP features, translating technical measurements into meaningful musical insights about production style and sonic qualities. Focus on what the advanced features reveal about timbre, texture, dynamics, and frequency content that standard features miss."}, 
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=250,
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
    formatted_tracks = []
    for t in tracks[:20]:
        artist_names = ', '.join(a.get('name', '') for a in t.get('artists', []))
        formatted_tracks.append(f"{t.get('name','')} – {artist_names}")
    top_tracks_text = "; ".join(formatted_tracks)
    
    # Extract librosa-specific audio features for enhanced analysis
    librosa_features = {k: v for k, v in audio_summary.items() if k in [
        "librosa_tempo", "spectral_centroid_mean", "rms_mean", "zcr_mean", 
        "flatness_mean", "rolloff_mean", "onset_mean", "dominant_pitch_class",
        "spectral_bandwidth_mean", "spectral_contrast_mean", "chroma_stft_mean",
        "mfcc_mean", "harmonic_mean", "percussive_mean"
    ] and v is not None}
    
    # Prepare feature explanations for the prompt with more detailed descriptions
    feature_explanations = {
        "librosa_tempo": "Beats per minute detected through signal processing (affects energy perception and groove feel)",
        "spectral_centroid_mean": "Average frequency distribution center - higher values (>3000) indicate brighter/sharper production with emphasis on high frequencies; lower values (<1500) suggest warmer, bass-focused mixes",
        "rms_mean": "Root mean square energy - indicates loudness, dynamic range, and compression level; higher values suggest more aggressive mastering",
        "zcr_mean": "Zero crossing rate - higher values indicate more percussive/noisy content with transient-rich production; lower values suggest smoother, more tonal production",
        "flatness_mean": "Spectral flatness - higher values indicate noise-like textures (distortion, complex timbres); lower values suggest more tonal/harmonic production",
        "rolloff_mean": "Frequency below which 85% of energy is contained - indicates brightness/darkness of the mix and high-frequency content",
        "onset_mean": "Strength of note onsets/attacks - indicates rhythmic clarity, articulation, and transient preservation in production",
        "dominant_pitch_class": "Most common musical note/key - indicates tonal center and harmonic foundation",
        "spectral_bandwidth_mean": "Width of the spectrum - indicates frequency range and richness; wider bandwidth suggests fuller production",
        "spectral_contrast_mean": "Contrast between peaks and valleys in spectrum - higher values indicate clarity and separation in mix; lower values suggest a more blended sound",
        "chroma_stft_mean": "Distribution of energy across pitch classes - indicates harmonic content and complexity",
        "mfcc_mean": "Mel-frequency cepstral coefficients - captures timbral characteristics and instrument/vocal qualities",
        "harmonic_mean": "Energy in harmonic components - indicates melodic content and tonal elements",
        "percussive_mean": "Energy in percussive components - indicates rhythmic emphasis and beat prominence"
    }
    
    # Production style implications based on feature combinations
    production_implications = []
    if librosa_features.get("spectral_centroid_mean") and librosa_features.get("rms_mean"):
        sc = librosa_features["spectral_centroid_mean"]
        rms = librosa_features["rms_mean"]
        if sc > 3000 and rms > 0.2:
            production_implications.append("Bright, loud production with emphasis on clarity and presence")
        elif sc < 1500 and rms > 0.15:
            production_implications.append("Warm, bass-heavy production with substantial power")
        elif sc > 2500 and rms < 0.1:
            production_implications.append("Bright but dynamic production with preserved transients")
    
    if librosa_features.get("zcr_mean") and librosa_features.get("flatness_mean"):
        zcr = librosa_features["zcr_mean"]
        flat = librosa_features["flatness_mean"]
        if zcr > 0.1 and flat > 0.3:
            production_implications.append("Textured production with complex timbres and noise elements")
        elif zcr < 0.05 and flat < 0.2:
            production_implications.append("Smooth, harmonically rich production with minimal distortion")
    
    if librosa_features.get("spectral_bandwidth_mean") and librosa_features.get("spectral_contrast_mean"):
        bw = librosa_features["spectral_bandwidth_mean"]
        contrast = librosa_features["spectral_contrast_mean"]
        if bw > 2000 and contrast > 0.5:
            production_implications.append("Wide frequency spectrum with clear separation between elements")
        elif bw < 1500 and contrast < 0.3:
            production_implications.append("Focused frequency range with blended, cohesive mix")
    
    # Create detailed feature descriptions for the prompt
    librosa_descriptions = []
    for feature, value in librosa_features.items():
        if feature in feature_explanations:
            librosa_descriptions.append(f"- {feature}: {value} ({feature_explanations[feature]})")
    
    librosa_section = "\n".join(librosa_descriptions) if librosa_descriptions else "No advanced audio features available"
    prompt = (
        "As a music expert and audio engineer, analyze this user's musical preferences with particular attention to the advanced audio features:\n\n"
        f"User's Top Tracks: [{top_tracks_text}]\n\n"
        f"Basic Audio Features:\n{json.dumps({k: audio_summary.get(k) for k in ['energy','danceability','valence','acousticness','tempo']}, indent=2, ensure_ascii=False)}\n\n"
        f"Advanced Audio Features (Librosa Analysis):\n{librosa_section}\n\n"
        f"Top Genres: {genres}\n"
        f"Top Artists: {artists}\n\n"
        "The advanced audio features reveal nuances about timbre, rhythm, and musical structure "
        "that basic features might miss. Use these to identify specific instrument characteristics, "
        "production styles, and sonic qualities.\n\n"
        "Pay special attention to:\n"
        "- Spectral features (centroid, bandwidth, rolloff) for brightness/darkness and frequency balance\n"
        "- Dynamic features (RMS, onset strength) for compression, punch, and articulation\n"
        "- Textural features (flatness, ZCR) for tonal vs. noise content and production texture\n"
        "- Harmonic features (chroma, pitch class) for key relationships and harmonic complexity\n\n"
        f"Production Style Implications Based on Feature Combinations:\n{', '.join(production_implications) if production_implications else 'No specific production style implications detected'}\n\n"
        "Please provide analysis in this exact JSON format:\n"
        "{\n"
        "  \"favoriteGenre\": \"Primary genre with explanation\",\n"
        "  \"topArtist\": \"Most significant artist with reason\",\n"
        "  \"primaryLanguage\": \"Detected language\",\n"
        "  \"commonInstruments\": [\"instrument1\", \"instrument2\", \"instrument3\"],\n"
        "  \"musicalSense\": \"3-4 sentence personality description of their musical taste\",\n"
        "  \"advancedInsights\": \"2-3 detailed sentences about what the librosa features reveal about production style (e.g., warm/bright, clean/distorted, sparse/dense) and sonic qualities (e.g., textural elements, frequency balance, dynamic range)\",\n"
        "  \"recommendations\": {\n"
        "    \"reason\": \"Why these recommendations fit\",\n"
        "    \"searchTerms\": [\"term1\", \"term2\", \"term3\"]\n"
        "  }\n"
        "}\n\n"
        "Be insightful and personal in the analysis, focusing on what the advanced audio features reveal about production techniques and sonic characteristics that define the user's music taste."
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a music expert and audio engineer who analyzes both standard and advanced audio features. You have deep knowledge of audio signal processing, music production techniques, and sonic characteristics. You can translate technical audio measurements into meaningful musical insights about production style and sonic qualities. Focus on what the advanced features reveal about timbre, texture, dynamics, and frequency content that standard features miss. Pay special attention to production style implications derived from combinations of features. Always return valid JSON and nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=600,
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
    """Ask the LLM to map numeric features to human labels, incorporating advanced librosa features.

    Returns a dict like {energy: High, danceability: Medium, ...}. If LLM fails or not configured, returns None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    try:
        # Extract standard and librosa features
        standard_features = {k: audio_summary.get(k) for k in ["danceability","energy","valence","tempo","acousticness","instrumentalness","speechiness","liveness"]}
        librosa_features = {k: audio_summary.get(k) for k in ["librosa_tempo","spectral_centroid_mean","rms_mean","zcr_mean","flatness_mean","rolloff_mean","onset_mean","dominant_pitch_class"] if audio_summary.get(k) is not None}
        
        # Prepare feature explanations with more detailed descriptions
        feature_explanations = {
            "librosa_tempo": "Beats per minute detected through signal processing (affects energy perception and groove feel)",
            "spectral_centroid_mean": "Average frequency distribution center - higher values (>3000) indicate brighter/sharper production with emphasis on high frequencies; lower values (<1500) suggest warmer, bass-focused mixes",
            "rms_mean": "Root mean square energy - indicates loudness, dynamic range, and compression level; higher values suggest more aggressive mastering",
            "zcr_mean": "Zero crossing rate - higher values indicate more percussive/noisy content with transient-rich production; lower values suggest smoother, more tonal production",
            "flatness_mean": "Spectral flatness - higher values indicate noise-like textures (distortion, complex timbres); lower values suggest more tonal/harmonic production",
            "rolloff_mean": "Frequency below which 85% of energy is contained - indicates brightness/darkness of the mix and high-frequency content",
            "onset_mean": "Strength of note onsets/attacks - indicates rhythmic clarity, articulation, and transient preservation in production",
            "dominant_pitch_class": "Most common musical note/key - indicates tonal center and harmonic foundation"
        }
        
        # Create explanations for the librosa features
        librosa_explanations = "\nLibrosa Advanced Features:\n" + "\n".join([f"- {k}: {librosa_features[k]} ({feature_explanations[k]})" for k in librosa_features])
        
        prompt = (
            "Given these audio features from Spotify and librosa, assign human-friendly ratings.\n"
            "Return JSON only with keys: danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness, liveness.\n"
            "Rules: use 'Low'/'Medium'/'High' for all except 'tempo' which must be 'Slow'/'Moderate'/'Fast'.\n"
            f"Standard Features: {json.dumps(standard_features, ensure_ascii=False)}\n"
            f"{librosa_explanations}\n\n"
            "Use both standard and advanced features to make more accurate assessments. For example, high spectral centroid suggests higher energy, high RMS suggests higher loudness/energy, etc."
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a music expert who analyzes audio features and provides accurate ratings. You understand both standard streaming platform metrics and advanced signal processing measurements from librosa."},
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
    """Infer likely instruments in the user's music using genres, ratings, and advanced librosa features.
    
    Returns a list of 3-5 instrument names, or None if the LLM fails or is not configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    try:
        # Extract librosa-specific features
        librosa_features = {k: audio_summary.get(k) for k in [
            "librosa_tempo", "spectral_centroid_mean", "rms_mean", "zcr_mean", 
            "flatness_mean", "rolloff_mean", "onset_mean", "dominant_pitch_class"
        ] if audio_summary.get(k) is not None}
        
        # Prepare feature explanations and their instrument implications
        feature_implications = ""
        if librosa_features:
            feature_implications = "\nLibrosa Feature Instrument Implications:\n"
            if "spectral_centroid_mean" in librosa_features:
                sc = librosa_features["spectral_centroid_mean"]
                if sc > 3000:
                    feature_implications += "- High spectral centroid (>3000) suggests bright instruments like cymbals, violins, flutes, trumpets, and high-pitched synthesizers\n"
                elif sc > 1500:
                    feature_implications += "- Medium spectral centroid (1500-3000) suggests balanced instruments like guitars, pianos, saxophones, and mid-range synthesizers\n"
                else:
                    feature_implications += "- Low spectral centroid (<1500) suggests bass-heavy instruments like bass guitar, tuba, cello, baritone saxophone, and sub-bass synthesizers\n"
            
            if "zcr_mean" in librosa_features:
                zcr = librosa_features["zcr_mean"]
                if zcr > 0.1:
                    feature_implications += "- High zero crossing rate (>0.1) suggests percussive instruments (drums, hi-hats), distorted guitars, or instruments with significant noise components\n"
                else:
                    feature_implications += "- Low zero crossing rate (<0.1) suggests smoother instruments like strings, synth pads, organs, or clean electric guitars\n"
            
            if "onset_mean" in librosa_features:
                onset = librosa_features["onset_mean"]
                if onset > 0.2:
                    feature_implications += "- Strong onsets (>0.2) suggest drums, piano, plucked string instruments (guitar, harp), or staccato playing techniques\n"
                else:
                    feature_implications += "- Weaker onsets (<0.2) suggest sustained instruments like strings, pads, ambient synthesizers, or legato playing techniques\n"
            
            if "flatness_mean" in librosa_features:
                flat = librosa_features["flatness_mean"]
                if flat > 0.3:
                    feature_implications += "- High spectral flatness (>0.3) suggests noise-based instruments like drums, percussion, distorted guitars, or experimental electronic sounds\n"
                else:
                    feature_implications += "- Low spectral flatness (<0.3) suggests harmonic instruments like piano, clean guitars, orchestral instruments, or tonal synthesizers\n"
            
            if "rms_mean" in librosa_features:
                rms = librosa_features["rms_mean"]
                if rms > 0.2:
                    feature_implications += "- High RMS energy (>0.2) suggests loud instruments like drums, brass, distorted guitars, or heavily compressed electronic instruments\n"
                else:
                    feature_implications += "- Lower RMS energy (<0.2) suggests softer instruments like acoustic guitar, piano, strings, or ambient synthesizers\n"
        
        prompt = (
            "Infer likely instruments present in the user's music from genres, ratings and audio features.\n"
            "Return a JSON array of 3-5 concise instrument names (lowercase).\n"
            f"Genres: {genres}\nRatings: {ratings}\n"
            f"Standard Features: {json.dumps({k: v for k, v in audio_summary.items() if k not in librosa_features}, ensure_ascii=False)}\n"
            f"Advanced Features: {json.dumps(librosa_features, ensure_ascii=False)}"
            f"{feature_implications}"
            "\n\nUse both genre information and audio characteristics to identify the most likely instruments."
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a music expert who can identify instruments from audio characteristics. You understand how spectral features, rhythm patterns, and genre conventions relate to specific instruments. Pay special attention to the detailed librosa feature implications provided, as they offer valuable insights into the sonic characteristics that suggest specific instrument types. Consider how combinations of features (spectral centroid, zero crossing rate, flatness, RMS, onset strength) can reveal the presence of particular instrument families and specific instruments."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=150,
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
    
    # Extract librosa-specific features for the advancedInsights field
    librosa_features = {k: v for k, v in audio_summary.items() if k in [
        "librosa_tempo", "spectral_centroid_mean", "rms_mean", "zcr_mean", 
        "flatness_mean", "rolloff_mean", "onset_mean", "dominant_pitch_class",
        "spectral_bandwidth_mean", "spectral_contrast_mean", "chroma_stft_mean",
        "mfcc_mean", "harmonic_mean", "percussive_mean"
    ] and v is not None}
    
    # Create a fallback advanced insights description with detailed production style analysis
    advanced_insights = "Based on the audio analysis, "
    
    # Frequency balance and tonal character based on spectral centroid
    if librosa_features.get("spectral_centroid_mean"):
        sc = librosa_features["spectral_centroid_mean"]
        if sc > 3000:
            advanced_insights += "the music has bright, treble-focused production with crisp highs and emphasis on clarity and presence. "
        elif sc > 1500:
            advanced_insights += "the music has balanced frequency content with well-defined mids and natural tonal balance. "
        else:
            advanced_insights += "the music has warm, bass-focused production with rich low-end and subdued high frequencies. "
    
    # Dynamic range and texture based on RMS and ZCR
    if librosa_features.get("rms_mean") and librosa_features.get("zcr_mean"):
        rms = librosa_features["rms_mean"]
        zcr = librosa_features["zcr_mean"]
        if rms > 0.2 and zcr > 0.1:
            advanced_insights += "The tracks feature dynamic, textured production with both percussive elements and tonal richness, suggesting detailed mixing with attention to transient preservation. "
        elif rms > 0.15 and zcr > 0.08:
            advanced_insights += "The production has good dynamic range with moderate compression and a balance of smooth and textured elements. "
        elif rms > 0.2 and zcr < 0.08:
            advanced_insights += "The production is loud and dense with heavy compression but maintains tonal smoothness, suggesting modern mastering techniques. "
        elif rms < 0.15 and zcr < 0.08:
            advanced_insights += "The production style is subtle with gentle dynamics, smooth textures, and likely minimal processing for a natural sound. "
    
    # Spectral characteristics based on flatness and bandwidth
    if librosa_features.get("flatness_mean") and librosa_features.get("spectral_bandwidth_mean"):
        flat = librosa_features["flatness_mean"]
        bw = librosa_features["spectral_bandwidth_mean"]
        if flat > 0.3 and bw > 2000:
            advanced_insights += "The sonic palette includes complex textures and wide frequency content, suggesting layered production with diverse sound sources. "
        elif flat < 0.2 and bw > 2000:
            advanced_insights += "The production features clear harmonic content across a wide frequency range, indicating well-balanced mixing with attention to both clarity and fullness. "
        elif flat < 0.2 and bw < 1500:
            advanced_insights += "The production focuses on tonal purity within a controlled frequency range, suggesting intentional filtering and focused arrangement. "
    
    # Rhythmic characteristics based on tempo and onset strength
    if librosa_features.get("librosa_tempo") and librosa_features.get("onset_mean"):
        tempo = librosa_features["librosa_tempo"]
        onset = librosa_features["onset_mean"]
        if tempo > 120 and onset > 0.2:
            advanced_insights += f"The rhythm analysis shows an energetic {tempo:.1f} BPM groove with pronounced attacks, suggesting tight rhythmic production with emphasis on percussive elements. "
        elif tempo > 120 and onset < 0.2:
            advanced_insights += f"The rhythm flows at {tempo:.1f} BPM with smooth transitions between notes, creating a fluid but driving feel. "
        elif tempo < 100 and onset > 0.2:
            advanced_insights += f"The slower {tempo:.1f} BPM tempo is punctuated by distinct note attacks, creating a deliberate, impactful rhythmic character. "
        elif tempo < 100 and onset < 0.2:
            advanced_insights += f"The relaxed {tempo:.1f} BPM tempo pairs with gentle note transitions, creating an atmospheric, flowing rhythmic foundation. "
        else:
            advanced_insights += f"The rhythm analysis shows a balanced {tempo:.1f} BPM groove with moderate articulation. "
    
    if not librosa_features:
        advanced_insights = "Advanced audio analysis unavailable. Consider analyzing with more audio samples for detailed production insights."
    
    return {
        "favoriteGenre": favorite_genre or "mixed",
        "topArtist": artists[0] if artists else "various",
        "primaryLanguage": None,
        "commonInstruments": common_instruments,
        "musicalSense": desc,
        "advancedInsights": advanced_insights,
        "recommendations": {
            "reason": "Based on energy/valence/tempo balance",
            "searchTerms": [favorite_genre or "indie", ratings.get("tempo","Moderate"), "similar artists"]
        }
    }


