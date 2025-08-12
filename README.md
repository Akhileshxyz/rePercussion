rePercussion
============
rePercussion — concise overview
A Flask web app that connects to Spotify to analyze your music taste and deliver personalized insights and recommendations.
Data sources: Spotify OAuth (liked songs, top tracks/artists, audio features).
Audio analysis: librosa enriches features (tempo, spectral traits) for smarter ratings.
AI summaries (optional): GPT crafts engaging, personalized descriptions and can refine ratings/instrument inference.
Key pages:
/liked-songs: Your saved tracks.
/playlist: Paste a playlist URL/ID for averages, ratings, instruments, and a summary.
/sense: Overall taste profile from your top tracks.
/recommendations: 5–10 unique, non-duplicate song recs tailored to your likes.
APIs (optional): POST /api/llm-summary and POST /api/llm-analyze for programmatic summaries and JSON analyses.

============
Local setup (Windows/PowerShell)
- Install Python 3.11 (x64). Ensure `py -3.11 --version` works.

1) Create and activate a virtual env
```
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip setuptools wheel
```

2) Create `.env` in project root
```
SPOTIPY_CLIENT_ID="<your_spotify_client_id>"
SPOTIPY_CLIENT_SECRET="<your_spotify_client_secret>"
SPOTIPY_REDIRECT_URI="http://127.0.0.1:5000/callback"   # or your ngrok https URL + /callback
FRONTEND_URL=""                                         # optional
OPENAI_API_KEY="<optional_openai_key>"                  # optional for GPT summaries
OPENAI_MODEL="gpt-4o-mini"                              # optional
```
Add the same redirect URL in the Spotify Developer Dashboard.

3) Install dependencies
```
.\.venv\Scripts\pip install -r requirements.txt
```

4) Run the server
- Dev (Flask):
```
.\.venv\Scripts\python -m flask --app app.flask_app run --debug
```
- Production-style (Waitress):
```
.\.venv\Scripts\waitress-serve --host=0.0.0.0 --port=5000 wsgi:app
```

5) (Optional) ngrok tunnel
```
ngrok http 5000 --domain=<your-ngrok-subdomain>.ngrok-free.app
```
Update `SPOTIPY_REDIRECT_URI` to `https://<your-ngrok-subdomain>.ngrok-free.app/callback` and add it to Spotify Dashboard.

6) Use the app
- Visit `http://127.0.0.1:5000`
- `/liked-songs` – your saved tracks
- `/playlist` – paste a playlist URL/ID for analysis and a personalized summary
- `/sense` – analysis from your top tracks (librosa + optional GPT)
- `/recommendations` – 5–10 unique recs based on your Liked Songs

7) Optional LLM endpoints
- POST `/api/llm-summary` with `{ ratings, audio_summary, favorite_genre }` → short GPT summary
- POST `/api/llm-analyze` with `{ tracks, audio_summary, genres, artists }` → structured JSON analysis

Heroku (optional)
```
heroku create <your-app-name>
heroku config:set SPOTIPY_CLIENT_ID=... SPOTIPY_CLIENT_SECRET=... \
  SPOTIPY_REDIRECT_URI="https://<your-app-name>.herokuapp.com/callback" \
  FRONTEND_URL="" OPENAI_API_KEY="..." OPENAI_MODEL="gpt-4o-mini"
git push heroku main
```

