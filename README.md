rePercussion
============

Run locally
-
1) Create `.env` with your Spotify credentials:

```
SPOTIPY_CLIENT_ID="..."
SPOTIPY_CLIENT_SECRET="..."
SPOTIPY_REDIRECT_URI="https://<your-heroku-app>.herokuapp.com/callback"
FRONTEND_URL="https://<your-streamlit-app>.streamlit.app"
```

2) Install deps and run API:

```
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Deploy to Heroku
-
1) Commit and push to a GitHub repo.

2) Create Heroku app (Python):

```
heroku create your-app-name
```

3) Set config vars on Heroku:

```
heroku config:set SPOTIPY_CLIENT_ID=... SPOTIPY_CLIENT_SECRET=... \
  SPOTIPY_REDIRECT_URI="https://your-app-name.herokuapp.com/callback" \
  FRONTEND_URL="https://<your-streamlit-app>.streamlit.app"
```

4) Deploy:

```
git push heroku main
```

5) In Spotify Dashboard, add Redirect URI:

```
https://your-app-name.herokuapp.com/callback
```

