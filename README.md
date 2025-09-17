# rePercussion

## Brief Explanation

rePercussion is a Flask web app that connects to Spotify to analyze your music taste and deliver personalized insights and recommendations.

-   **Data sources**: Spotify OAuth (liked songs, top tracks/artists, audio features).
-   **Audio analysis**: `librosa` enriches features (tempo, spectral traits) for smarter ratings.
-   **AI summaries (optional)**: GPT crafts engaging, personalized descriptions and can refine ratings/instrument inference.

### Key Pages

-   `/liked-songs`: Your saved tracks.
-   `/playlist`: Paste a playlist URL/ID for averages, ratings, instruments, and a summary.
-   `/sense`: Overall taste profile from your top tracks.
-   `/recommendations`: 5–10 unique, non-duplicate song recs tailored to your likes.

---

## How to Run

### Prerequisites

-   Python 3.11 or later
-   A Spotify Developer account and API credentials

### 1. Set Up Environment

First, clone the repository and create a virtual environment:

```bash
git clone <repository-url>
cd <repository-name>
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 2. Configure Credentials

Create a `.env` file in the project root and add your Spotify API credentials. You can also add an optional OpenAI key for AI-powered summaries.

```
SPOTIPY_CLIENT_ID="<your_spotify_client_id>"
SPOTIPY_CLIENT_SECRET="<your_spotify_client_secret>"
SPOTIPY_REDIRECT_URI="http://127.0.0.1:5000/callback"
OPENAI_API_KEY="<optional_openai_key>"
OPENAI_MODEL="gpt-4o-mini"
```

> **Note**: You must add your `SPOTIPY_REDIRECT_URI` to the settings on your Spotify Developer Dashboard.

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

You can run the app using the built-in Flask development server:

```bash
flask --app app.flask_app run --debug
```

The application will be available at `http://127.0.0.1:5000`.
