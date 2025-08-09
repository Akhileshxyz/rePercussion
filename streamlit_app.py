import os
from typing import Any, Dict

import requests
import streamlit as st


API_BASE = os.getenv("API_BASE", "")  # e.g., https://<your-vercel-app>.vercel.app


def _get_query_params() -> Dict[str, Any]:
    # Streamlit 1.30+: st.query_params; older: experimental_get_query_params
    try:
        return dict(st.query_params)  # type: ignore[attr-defined]
    except Exception:
        return {k: v[0] if isinstance(v, list) and v else v for k, v in st.experimental_get_query_params().items()}


def _clear_query_params():
    try:
        st.query_params.clear()  # type: ignore[attr-defined]
    except Exception:
        st.experimental_set_query_params()


def bootstrap_session_from_query():
    params = _get_query_params()
    access_token = params.get("access_token")
    refresh_token = params.get("refresh_token")
    expires_at = params.get("expires_at")
    if access_token:
        st.session_state["access_token"] = access_token
    if refresh_token:
        st.session_state["refresh_token"] = refresh_token
    if expires_at:
        st.session_state["expires_at"] = expires_at
    if params:
        _clear_query_params()


def call_api_me(token: str) -> Dict[str, Any] | None:
    try:
        resp = requests.get(
            f"{API_BASE}/api/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.ok:
            return resp.json()
        return None
    except Exception:
        return None


def main():
    st.set_page_config(page_title="rePercussion", page_icon="🎵")

    bootstrap_session_from_query()

    st.title("rePercussion 🎵")
    st.caption("Analyze your musical taste using audio features and lyrics")

    access_token = st.session_state.get("access_token")

    if not access_token:
        try:
            # Streamlit >= 1.30
            st.link_button("Login with Spotify", f"{API_BASE}/login")  # type: ignore[attr-defined]
        except Exception:
            st.markdown(f"[Login with Spotify]({API_BASE}/login)")
        st.stop()

    profile = call_api_me(access_token)
    if not profile:
        st.error("Could not fetch your Spotify profile. Please try logging in again.")
        try:
            st.link_button("Re-login with Spotify", f"{API_BASE}/login")  # type: ignore[attr-defined]
        except Exception:
            st.markdown(f"[Re-login with Spotify]({API_BASE}/login)")
        st.stop()

    display_name = profile.get("display_name") or profile.get("id")
    st.success(f"Welcome, {display_name}!")

    tabs = st.tabs(["Liked Songs", "Share a Playlist", "Musical Sense", "Recommendations"])

    with tabs[0]:
        st.info("Placeholder: View and analyze your liked songs.")

    with tabs[1]:
        st.info("Placeholder: Paste a playlist link to analyze and share.")

    with tabs[2]:
        st.info("Placeholder: Audio features and lyric embedding-based analysis.")

    with tabs[3]:
        st.info("Placeholder: Personalized recommendations.")


if __name__ == "__main__":
    main()


