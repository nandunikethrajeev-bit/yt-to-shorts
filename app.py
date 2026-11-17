"""Streamlit web UI for converting long YouTube videos into 15-30s Shorts."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from yt_shorts.pipeline import run_pipeline  # noqa: E402

load_dotenv()


def get_secret(name: str, default: str = "") -> str:
    """Read a config secret from environment variables, falling back to st.secrets."""
    env_val = os.environ.get(name, "").strip()
    if env_val:
        return env_val
    try:
        secret_val = st.secrets.get(name, default)
        return str(secret_val).strip() if secret_val is not None else default
    except Exception:
        return default


def _get_session_cookie_dir() -> Path:
    """Return a session-isolated temporary directory for uploaded/secret cookies."""
    if "session_cookie_dir" not in st.session_state:
        st.session_state["session_cookie_dir"] = tempfile.mkdtemp(prefix="yt_shorts_cookies_")
    cookie_dir = Path(st.session_state["session_cookie_dir"])
    cookie_dir.mkdir(parents=True, exist_ok=True)
    return cookie_dir


st.set_page_config(page_title="YouTube to Shorts (Free)", page_icon="🎬", layout="centered")
st.title("YouTube to Shorts (15–30s)")
st.caption("100% free local converter using yt-dlp, ffmpeg, and your Gemini API key.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)
with col1:
    num_clips = st.slider("Number of Shorts", min_value=1, max_value=8, value=3)
    min_sec, max_sec = st.slider(
        "Clip duration (seconds)", min_value=10, max_value=60, value=(15, 30)
    )
with col2:
    layout = st.selectbox("9:16 Layout", options=["blur", "crop"], index=0)
    quality_label = st.selectbox(
        "Download Quality",
        options=["720p (faster for long streams)", "1080p"],
        index=0,
    )
    max_height = 720 if quality_label.startswith("720p") else 1080
    burn_captions = st.checkbox("Burn in captions", value=True)

default_api_key = get_secret("GEMINI_API_KEY")
with st.expander("Gemini API Settings (Free Tier)", expanded=not bool(default_api_key)):
    api_key = st.text_input(
        "GEMINI_API_KEY (optional — leave blank for free heatmap fallback)",
        value=default_api_key,
        type="password",
    )
    model = st.selectbox(
        "Gemini Model",
        options=["gemini-3.5-flash-lite", "gemini-4-hydrogen"],
        index=0,
    )

session_cookie_file = _get_session_cookie_dir() / "cookies.txt"
secret_cookies_text = get_secret("YOUTUBE_COOKIES")
local_root_cookies = Path(__file__).resolve().parent / "cookies.txt"

if secret_cookies_text and not session_cookie_file.exists():
    session_cookie_file.write_text(secret_cookies_text, encoding="utf-8")

has_session_cookies = session_cookie_file.exists() and session_cookie_file.stat().st_size > 0
has_local_cookies = local_root_cookies.exists() and local_root_cookies.stat().st_size > 0
has_active_cookies = has_session_cookies or has_local_cookies

if has_active_cookies:
    st.success("✅ YouTube cookies are active for this session! Age-restricted videos are unlocked.")
else:
    st.info("💡 **Tip:** Regular YouTube videos work with zero setup. To process **age-restricted** videos, add a `cookies.txt` file below.")

with st.expander("Unlock Age-Restricted & Private Videos (cookies.txt)", expanded=not has_active_cookies):
    st.markdown(
        """
        **Why does this happen?**  
        YouTube blocks unauthenticated downloads of age-restricted videos and often flags cloud datacenter IPs.
        
        **Quick Setup (takes 30 seconds):**
        1. In Brave or Chrome (Incognito/Private window), sign in to YouTube and open `https://www.youtube.com/robots.txt`.
        2. Export cookies with **[Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)**, then close the Incognito window.
        3. Upload the exported `cookies.txt` below (stored only for your current session).
        """
    )
    uploaded_cookies = st.file_uploader(
        "Upload cookies.txt (Session-only)",
        type=["txt"],
        help="Export from YouTube using the 'Get cookies.txt LOCALLY' extension.",
    )
    if uploaded_cookies is not None:
        session_cookie_file.write_bytes(uploaded_cookies.getvalue())
        st.success("Loaded `cookies.txt` for this session! You can now generate shorts from age-restricted videos.")
        has_session_cookies = True
        has_active_cookies = True

    if sys.platform in ("win32", "darwin"):
        cookies_browser = st.selectbox(
            "Or load cookies from browser (local only; requires browser to be closed)",
            options=["none", "brave", "chrome", "edge", "firefox"],
            index=0,
            help="Note: Your browser must be completely closed for direct extraction due to OS file locks.",
        )
    else:
        cookies_browser = "none"

if st.button("Generate Shorts", type="primary", disabled=not url.strip()):
    status_box = st.status("Starting pipeline...", expanded=True)
    cookies_path: Path | None = None
    if uploaded_cookies is not None:
        session_cookie_file.write_bytes(uploaded_cookies.getvalue())
        cookies_path = session_cookie_file
    elif has_session_cookies:
        cookies_path = session_cookie_file
    elif has_local_cookies:
        cookies_path = local_root_cookies

    try:

        result = run_pipeline(
            url=url.strip(),
            output_root=Path("output"),
            num_clips=num_clips,
            min_sec=float(min_sec),
            max_sec=float(max_sec),
            layout=layout,
            burn_captions=burn_captions,
            api_key=api_key.strip() or None,
            model=model,
            cookies_from_browser=None if cookies_browser == "none" else cookies_browser,
            cookies_file=cookies_path,
            max_height=max_height,
            on_progress=lambda msg: status_box.write(msg),
        )
        status_box.update(label=f"Created {len(result.clips)} Shorts!", state="complete")

        for clip in result.clips:
            st.subheader(f"{clip.index}. {clip.hook_title}")
            st.caption(
                f"{clip.start_time:.1f}s – {clip.end_time:.1f}s ({clip.duration:.1f}s) | "
                f"Score: {clip.viral_score}/100 | {clip.reason}"
            )
            st.video(str(clip.file_path))
            st.download_button(
                label=f"Download {clip.file_path.name}",
                data=clip.file_path.read_bytes(),
                file_name=clip.file_path.name,
                mime="video/mp4",
                key=f"dl_{clip.index}",
            )
    except Exception as exc:
        status_box.update(label="Pipeline failed", state="error")
        st.error(str(exc))
