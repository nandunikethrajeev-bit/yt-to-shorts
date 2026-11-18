"""Streamlit web UI for converting long YouTube videos into 15-30s Shorts."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from yt_shorts.downloader import (
    AuthStatus,
    DownloadErrorCategory,
    YouTubeDownloadError,
    inspect_cookie_file,
    verify_youtube_auth,
)  # noqa: E402
from yt_shorts.pipeline import PipelineResult, run_pipeline  # noqa: E402

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


def _render_clips(result: PipelineResult, key_prefix: str) -> None:
    """Render generated Shorts clips and download buttons."""
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
            key=f"{key_prefix}_dl_{clip.index}",
        )


st.set_page_config(page_title="YouTube to Shorts (Free)", page_icon="🎬", layout="centered")
st.title("YouTube to Shorts (15–30s)")
st.caption("100% free converter using yt-dlp, ffmpeg, and your Gemini API key.")

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

tab_yt, tab_upload = st.tabs(["YouTube URL", "Upload Video File (.mp4 / .mov)"])

po_token = get_secret("YOUTUBE_PO_TOKEN") or None
pot_server_url = get_secret("YOUTUBE_POT_SERVER_URL") or None

with tab_yt:
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    session_cookie_file = _get_session_cookie_dir() / "cookies.txt"
    secret_cookies_text = get_secret("YOUTUBE_COOKIES")
    local_root_cookies = Path(__file__).resolve().parent / "cookies.txt"

    if secret_cookies_text and not session_cookie_file.exists():
        session_cookie_file.write_text(secret_cookies_text, encoding="utf-8")

    active_cookie_file: Path | None = None
    if session_cookie_file.exists() and session_cookie_file.stat().st_size > 0:
        active_cookie_file = session_cookie_file
    elif local_root_cookies.exists() and local_root_cookies.stat().st_size > 0:
        active_cookie_file = local_root_cookies

    cookie_check = inspect_cookie_file(active_cookie_file)
    verified_status = st.session_state.get("yt_auth_result")

    def _render_auth_banner(check_res: Any) -> None:
        if check_res.status == AuthStatus.VERIFIED:
            st.success(f"🟢 **{check_res.status.value}** — {check_res.detail}")
        elif check_res.status == AuthStatus.PARSED:
            st.info(f"🟡 **{check_res.status.value}** — {check_res.detail}")
        elif check_res.status == AuthStatus.CONFIGURED:
            st.warning(f"🟠 **{check_res.status.value}** — {check_res.detail}")
        elif check_res.status == AuthStatus.FAILED:
            st.error(f"🔴 **{check_res.status.value}** — {check_res.detail}")
        else:
            st.info(f"⚪ **{check_res.status.value}** — Regular public videos work without cookies; upload `cookies.txt` for age-restricted videos.")

    if verified_status is not None and active_cookie_file is not None:
        _render_auth_banner(verified_status)
    else:
        _render_auth_banner(cookie_check)

    with st.expander(
        "YouTube Authentication (cookies.txt)",
        expanded=cookie_check.status != AuthStatus.PARSED,
    ):
        st.markdown(
            """
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
            active_cookie_file = session_cookie_file
            cookie_check = inspect_cookie_file(active_cookie_file)
            st.session_state.pop("yt_auth_result", None)
            _render_auth_banner(cookie_check)

        if active_cookie_file is not None and st.button("Verify YouTube Cookies", key="btn_verify_cookies"):
            with st.spinner("Verifying cookies against YouTube..."):
                st.session_state["yt_auth_result"] = verify_youtube_auth(active_cookie_file)
            st.rerun()

        if sys.platform in ("win32", "darwin"):
            cookies_browser = st.selectbox(
                "Or load cookies from browser (local only; requires browser to be closed)",
                options=["none", "brave", "chrome", "edge", "firefox"],
                index=0,
                help="Note: Your browser must be completely closed for direct extraction due to OS file locks.",
            )
        else:
            cookies_browser = "none"

    if st.button("Generate Shorts from YouTube", type="primary", disabled=not url.strip(), key="btn_yt"):
        status_box = st.status("Starting pipeline...", expanded=True)
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
                cookies_file=active_cookie_file,
                max_height=max_height,
                po_token=po_token,
                pot_server_url=pot_server_url,
                on_progress=lambda msg: status_box.write(msg),
            )
            st.session_state["last_yt_result"] = result
            st.session_state.pop("last_yt_error", None)
            if active_cookie_file is not None:
                st.session_state["yt_auth_result"] = inspect_cookie_file(active_cookie_file)
            status_box.update(label=f"Created {len(result.clips)} Shorts!", state="complete")
        except YouTubeDownloadError as exc:
            status_box.update(label="Pipeline failed", state="error")
            if exc.category in {
                DownloadErrorCategory.INVALID_COOKIES,
            } and active_cookie_file is not None:
                from yt_shorts.downloader import CookieCheckResult

                st.session_state["yt_auth_result"] = CookieCheckResult(
                    status=AuthStatus.FAILED,
                    detail="YouTube rejected the provided cookies.",
                )
            st.session_state["last_yt_error"] = (exc.category.value, str(exc))
        except Exception as exc:
            status_box.update(label="Pipeline failed", state="error")
            st.session_state["last_yt_error"] = ("error", str(exc))

    if "last_yt_error" in st.session_state:
        err_cat, err_msg = st.session_state["last_yt_error"]
        st.error(err_msg)
        if err_cat in {
            DownloadErrorCategory.BOT_VERIFICATION.value,
            DownloadErrorCategory.CLOUD_IP_BLOCKED.value,
            DownloadErrorCategory.MISSING_PO_TOKEN.value,
        }:
            st.warning(
                "**Streamlit Cloud Datacenter IP Notice:** Shared cloud IPs are frequently blocked by YouTube "
                "even with valid cookies. You can switch immediately to the **Upload Video File (.mp4 / .mov)** "
                "tab above to clip any local video with 100% reliability."
            )

    if "last_yt_result" in st.session_state:
        _render_clips(st.session_state["last_yt_result"], key_prefix="yt")

with tab_upload:
    st.info("Upload any `.mp4`, `.mov`, `.mkv`, or `.webm` video directly to bypass YouTube download filters completely.")
    uploaded_video = st.file_uploader(
        "Upload Video File (.mp4 / .mov)",
        type=["mp4", "mov", "mkv", "webm"],
        help="Your video is processed locally in the container and temporary files are cleaned up after rendering.",
    )
    if st.button("Generate Shorts from Uploaded Video", type="primary", disabled=uploaded_video is None, key="btn_upload"):
        assert uploaded_video is not None
        status_box = st.status("Saving uploaded video...", expanded=True)
        try:
            cache_dir = Path("output") / "_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(uploaded_video.name).name
            local_video_path = cache_dir / safe_name
            local_video_path.write_bytes(uploaded_video.getvalue())

            result = run_pipeline(
                local_video_path=local_video_path,
                output_root=Path("output"),
                num_clips=num_clips,
                min_sec=float(min_sec),
                max_sec=float(max_sec),
                layout=layout,
                burn_captions=burn_captions,
                api_key=api_key.strip() or None,
                model=model,
                on_progress=lambda msg: status_box.write(msg),
            )
            st.session_state["last_upload_result"] = result
            status_box.update(label=f"Created {len(result.clips)} Shorts!", state="complete")
        except Exception as exc:
            status_box.update(label="Pipeline failed", state="error")
            st.error(str(exc))

    if "last_upload_result" in st.session_state:
        _render_clips(st.session_state["last_upload_result"], key_prefix="upload")
