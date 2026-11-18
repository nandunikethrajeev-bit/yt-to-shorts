"""Download YouTube videos, metadata, and transcripts using yt-dlp."""

from __future__ import annotations

import html
import http.cookiejar
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass
class TranscriptSegment:
    """A single timestamped transcript cue."""

    start: float
    end: float
    text: str


@dataclass
class DownloadResult:
    """Downloaded video assets and parsed metadata."""

    video_id: str
    title: str
    duration: float
    video_path: Path
    transcript: list[TranscriptSegment] = field(default_factory=list)
    heatmap: list[dict[str, float]] = field(default_factory=list)


_TIMESTAMP_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(hours: str | None, minutes: str, seconds: str, millis: str) -> float:
    h = int(hours) if hours else 0
    return h * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def parse_subtitles_file(sub_path: Path) -> list[TranscriptSegment]:
    """Parse a .vtt or .srt subtitle file into clean, deduplicated segments."""
    raw = sub_path.read_text(encoding="utf-8", errors="replace")
    segments: list[TranscriptSegment] = []

    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        ts_match = None
        text_start_idx = 0
        for idx, line in enumerate(lines):
            ts_match = _TIMESTAMP_RE.search(line)
            if ts_match:
                text_start_idx = idx + 1
                break

        if not ts_match:
            continue

        start = _to_seconds(*ts_match.groups()[:4])
        end = _to_seconds(*ts_match.groups()[4:])
        if end <= start:
            continue

        raw_text_lines = lines[text_start_idx:]
        cleaned_lines: list[str] = []
        for line in raw_text_lines:
            no_tags = _TAG_RE.sub("", line)
            unescaped = html.unescape(no_tags).strip()
            if unescaped:
                cleaned_lines.append(unescaped)

        text = " ".join(cleaned_lines)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue

        # YouTube auto-captions often repeat the previous cue's text in rolling blocks.
        if segments and segments[-1].text == text:
            segments[-1].end = max(segments[-1].end, end)
            continue

        if segments and text.startswith(segments[-1].text) and len(text) > len(segments[-1].text):
            remainder = text[len(segments[-1].text) :].strip()
            if remainder:
                segments.append(TranscriptSegment(start=start, end=end, text=remainder))
            else:
                segments[-1].end = max(segments[-1].end, end)
            continue

        segments.append(TranscriptSegment(start=start, end=end, text=text))

    return segments


def _transcribe_with_faster_whisper(video_path: Path) -> list[TranscriptSegment]:
    """Optional local transcription fallback if faster-whisper is installed."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError:
        return []

    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    raw_segments, _ = model.transcribe(str(video_path), beam_size=1)
    return [
        TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
        for seg in raw_segments
        if seg.text.strip()
    ]


def _find_subtitle_file(work_dir: Path, video_id: str) -> Path | None:
    """Find the best downloaded subtitle file (preferring manual English, then auto)."""
    candidates = sorted(work_dir.glob(f"{video_id}*.vtt")) + sorted(work_dir.glob(f"{video_id}*.srt"))
    if not candidates:
        return None
    for ext in (".en.vtt", ".en.srt"):
        for path in candidates:
            if path.name.endswith(ext):
                return path
    return candidates[0]


class AuthStatus(str, Enum):
    """Genuine YouTube cookie authentication status."""

    NOT_CONFIGURED = "Authentication not configured"
    CONFIGURED = "Cookie file configured"
    PARSED = "Cookie file parsed successfully"
    VERIFIED = "Authentication verified"
    FAILED = "Authentication verification failed"


@dataclass
class CookieCheckResult:
    """Result of parsing or verifying a Netscape cookies.txt file."""

    status: AuthStatus
    detail: str
    cookie_count: int = 0
    has_auth_tokens: bool = False


class DownloadErrorCategory(str, Enum):
    """Structured error classifications for YouTube acquisition."""

    INVALID_URL = "invalid_url"
    BOT_VERIFICATION = "bot_verification"
    INVALID_COOKIES = "invalid_cookies"
    MISSING_PO_TOKEN = "missing_po_token"
    CLOUD_IP_BLOCKED = "cloud_ip_blocked"
    VIDEO_UNAVAILABLE = "video_unavailable"
    PRIVATE_OR_RESTRICTED = "private_or_restricted"
    GEO_RESTRICTED = "geo_restricted"
    UNSUPPORTED_FORMAT = "unsupported_format"
    FFMPEG_ERROR = "ffmpeg_error"
    NETWORK_TIMEOUT = "network_timeout"
    UNKNOWN = "unknown"


class YouTubeDownloadError(RuntimeError):
    """Actionable error raised when YouTube video acquisition fails."""

    def __init__(
        self,
        category: DownloadErrorCategory,
        message: str,
        raw_error: str = "",
        is_transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.user_message = message
        self.raw_error = raw_error
        self.is_transient = is_transient


_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}
_AUTH_COOKIE_NAMES = {
    "LOGIN_INFO",
    "SID",
    "__Secure-1PSID",
    "__Secure-3PSID",
    "SAPISID",
    "__Secure-1PAPISID",
    "__Secure-3PAPISID",
}


def validate_youtube_url(url: str) -> str:
    """Validate and normalize a YouTube URL before calling yt-dlp."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise YouTubeDownloadError(
            DownloadErrorCategory.INVALID_URL,
            "Please enter a YouTube URL.",
        )
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise YouTubeDownloadError(
            DownloadErrorCategory.INVALID_URL,
            "Invalid URL scheme. Please enter a full https:// YouTube URL.",
        )
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise YouTubeDownloadError(
            DownloadErrorCategory.INVALID_URL,
            f"Unsupported URL host '{host or 'unknown'}'. Please provide a valid youtube.com or youtu.be URL.",
        )
    return cleaned


def inspect_cookie_file(cookie_path: Path | None) -> CookieCheckResult:
    """Parse a Netscape cookies.txt file and check for valid, non-expired YouTube auth tokens."""
    if cookie_path is None or not cookie_path.exists():
        return CookieCheckResult(
            status=AuthStatus.NOT_CONFIGURED,
            detail="No cookies.txt configured.",
        )
    try:
        if cookie_path.stat().st_size == 0:
            return CookieCheckResult(
                status=AuthStatus.FAILED,
                detail="Cookie file is empty.",
            )
    except OSError as exc:
        return CookieCheckResult(
            status=AuthStatus.FAILED,
            detail=f"Cannot read cookie file: {exc}",
        )

    jar = http.cookiejar.MozillaCookieJar()
    try:
        jar.load(str(cookie_path), ignore_discard=True, ignore_expires=True)
    except Exception as exc:
        return CookieCheckResult(
            status=AuthStatus.FAILED,
            detail=f"Malformed Netscape cookie file: {exc}",
        )

    now = time.time()
    yt_cookies = [c for c in jar if "youtube.com" in (c.domain or "").lower()]
    if not yt_cookies:
        return CookieCheckResult(
            status=AuthStatus.FAILED,
            detail="Cookie file parsed, but contains no youtube.com cookies.",
            cookie_count=len(jar),
        )

    active_auth = [
        c
        for c in yt_cookies
        if c.name in _AUTH_COOKIE_NAMES and (c.expires is None or c.expires == 0 or c.expires > now)
    ]
    has_login = any(c.name == "LOGIN_INFO" for c in active_auth)
    has_sid = any(c.name in {"SID", "__Secure-1PSID", "__Secure-3PSID"} for c in active_auth)
    has_sapisid = any(c.name in {"SAPISID", "__Secure-1PAPISID", "__Secure-3PAPISID"} for c in active_auth)

    if not (has_login and has_sid and has_sapisid):
        return CookieCheckResult(
            status=AuthStatus.CONFIGURED,
            detail=(
                "Cookie file parsed, but required non-expired YouTube auth tokens "
                "(LOGIN_INFO, SID, SAPISID) are missing or expired."
            ),
            cookie_count=len(yt_cookies),
            has_auth_tokens=False,
        )

    return CookieCheckResult(
        status=AuthStatus.PARSED,
        detail=(
            f"Parsed {len(yt_cookies)} YouTube cookies with valid auth token structure "
            "(not yet verified against YouTube)."
        ),
        cookie_count=len(yt_cookies),
        has_auth_tokens=True,
    )


def verify_youtube_auth(
    cookie_path: Path,
    probe_url: str = "https://www.youtube.com/watch?v=jNQXAC9IVRw",
) -> CookieCheckResult:
    """Verify whether cookies.txt is accepted by YouTube without logging cookie values."""
    parsed = inspect_cookie_file(cookie_path)
    if parsed.status in {AuthStatus.NOT_CONFIGURED, AuthStatus.FAILED} or not parsed.has_auth_tokens:
        return CookieCheckResult(
            status=AuthStatus.FAILED,
            detail=parsed.detail,
            cookie_count=parsed.cookie_count,
            has_auth_tokens=parsed.has_auth_tokens,
        )

    import yt_dlp  # type: ignore[import-untyped]

    logger = _YtDlpLogger()
    with tempfile.TemporaryDirectory(prefix="yt_auth_check_") as tmp_dir:
        tmp_cookie = Path(tmp_dir) / "cookies.txt"
        shutil.copyfile(cookie_path, tmp_cookie)
        opts: dict[str, Any] = {
            "cookiefile": str(tmp_cookie),
            "quiet": True,
            "no_warnings": False,
            "skip_download": True,
            "logger": logger,
            "socket_timeout": 15,
            "remote_components": ["ejs:github"],
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(probe_url, download=False)
        except Exception as exc:
            return CookieCheckResult(
                status=AuthStatus.FAILED,
                detail=f"YouTube rejected authentication check: {exc}",
                cookie_count=parsed.cookie_count,
                has_auth_tokens=parsed.has_auth_tokens,
            )

    if logger.cookies_rejected:
        return CookieCheckResult(
            status=AuthStatus.FAILED,
            detail="YouTube reported that the provided account cookies are rotated or expired.",
            cookie_count=parsed.cookie_count,
            has_auth_tokens=parsed.has_auth_tokens,
        )

    return CookieCheckResult(
        status=AuthStatus.VERIFIED,
        detail="YouTube accepted the session cookies.",
        cookie_count=parsed.cookie_count,
        has_auth_tokens=True,
    )


class _YtDlpLogger:
    """Capture yt-dlp warnings/errors without leaking cookie contents."""

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def debug(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    @property
    def cookies_rejected(self) -> bool:
        return any(
            "cookies are no longer valid" in w.lower() or "rotated" in w.lower()
            for w in self.warnings
        )

    @property
    def po_token_missing(self) -> bool:
        return any("po token" in w.lower() for w in self.warnings)


def _detect_js_runtimes() -> dict[str, Any]:
    """Detect installed JavaScript runtimes for yt-dlp EJS challenge solving."""
    js_runtimes: dict[str, Any] = {}
    for rt in ("node", "deno", "bun", "quickjs"):
        if shutil.which(rt):
            js_runtimes[rt] = {}
    if not js_runtimes:
        default_node = Path(r"C:\Program Files\nodejs\node.exe")
        if default_node.exists():
            js_runtimes["node"] = {"path": str(default_node)}
    return js_runtimes


def validate_video_file(video_path: Path, ffmpeg_exe: str) -> float:
    """Verify downloaded video exists, is non-empty, and is readable by FFmpeg."""
    if not video_path.exists():
        raise YouTubeDownloadError(
            DownloadErrorCategory.FFMPEG_ERROR,
            f"Downloaded video file does not exist: {video_path.name}",
        )
    if video_path.stat().st_size == 0:
        raise YouTubeDownloadError(
            DownloadErrorCategory.FFMPEG_ERROR,
            f"Downloaded video file is empty (0 bytes): {video_path.name}",
        )
    probe = subprocess.run(
        [ffmpeg_exe, "-v", "error", "-i", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    # ffmpeg -i without output exits 1 with stream info; check stderr for fatal open/parse errors
    probe_full = subprocess.run(
        [ffmpeg_exe, "-i", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    stderr = probe_full.stderr or ""
    if "Invalid data found" in stderr or "No such file" in stderr:
        raise YouTubeDownloadError(
            DownloadErrorCategory.FFMPEG_ERROR,
            f"Downloaded file is corrupted or unreadable by FFmpeg: {video_path.name}",
            raw_error=stderr[-500:],
        )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if not match:
        if probe.returncode != 0 and "Video:" not in stderr:
            raise YouTubeDownloadError(
                DownloadErrorCategory.FFMPEG_ERROR,
                f"FFmpeg could not detect a valid video stream in {video_path.name}.",
                raw_error=stderr[-500:],
            )
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _classify_ytdlp_error(
    raw_msg: str,
    cookies_used: bool,
    logger: _YtDlpLogger,
) -> YouTubeDownloadError:
    """Classify a yt-dlp DownloadError into an actionable error category."""
    lower = raw_msg.lower()

    if any(p in lower for p in ("timed out", "timeout", "connection reset", "temporary failure", "503", "502", "429", "too many requests")):
        return YouTubeDownloadError(
            DownloadErrorCategory.NETWORK_TIMEOUT,
            "Network timeout or rate limit while contacting YouTube. Please retry in a moment.",
            raw_error=raw_msg,
            is_transient=True,
        )

    if "po token" in lower or (logger.po_token_missing and "format" in lower):
        return YouTubeDownloadError(
            DownloadErrorCategory.MISSING_PO_TOKEN,
            "YouTube requires a valid PO Token (Proof-of-Origin) for this stream. "
            "Configure YOUTUBE_PO_TOKEN or a bgutil PO Token server (YOUTUBE_POT_SERVER_URL) in Secrets, "
            "or use the Upload Video File tab.",
            raw_error=raw_msg,
        )

    if "confirm your age" in lower or "inappropriate for some users" in lower:
        if cookies_used:
            detail = (
                "YouTube rejected the provided cookies.txt (cookies are expired, missing auth tokens, "
                "or rotated in the browser). Re-export a fresh cookies.txt from an Incognito window "
                "(sign in -> open youtube.com/robots.txt -> export -> close Incognito)."
            )
            if logger.cookies_rejected:
                detail += " (yt-dlp confirmed cookies were rotated/invalidated)"
            return YouTubeDownloadError(
                DownloadErrorCategory.INVALID_COOKIES,
                f"{detail}\n\nyt-dlp error: {raw_msg}",
                raw_error=raw_msg,
            )
        return YouTubeDownloadError(
            DownloadErrorCategory.PRIVATE_OR_RESTRICTED,
            "This YouTube video is age-restricted and requires verified account cookies. "
            f"Upload a valid cookies.txt or use the Upload Video File tab.\n\nyt-dlp error: {raw_msg}",
            raw_error=raw_msg,
        )

    if "not a bot" in lower or ("sign in to confirm" in lower and "confirm your age" not in lower):
        return YouTubeDownloadError(
            DownloadErrorCategory.BOT_VERIFICATION,
            "YouTube triggered bot verification on this server's datacenter IP. "
            "On Streamlit Cloud, datacenter IPs are frequently challenged even with valid cookies. "
            "Please download the video locally and use the 'Upload Video File (.mp4 / .mov)' tab, "
            f"or configure a PO Token provider.\n\nyt-dlp error: {raw_msg}",
            raw_error=raw_msg,
        )

    if "403" in lower or "forbidden" in lower or "page needs to be reloaded" in lower:
        return YouTubeDownloadError(
            DownloadErrorCategory.CLOUD_IP_BLOCKED,
            "YouTube blocked the request from this cloud datacenter IP (HTTP 403 / page reload challenge). "
            "Please use the 'Upload Video File (.mp4 / .mov)' tab for guaranteed processing.\n\n"
            f"yt-dlp error: {raw_msg}",
            raw_error=raw_msg,
        )

    if any(p in lower for p in ("not available in your country", "geo-restricted", "blocked it in your country")):
        return YouTubeDownloadError(
            DownloadErrorCategory.GEO_RESTRICTED,
            f"This YouTube video is geographically restricted in the server's region.\n\nyt-dlp error: {raw_msg}",
            raw_error=raw_msg,
        )

    if any(p in lower for p in ("private video", "members-only", "join this channel")):
        return YouTubeDownloadError(
            DownloadErrorCategory.PRIVATE_OR_RESTRICTED,
            f"This YouTube video is private or restricted.\n\nyt-dlp error: {raw_msg}",
            raw_error=raw_msg,
        )

    if any(p in lower for p in ("video unavailable", "has been removed", "account associated", "does not exist")):
        return YouTubeDownloadError(
            DownloadErrorCategory.VIDEO_UNAVAILABLE,
            f"This YouTube video is unavailable or deleted.\n\nyt-dlp error: {raw_msg}",
            raw_error=raw_msg,
        )

    if "requested format is not available" in lower:
        return YouTubeDownloadError(
            DownloadErrorCategory.UNSUPPORTED_FORMAT,
            f"No compatible video/audio format was returned by YouTube.\n\nyt-dlp error: {raw_msg}",
            raw_error=raw_msg,
        )

    if "ffmpeg" in lower or "ffprobe" in lower:
        return YouTubeDownloadError(
            DownloadErrorCategory.FFMPEG_ERROR,
            f"FFmpeg failed while processing the downloaded streams.\n\nyt-dlp error: {raw_msg}",
            raw_error=raw_msg,
        )

    return YouTubeDownloadError(
        DownloadErrorCategory.UNKNOWN,
        f"Failed to download YouTube video: {raw_msg}",
        raw_error=raw_msg,
    )


def download_youtube_video(
    url: str,
    output_dir: Path,
    ffmpeg_location: str | None = None,
    use_whisper_fallback: bool = True,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
    max_height: int = 1080,
    po_token: str | None = None,
    pot_server_url: str | None = None,
) -> DownloadResult:
    """Download a YouTube video, its subtitles, and heatmap metadata safely."""
    import yt_dlp  # type: ignore[import-untyped]

    clean_url = validate_youtube_url(url)

    if not ffmpeg_location:
        from yt_shorts.editor import resolve_ffmpeg_exe

        ffmpeg_location = resolve_ffmpeg_exe()
    if not Path(ffmpeg_location).exists() and not shutil.which(ffmpeg_location):
        raise YouTubeDownloadError(
            DownloadErrorCategory.FFMPEG_ERROR,
            f"FFmpeg executable not found at: {ffmpeg_location}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    temp_dir = Path(tempfile.mkdtemp(prefix=f"yt_dl_{run_id}_", dir=output_dir))
    out_tmpl = str(temp_dir / "%(id)s.%(ext)s")

    project_root = Path(__file__).resolve().parent.parent.parent
    auto_cookies = project_root / "cookies.txt"
    if not cookies_file and auto_cookies.exists():
        cookies_file = auto_cookies

    resolved_po_token = (po_token or os.environ.get("YOUTUBE_PO_TOKEN") or "").strip()
    resolved_pot_server = (pot_server_url or os.environ.get("YOUTUBE_POT_SERVER_URL") or "").strip()

    is_local = sys.platform in ("win32", "darwin")
    ytdl_logger = _YtDlpLogger()
    if is_local:
        youtube_args: dict[str, Any] = {
            "player_client": ["web", "mweb", "android"],
        }
    else:
        youtube_args = {
            "player_client": ["ios", "android", "mweb"],
            "player_skip": ["configs", "webpage"],
        }
    if resolved_po_token:
        youtube_args["po_token"] = [resolved_po_token]

    extractor_args: dict[str, Any] = {"youtube": youtube_args}
    if resolved_pot_server:
        extractor_args["youtubepot-bgutilhttp"] = {"base_url": [resolved_pot_server]}

    ydl_opts: dict[str, Any] = {
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "merge_output_format": "mp4",
        "outtmpl": out_tmpl,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt/srt/best",
        "writeinfojson": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "logger": ytdl_logger,
        "remote_components": ["ejs:github"],
        "geo_bypass": True,
        "check_formats": False,
        "socket_timeout": 30,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        "extractor_args": extractor_args,
        "ffmpeg_location": ffmpeg_location,
    }
    js_runtimes = _detect_js_runtimes()
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes

    cookies_used = False
    if cookies_file and Path(cookies_file).exists():
        cookie_check = inspect_cookie_file(Path(cookies_file))
        if cookie_check.status == AuthStatus.FAILED:
            raise YouTubeDownloadError(
                DownloadErrorCategory.INVALID_COOKIES,
                f"Invalid cookies.txt: {cookie_check.detail}",
            )
        cookies_used = True
        temp_cookies = temp_dir / "_cookies_session.txt"
        shutil.copyfile(cookies_file, temp_cookies)
        ydl_opts["cookiefile"] = str(temp_cookies)
    elif cookies_from_browser and cookies_from_browser != "none":
        cookies_used = True
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    def _run_extract_with_backoff(opts: dict[str, Any]) -> dict[str, Any]:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    return ydl.extract_info(clean_url, download=True)
            except yt_dlp.utils.DownloadError as err:
                classified = _classify_ytdlp_error(str(err), cookies_used, ytdl_logger)
                if classified.is_transient and attempt < max_attempts - 1:
                    time.sleep(2**attempt)
                    continue
                raise classified from err
        raise YouTubeDownloadError(DownloadErrorCategory.UNKNOWN, "Download failed after retries.")

    try:
        try:
            info = _run_extract_with_backoff(ydl_opts)
        except YouTubeDownloadError as primary_err:
            if primary_err.category in {
                DownloadErrorCategory.BOT_VERIFICATION,
                DownloadErrorCategory.CLOUD_IP_BLOCKED,
            }:
                mobile_fallback_opts = {
                    k: v
                    for k, v in ydl_opts.items()
                    if k not in ("cookiefile", "cookiesfrombrowser")
                }
                mobile_fallback_opts["extractor_args"] = {
                    **extractor_args,
                    "youtube": {
                        "player_client": ["ios", "android"],
                        "player_skip": ["webpage", "configs"],
                        **({"po_token": [resolved_po_token]} if resolved_po_token else {}),
                    },
                }
                info = _run_extract_with_backoff(mobile_fallback_opts)
            elif primary_err.category == DownloadErrorCategory.UNSUPPORTED_FORMAT:
                format_fallback_opts = {**ydl_opts, "format": "best"}
                format_fallback_opts.pop("merge_output_format", None)
                info = _run_extract_with_backoff(format_fallback_opts)
            else:
                raise

        video_id = str(info.get("id") or "video")
        title = str(info.get("title") or video_id)
        duration = float(info.get("duration") or 0.0)
        heatmap = info.get("heatmap") or []

        candidate_path = temp_dir / f"{video_id}.mp4"
        if not candidate_path.exists():
            matches = [
                p
                for p in temp_dir.glob(f"{video_id}.*")
                if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
            ]
            if not matches:
                raise YouTubeDownloadError(
                    DownloadErrorCategory.FFMPEG_ERROR,
                    f"Downloaded video file not found for {video_id}.",
                )
            candidate_path = matches[0]

        probed_duration = validate_video_file(candidate_path, ffmpeg_location)
        if duration <= 0 and probed_duration > 0:
            duration = probed_duration

        unique_video_path = output_dir / f"{video_id}_{run_id}{candidate_path.suffix.lower()}"
        shutil.move(str(candidate_path), str(unique_video_path))

        info_json_path = temp_dir / f"{video_id}.info.json"
        if not heatmap and info_json_path.exists():
            try:
                saved_info = json.loads(info_json_path.read_text(encoding="utf-8"))
                heatmap = saved_info.get("heatmap") or []
            except (OSError, json.JSONDecodeError):
                pass

        sub_path = _find_subtitle_file(temp_dir, video_id)
        transcript = parse_subtitles_file(sub_path) if sub_path else []
        if not transcript and use_whisper_fallback:
            transcript = _transcribe_with_faster_whisper(unique_video_path)

        return DownloadResult(
            video_id=video_id,
            title=title,
            duration=duration,
            video_path=unique_video_path,
            transcript=transcript,
            heatmap=heatmap,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
