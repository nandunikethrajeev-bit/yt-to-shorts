"""Download YouTube videos, metadata, and transcripts using yt-dlp."""

from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import dataclass, field
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


class _YtDlpLogger:
    """Capture yt-dlp warnings/errors so cookie rejection reasons are preserved."""

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


def download_youtube_video(
    url: str,
    output_dir: Path,
    ffmpeg_location: str | None = None,
    use_whisper_fallback: bool = True,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
    max_height: int = 1080,
) -> DownloadResult:
    """Download a YouTube video, its subtitles, and heatmap metadata."""
    import yt_dlp  # type: ignore[import-untyped]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(output_dir / "%(id)s.%(ext)s")

    if not ffmpeg_location:
        try:
            from yt_shorts.editor import resolve_ffmpeg_exe

            ffmpeg_location = resolve_ffmpeg_exe()
        except Exception:
            pass

    js_runtimes: dict[str, Any] = {}
    for rt in ("node", "deno", "bun", "quickjs"):
        if shutil.which(rt):
            js_runtimes[rt] = {}
    if not js_runtimes:
        default_node = Path(r"C:\Program Files\nodejs\node.exe")
        if default_node.exists():
            js_runtimes["node"] = {"path": str(default_node)}

    ytdl_logger = _YtDlpLogger()
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
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "android", "mweb"],
                "player_skip": ["configs", "webpage"],
            }
        },
    }
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location
    project_root = Path(__file__).resolve().parent.parent.parent
    auto_cookies = project_root / "cookies.txt"
    if not cookies_file and auto_cookies.exists():
        cookies_file = auto_cookies

    cookies_used = False
    if cookies_file and Path(cookies_file).exists():
        cookies_used = True
        temp_cookies = output_dir / "_cookies_session.txt"
        try:
            shutil.copyfile(cookies_file, temp_cookies)
            ydl_opts["cookiefile"] = str(temp_cookies)
        except Exception:
            ydl_opts["cookiefile"] = str(cookies_file)
    elif cookies_from_browser and cookies_from_browser != "none":
        cookies_used = True
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        # Attempt 1: Standard download with configured options.
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc)
        lower_msg = msg.lower()
        is_bot_or_403 = any(
            phrase in lower_msg
            for phrase in ["bot", "page needs to be reloaded", "403", "forbidden"]
        )
        if is_bot_or_403:
            # Attempt 2: Retry purely with native mobile clients (ios/android) without web cookies
            # so desktop session cookies don't trigger mobile token mismatches on public videos.
            mobile_fallback_opts = {
                k: v
                for k, v in ydl_opts.items()
                if k not in ("cookiefile", "cookiesfrombrowser")
            }
            mobile_fallback_opts["extractor_args"] = {
                "youtube": {
                    "player_client": ["ios", "android"],
                    "player_skip": ["webpage", "configs"],
                }
            }
            try:
                with yt_dlp.YoutubeDL(mobile_fallback_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
            except yt_dlp.utils.DownloadError as retry_exc:
                raise RuntimeError(f"Failed to download YouTube video: {retry_exc}") from retry_exc
        elif "requested format is not available" in lower_msg:
            fallback_opts = {**ydl_opts, "format": "best"}
            fallback_opts.pop("merge_output_format", None)
            try:
                with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
            except yt_dlp.utils.DownloadError as retry_exc:
                raise RuntimeError(f"Failed to download YouTube video: {retry_exc}") from retry_exc
        elif "confirm your age" in msg.lower() or "inappropriate for some users" in msg.lower():
            if cookies_used:
                reason = (
                    "YouTube rejected the provided cookies.txt (cookies are expired, missing auth tokens, "
                    "or were rotated in the browser). Please re-export a fresh cookies.txt using an "
                    "Incognito/Private window (sign in -> open youtube.com/robots.txt -> export -> close Incognito)."
                )
                if ytdl_logger.cookies_rejected:
                    reason += " (yt-dlp confirmed cookies were rotated/invalidated)"
                raise RuntimeError(f"{reason}\n\nyt-dlp error: {msg}") from exc
            raise RuntimeError(
                "This YouTube video is age-restricted and requires sign-in verification. "
                "Please export and upload a valid cookies.txt file in the 'YouTube Cookies' section below, "
                f"or try a public, non-age-restricted video.\n\nyt-dlp error: {msg}"
            ) from exc
        elif "private video" in msg.lower() or "video unavailable" in msg.lower():
            raise RuntimeError(f"This YouTube video is private or unavailable.\n\nyt-dlp error: {msg}") from exc
        else:
            raise RuntimeError(f"Failed to download YouTube video: {msg}") from exc

    video_id = str(info.get("id") or "video")
    title = str(info.get("title") or video_id)
    duration = float(info.get("duration") or 0.0)
    heatmap = info.get("heatmap") or []

    video_path = output_dir / f"{video_id}.mp4"
    if not video_path.exists():
        # Fallback if yt-dlp saved another extension
        matches = [
            p
            for p in output_dir.glob(f"{video_id}.*")
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        ]
        if not matches:
            raise FileNotFoundError(f"Downloaded video file not found for {video_id} in {output_dir}")
        video_path = matches[0]

    info_json_path = output_dir / f"{video_id}.info.json"
    if not heatmap and info_json_path.exists():
        try:
            saved_info = json.loads(info_json_path.read_text(encoding="utf-8"))
            heatmap = saved_info.get("heatmap") or []
        except (OSError, json.JSONDecodeError):
            pass

    sub_path = _find_subtitle_file(output_dir, video_id)
    transcript = parse_subtitles_file(sub_path) if sub_path else []
    if not transcript and use_whisper_fallback:
        transcript = _transcribe_with_faster_whisper(video_path)

    return DownloadResult(
        video_id=video_id,
        title=title,
        duration=duration,
        video_path=video_path,
        transcript=transcript,
        heatmap=heatmap,
    )
