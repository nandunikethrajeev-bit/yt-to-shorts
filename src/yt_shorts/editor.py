"""Cut and format 9:16 vertical Shorts clips with optional burned-in captions using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal

from yt_shorts.downloader import TranscriptSegment

LayoutMode = Literal["blur", "crop"]


def resolve_ffmpeg_exe() -> str:
    """Return path to ffmpeg executable (system PATH or bundled imageio-ffmpeg)."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg not found. Install imageio-ffmpeg (`pip install imageio-ffmpeg`) "
            "or add ffmpeg to your system PATH."
        ) from exc


def _format_ass_time(seconds: float) -> str:
    """Format seconds as H:MM:SS.cs for ASS subtitles."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:
        secs += 1
        centis = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def write_clip_ass_subtitles(
    transcript: list[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    ass_path: Path,
) -> bool:
    """Write clip-relative ASS subtitles; returns True if any cues were written."""
    cues: list[tuple[float, float, str]] = []
    clip_duration = clip_end - clip_start

    for seg in transcript:
        if seg.end <= clip_start or seg.start >= clip_end:
            continue
        rel_start = max(0.0, seg.start - clip_start)
        rel_end = min(clip_duration, seg.end - clip_start)
        if rel_end - rel_start < 0.15:
            continue
        text = seg.text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()
        if text:
            cues.append((rel_start, rel_end, text))

    if not cues:
        return False

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: ShortCaption,DejaVu Sans,62,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,4,2,2,60,60,260,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [
        f"Dialogue: 0,{_format_ass_time(s)},{_format_ass_time(e)},ShortCaption,,0,0,0,,{txt}"
        for s, e, txt in cues
    ]
    ass_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return True


def render_short_clip(
    source_video: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    layout: LayoutMode = "blur",
    transcript: list[TranscriptSegment] | None = None,
    burn_captions: bool = True,
    ffmpeg_exe: str | None = None,
) -> Path:
    """Cut and render a 1080x1920 (9:16) vertical clip with optional captions."""
    ffmpeg = ffmpeg_exe or resolve_ffmpeg_exe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1.0, end_time - start_time)

    ass_path = output_path.with_suffix(".ass")
    has_subs = False
    if burn_captions and transcript:
        has_subs = write_clip_ass_subtitles(transcript, start_time, end_time, ass_path)

    if layout == "crop":
        base_filter = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,setsar=1"
        )
    else:
        base_filter = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=20:5[bg_blur];"
            "[fg]scale=1080:-2:force_original_aspect_ratio=decrease[fg_scaled];"
            "[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )

    if has_subs:
        # Use relative subtitle filename with cwd=output_path.parent to avoid Windows ':' escaping issues.
        filter_complex = f"{base_filter},subtitles={ass_path.name}[v]"
    else:
        filter_complex = f"{base_filter}[v]"

    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(source_video.resolve()),
        "-t",
        f"{duration:.3f}",
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path.name,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(output_path.parent.resolve()),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({result.returncode}):\n{result.stderr[-1500:]}")

    if ass_path.exists():
        ass_path.unlink(missing_ok=True)

    return output_path
