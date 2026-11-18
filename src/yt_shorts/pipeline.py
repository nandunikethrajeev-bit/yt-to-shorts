"""End-to-end YouTube to Shorts conversion pipeline."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from yt_shorts.analyzer import ClipCandidate, select_clips_with_gemini
from yt_shorts.downloader import (
    DownloadResult,
    _transcribe_with_faster_whisper,
    download_youtube_video,
)
from yt_shorts.editor import LayoutMode, render_short_clip, resolve_ffmpeg_exe


@dataclass
class RenderedClip:
    """Metadata and file path for a generated Short."""

    index: int
    file_path: Path
    start_time: float
    end_time: float
    duration: float
    hook_title: str
    reason: str
    viral_score: int


@dataclass
class PipelineResult:
    """Complete output of a YouTube-to-Shorts run."""

    video_id: str
    title: str
    output_dir: Path
    clips: list[RenderedClip]


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len] or "clip"


def _probe_video_duration(video_path: Path, ffmpeg_exe: str) -> float:
    """Probe a local video's duration in seconds using ffmpeg."""
    result = subprocess.run(
        [ffmpeg_exe, "-i", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr or "")
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def run_pipeline(
    url: str | None = None,
    output_root: Path = Path("output"),
    num_clips: int = 3,
    min_sec: float = 15.0,
    max_sec: float = 30.0,
    layout: LayoutMode = "blur",
    burn_captions: bool = True,
    api_key: str | None = None,
    model: str | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
    max_height: int = 1080,
    cleanup_cache: bool = True,
    local_video_path: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Download a YouTube video (or use a local file), pick highlights, and render 9:16 Shorts."""
    notify = on_progress or (lambda _msg: None)

    ffmpeg_exe = resolve_ffmpeg_exe()
    work_dir = output_root / "_cache"
    work_dir.mkdir(parents=True, exist_ok=True)

    if local_video_path is not None:
        if not local_video_path.exists():
            raise FileNotFoundError(f"Uploaded video file not found: {local_video_path}")
        notify(f"Inspecting uploaded video '{local_video_path.name}'...")
        duration = _probe_video_duration(local_video_path, ffmpeg_exe)
        transcript = _transcribe_with_faster_whisper(local_video_path)
        video_id = _slugify(local_video_path.stem, max_len=24) or "uploaded_video"
        dl = DownloadResult(
            video_id=video_id,
            title=local_video_path.stem,
            duration=duration,
            video_path=local_video_path,
            transcript=transcript,
            heatmap=[],
        )
        notify(f"Loaded '{dl.title}' ({dl.duration:.0f}s). Selecting top {num_clips} clips...")
    elif url:
        notify("Downloading video and captions with yt-dlp...")
        dl = download_youtube_video(
            url=url,
            output_dir=work_dir,
            ffmpeg_location=ffmpeg_exe,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
            max_height=max_height,
        )
        notify(f"Downloaded '{dl.title}' ({dl.duration:.0f}s). Selecting top {num_clips} clips...")
    else:
        raise ValueError("Either a YouTube URL or a local video file must be provided.")
    candidates: list[ClipCandidate] = select_clips_with_gemini(
        title=dl.title,
        video_duration=dl.duration,
        transcript=dl.transcript,
        heatmap=dl.heatmap,
        num_clips=num_clips,
        min_sec=min_sec,
        max_sec=max_sec,
        api_key=api_key,
        model=model,
    )

    video_out_dir = output_root / dl.video_id
    video_out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedClip] = []

    for idx, clip in enumerate(candidates, start=1):
        notify(
            f"Rendering clip {idx}/{len(candidates)}: "
            f"{clip.hook_title} ({clip.start_time:.1f}s - {clip.end_time:.1f}s)..."
        )
        filename = f"short_{idx:02d}_{_slugify(clip.hook_title)}.mp4"
        out_path = video_out_dir / filename
        render_short_clip(
            source_video=dl.video_path,
            output_path=out_path,
            start_time=clip.start_time,
            end_time=clip.end_time,
            layout=layout,
            transcript=dl.transcript,
            burn_captions=burn_captions,
            ffmpeg_exe=ffmpeg_exe,
        )
        rendered.append(
            RenderedClip(
                index=idx,
                file_path=out_path,
                start_time=clip.start_time,
                end_time=clip.end_time,
                duration=round(clip.end_time - clip.start_time, 2),
                hook_title=clip.hook_title,
                reason=clip.reason,
                viral_score=clip.viral_score,
            )
        )

    summary_path = video_out_dir / "clips_summary.json"
    summary_data = {
        "video_id": dl.video_id,
        "title": dl.title,
        "clips": [{**asdict(c), "file_path": str(c.file_path)} for c in rendered],
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    if cleanup_cache:
        if local_video_path is None or dl.video_path.resolve().parent == work_dir.resolve():
            dl.video_path.unlink(missing_ok=True)
        for temp_file in work_dir.iterdir():
            if temp_file.is_file():
                temp_file.unlink(missing_ok=True)

    notify(f"Done! Saved {len(rendered)} Shorts to {video_out_dir}")

    return PipelineResult(
        video_id=dl.video_id,
        title=dl.title,
        output_dir=video_out_dir,
        clips=rendered,
    )
