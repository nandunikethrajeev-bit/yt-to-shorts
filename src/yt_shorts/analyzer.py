"""Highlight selection using Gemini Interactions API (free tier) or heatmap fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from yt_shorts.downloader import TranscriptSegment


@dataclass
class ClipCandidate:
    """A validated 15-30s highlight clip candidate."""

    start_time: float
    end_time: float
    hook_title: str
    reason: str
    viral_score: int


_CLIP_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number", "description": "Clip start time in seconds."},
                    "end_time": {"type": "number", "description": "Clip end time in seconds."},
                    "hook_title": {"type": "string", "description": "Catchy short title (max 8 words) for the Short."},
                    "reason": {"type": "string", "description": "Why this 15-30s moment works well as a standalone Short."},
                    "viral_score": {"type": "integer", "description": "Estimated engagement score 1-100."},
                },
                "required": ["start_time", "end_time", "hook_title", "reason", "viral_score"],
            },
        }
    },
    "required": ["clips"],
}


def _clamp_and_dedupe_clips(
    raw_clips: list[ClipCandidate],
    video_duration: float,
    num_clips: int,
    min_sec: float,
    max_sec: float,
) -> list[ClipCandidate]:
    """Ensure clips are valid, within [min_sec, max_sec], and non-overlapping."""
    valid: list[ClipCandidate] = []
    target_mid = (min_sec + max_sec) / 2.0

    for clip in sorted(raw_clips, key=lambda c: c.viral_score, reverse=True):
        start = max(0.0, float(clip.start_time))
        end = float(clip.end_time)
        if video_duration > 0:
            end = min(end, video_duration)

        dur = end - start
        if dur < min_sec:
            end = start + min(target_mid, max_sec)
            if video_duration > 0 and end > video_duration:
                end = video_duration
                start = max(0.0, end - min_sec)
        elif dur > max_sec:
            end = start + max_sec

        if (end - start) < max(5.0, min_sec * 0.6):
            continue

        overlaps = any(not (end <= existing.start_time + 1.0 or start >= existing.end_time - 1.0) for existing in valid)
        if overlaps:
            continue

        valid.append(
            ClipCandidate(
                start_time=round(start, 2),
                end_time=round(end, 2),
                hook_title=clip.hook_title.strip() or f"Clip {len(valid) + 1}",
                reason=clip.reason.strip() or "Selected highlight segment.",
                viral_score=max(1, min(100, int(clip.viral_score))),
            )
        )
        if len(valid) >= num_clips:
            break

    return sorted(valid, key=lambda c: c.start_time)


def select_clips_fallback(
    video_duration: float,
    heatmap: list[dict[str, float]],
    num_clips: int = 3,
    min_sec: float = 15.0,
    max_sec: float = 30.0,
) -> list[ClipCandidate]:
    """Free offline fallback using YouTube 'Most Replayed' heatmap or even spacing."""
    clip_len = min(max_sec, max(min_sec, (min_sec + max_sec) / 2.0))
    candidates: list[ClipCandidate] = []

    if heatmap:
        sorted_peaks = sorted(heatmap, key=lambda h: float(h.get("value", 0.0)), reverse=True)
        for idx, peak in enumerate(sorted_peaks):
            p_start = float(peak.get("start_time", 0.0))
            p_end = float(peak.get("end_time", p_start + 5.0))
            center = (p_start + p_end) / 2.0
            start = max(0.0, center - clip_len / 2.0)
            end = start + clip_len
            if video_duration > 0 and end > video_duration:
                end = video_duration
                start = max(0.0, end - clip_len)
            score = int(round(min(100.0, max(50.0, float(peak.get("value", 0.5)) * 100))))
            candidates.append(
                ClipCandidate(
                    start_time=start,
                    end_time=end,
                    hook_title=f"Most Replayed Moment #{idx + 1}",
                    reason="High viewer replay density on YouTube heatmap.",
                    viral_score=score,
                )
            )
    else:
        total = video_duration if video_duration > clip_len else clip_len * num_clips
        step = max(clip_len + 2.0, total / max(1, num_clips + 1))
        for i in range(num_clips):
            start = min(max(0.0, (i + 1) * step - clip_len / 2.0), max(0.0, total - clip_len))
            candidates.append(
                ClipCandidate(
                    start_time=start,
                    end_time=start + clip_len,
                    hook_title=f"Highlight #{i + 1}",
                    reason="Evenly spaced fallback segment.",
                    viral_score=70 - i * 5,
                )
            )

    return _clamp_and_dedupe_clips(candidates, video_duration, num_clips, min_sec, max_sec)


def _chunk_transcript(
    transcript: list[TranscriptSegment],
    max_chars: int = 50000,
) -> list[tuple[float, float, str]]:
    """Split a timestamped transcript into sequential chunks without omitting the middle."""
    if not transcript:
        return []

    chunks: list[tuple[float, float, str]] = []
    current_lines: list[str] = []
    current_chars = 0
    chunk_start = transcript[0].start
    chunk_end = transcript[0].end

    for seg in transcript:
        line = f"[{seg.start:.1f}s-{seg.end:.1f}s] {seg.text}"
        line_len = len(line) + 1  # +1 for newline
        if current_lines and (current_chars + line_len) > max_chars:
            chunks.append((chunk_start, chunk_end, "\n".join(current_lines)))
            current_lines = [line]
            current_chars = line_len
            chunk_start = seg.start
            chunk_end = seg.end
        else:
            if not current_lines:
                chunk_start = seg.start
            current_lines.append(line)
            current_chars += line_len
            chunk_end = max(chunk_end, seg.end)

    if current_lines:
        chunks.append((chunk_start, chunk_end, "\n".join(current_lines)))

    return chunks


def _select_clips_for_chunk(
    client: Any,
    model: str,
    title: str,
    video_duration: float,
    chunk_start: float,
    chunk_end: float,
    chunk_text: str,
    heatmap: list[dict[str, float]],
    num_clips: int,
    min_sec: float,
    max_sec: float,
) -> list[ClipCandidate]:
    """Run Gemini selection on a single transcript chunk."""
    heatmap_note = ""
    if heatmap:
        in_range = [
            p
            for p in heatmap
            if float(p.get("end_time", 0.0)) >= chunk_start - 5.0
            and float(p.get("start_time", 0.0)) <= chunk_end + 5.0
        ]
        peaks_pool = in_range or heatmap
        top_peaks = sorted(peaks_pool, key=lambda h: float(h.get("value", 0.0)), reverse=True)[:8]
        peaks_str = ", ".join(
            f"{float(p.get('start_time', 0)):.1f}s-{float(p.get('end_time', 0)):.1f}s (score {float(p.get('value', 0)):.2f})"
            for p in top_peaks
        )
        heatmap_note = f"\nYouTube Most Replayed peaks: {peaks_str}\n"

    prompt = (
        f"You are an expert short-form video editor (YouTube Shorts / Reels).\n"
        f"Analyze the timestamped transcript segment below for the video titled: {title!r} "
        f"(total video duration: {video_duration:.1f}s; this segment covers {chunk_start:.1f}s to {chunk_end:.1f}s).\n"
        f"{heatmap_note}\n"
        f"Select the top {num_clips} most engaging, self-contained moments from this segment that work as standalone Shorts.\n"
        f"Rules:\n"
        f"- Each clip MUST be strictly between {min_sec:.0f} and {max_sec:.0f} seconds long (end_time - start_time).\n"
        f"- Start on a strong hook or complete sentence; end cleanly at a natural pause.\n"
        f"- Clips must NOT overlap.\n"
        f"- All timestamps must be within {chunk_start:.1f} and {min(video_duration, chunk_end) if video_duration > 0 else chunk_end:.1f} seconds.\n\n"
        f"Transcript:\n{chunk_text}"
    )

    interaction = client.interactions.create(
        model=model,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": _CLIP_LIST_SCHEMA,
        },
        store=False,
    )

    raw_json = interaction.output_text or "{}"
    parsed: dict[str, Any] = json.loads(raw_json)
    items = parsed.get("clips") or []
    return [
        ClipCandidate(
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            hook_title=str(item.get("hook_title", "")),
            reason=str(item.get("reason", "")),
            viral_score=int(item.get("viral_score", 75)),
        )
        for item in items
    ]


def select_clips_with_gemini(
    title: str,
    video_duration: float,
    transcript: list[TranscriptSegment],
    heatmap: list[dict[str, float]] | None = None,
    num_clips: int = 3,
    min_sec: float = 15.0,
    max_sec: float = 30.0,
    api_key: str | None = None,
    model: str | None = None,
) -> list[ClipCandidate]:
    """Select the top engaging 15-30s clips from transcript + heatmap using Gemini."""
    resolved_key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not resolved_key or not transcript:
        return select_clips_fallback(video_duration, heatmap or [], num_clips, min_sec, max_sec)

    from google import genai

    resolved_model = (model or os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite").strip()
    client = genai.Client(api_key=resolved_key)

    chunks = _chunk_transcript(transcript)
    per_chunk_clips = num_clips if len(chunks) == 1 else max(2, min(num_clips, 4))
    all_candidates: list[ClipCandidate] = []
    heatmap_list = heatmap or []

    for chunk_start, chunk_end, chunk_text in chunks:
        all_candidates.extend(
            _select_clips_for_chunk(
                client=client,
                model=resolved_model,
                title=title,
                video_duration=video_duration,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                chunk_text=chunk_text,
                heatmap=heatmap_list,
                num_clips=per_chunk_clips,
                min_sec=min_sec,
                max_sec=max_sec,
            )
        )

    clamped = _clamp_and_dedupe_clips(all_candidates, video_duration, num_clips, min_sec, max_sec)
    if not clamped:
        return select_clips_fallback(video_duration, heatmap_list, num_clips, min_sec, max_sec)
    return clamped
