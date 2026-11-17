"""Unit tests for VTT parsing, clip clamping, fallback selection, and ASS subtitles."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yt_shorts.analyzer import ClipCandidate, _chunk_transcript, _clamp_and_dedupe_clips, select_clips_fallback
from yt_shorts.downloader import TranscriptSegment, parse_subtitles_file
from yt_shorts.editor import write_clip_ass_subtitles


class TestYtShorts(unittest.TestCase):
    def test_parse_vtt_deduplicates_rolling_captions(self) -> None:
        vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "<00:00:01.200><c>Hello</c> world\n\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "Hello world\n\n"
            "00:00:05.000 --> 00:00:08.000\n"
            "Hello world this is a test\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.en.vtt"
            p.write_text(vtt, encoding="utf-8")
            segs = parse_subtitles_file(p)

        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].text, "Hello world")
        self.assertEqual(segs[0].end, 5.0)
        self.assertEqual(segs[1].text, "this is a test")

    def test_clamp_and_dedupe_clips(self) -> None:
        raw = [
            ClipCandidate(start_time=10.0, end_time=18.0, hook_title="Short", reason="too short", viral_score=90),
            ClipCandidate(start_time=12.0, end_time=35.0, hook_title="Overlaps", reason="overlap", viral_score=85),
            ClipCandidate(start_time=60.0, end_time=110.0, hook_title="Too long", reason="long", viral_score=80),
        ]
        clamped = _clamp_and_dedupe_clips(raw, video_duration=120.0, num_clips=3, min_sec=15.0, max_sec=30.0)
        self.assertEqual(len(clamped), 2)
        for clip in clamped:
            dur = clip.end_time - clip.start_time
            self.assertGreaterEqual(dur, 15.0)
            self.assertLessEqual(dur, 30.0)

    def test_fallback_uses_heatmap_when_available(self) -> None:
        heatmap = [
            {"start_time": 40.0, "end_time": 45.0, "value": 0.95},
            {"start_time": 120.0, "end_time": 125.0, "value": 0.80},
        ]
        clips = select_clips_fallback(video_duration=200.0, heatmap=heatmap, num_clips=2, min_sec=15.0, max_sec=30.0)
        self.assertEqual(len(clips), 2)
        self.assertTrue(all(15.0 <= (c.end_time - c.start_time) <= 30.0 for c in clips))

    def test_write_clip_ass_subtitles(self) -> None:
        transcript = [
            TranscriptSegment(start=5.0, end=8.0, text="Ignored before clip"),
            TranscriptSegment(start=10.0, end=14.0, text="First line in clip"),
            TranscriptSegment(start=14.5, end=19.0, text="Second line in clip"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ass = Path(tmp) / "clip.ass"
            ok = write_clip_ass_subtitles(transcript, clip_start=12.0, clip_end=30.0, ass_path=ass)
            self.assertTrue(ok)
            content = ass.read_text(encoding="utf-8")
            self.assertIn("First line in clip", content)
            self.assertIn("Second line in clip", content)
            self.assertNotIn("Ignored before clip", content)

    def test_chunk_transcript_preserves_all_segments(self) -> None:
        transcript = [
            TranscriptSegment(start=i * 10.0, end=i * 10.0 + 5.0, text=f"Segment number {i} in long stream")
            for i in range(60)
        ]
        chunks = _chunk_transcript(transcript, max_chars=400)
        self.assertGreater(len(chunks), 1)
        combined = "\n".join(text for _, _, text in chunks)
        for i in range(60):
            self.assertIn(f"Segment number {i} in long stream", combined)


if __name__ == "__main__":
    unittest.main()
