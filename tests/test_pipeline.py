"""Unit tests for VTT parsing, clip clamping, fallback selection, and ASS subtitles."""

from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yt_shorts.analyzer import ClipCandidate, _chunk_transcript, _clamp_and_dedupe_clips, select_clips_fallback
from yt_shorts.downloader import (
    AuthStatus,
    DownloadErrorCategory,
    TranscriptSegment,
    YouTubeDownloadError,
    download_youtube_video,
    inspect_cookie_file,
    parse_subtitles_file,
    validate_video_file,
    validate_youtube_url,
    verify_youtube_auth,
)
from yt_shorts.editor import resolve_ffmpeg_exe, write_clip_ass_subtitles
from yt_shorts.pipeline import run_pipeline


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

    def _make_sample_mp4(self, target: Path, duration_sec: int = 16) -> Path:
        ffmpeg = resolve_ffmpeg_exe()
        res = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=blue:s=640x360:d={duration_sec}",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r=44100:cl=stereo:d={duration_sec}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        return target

    def test_validate_youtube_url(self) -> None:
        self.assertEqual(
            validate_youtube_url("https://www.youtube.com/watch?v=jNQXAC9IVRw"),
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        )
        self.assertEqual(
            validate_youtube_url("https://youtu.be/jNQXAC9IVRw"),
            "https://youtu.be/jNQXAC9IVRw",
        )
        for bad in ["", "ftp://youtube.com/watch?v=1", "https://example.com/video.mp4"]:
            with self.assertRaises(YouTubeDownloadError) as ctx:
                validate_youtube_url(bad)
            self.assertEqual(ctx.exception.category, DownloadErrorCategory.INVALID_URL)

    def test_inspect_cookie_file_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            self.assertEqual(inspect_cookie_file(tmp_dir / "missing.txt").status, AuthStatus.NOT_CONFIGURED)

            empty = tmp_dir / "empty.txt"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(inspect_cookie_file(empty).status, AuthStatus.FAILED)

            bad = tmp_dir / "bad.txt"
            bad.write_text("not a netscape file", encoding="utf-8")
            self.assertEqual(inspect_cookie_file(bad).status, AuthStatus.FAILED)

            no_auth = tmp_dir / "no_auth.txt"
            no_auth.write_text(
                "# Netscape HTTP Cookie File\n"
                ".youtube.com\tTRUE\t/\tTRUE\t0\tPREF\thl=en\n",
                encoding="utf-8",
            )
            self.assertEqual(inspect_cookie_file(no_auth).status, AuthStatus.CONFIGURED)

            valid = tmp_dir / "valid.txt"
            valid.write_text(
                "# Netscape HTTP Cookie File\n"
                ".youtube.com\tTRUE\t/\tTRUE\t2500000000\tLOGIN_INFO\tabc\n"
                ".youtube.com\tTRUE\t/\tTRUE\t2500000000\tSID\txyz\n"
                ".youtube.com\tTRUE\t/\tTRUE\t2500000000\tSAPISID\tqrs\n",
                encoding="utf-8",
            )
            res = inspect_cookie_file(valid)
            self.assertEqual(res.status, AuthStatus.PARSED)
            self.assertTrue(res.has_auth_tokens)

    def test_verify_youtube_auth_failure(self) -> None:
        import yt_dlp

        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "cookies.txt"
            valid.write_text(
                "# Netscape HTTP Cookie File\n"
                ".youtube.com\tTRUE\t/\tTRUE\t2500000000\tLOGIN_INFO\tabc\n"
                ".youtube.com\tTRUE\t/\tTRUE\t2500000000\tSID\txyz\n"
                ".youtube.com\tTRUE\t/\tTRUE\t2500000000\tSAPISID\tqrs\n",
                encoding="utf-8",
            )
            with patch("yt_dlp.YoutubeDL") as mock_ydl:
                mock_ydl.return_value.__enter__.return_value.extract_info.side_effect = (
                    yt_dlp.utils.DownloadError("Sign in to confirm your age")
                )
                res = verify_youtube_auth(valid)
                self.assertEqual(res.status, AuthStatus.FAILED)

    def test_po_token_and_error_classifications(self) -> None:
        import yt_dlp

        captured_opts: list[dict] = []

        class BotMockYDL:
            def __init__(self, opts):
                captured_opts.append(opts)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                raise yt_dlp.utils.DownloadError("Sign in to confirm you’re not a bot")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("yt_dlp.YoutubeDL", BotMockYDL):
                with self.assertRaises(YouTubeDownloadError) as ctx:
                    download_youtube_video(
                        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
                        Path(tmp),
                        po_token="mweb.gvs+TESTTOKEN",
                        pot_server_url="http://127.0.0.1:4416",
                    )
                self.assertEqual(ctx.exception.category, DownloadErrorCategory.BOT_VERIFICATION)
                self.assertEqual(captured_opts[0]["extractor_args"]["youtube"]["po_token"], ["mweb.gvs+TESTTOKEN"])
                self.assertEqual(
                    captured_opts[0]["extractor_args"]["youtubepot-bgutilhttp"]["base_url"],
                    ["http://127.0.0.1:4416"],
                )

        class UnavailableMockYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                raise yt_dlp.utils.DownloadError("Video unavailable. This video has been removed")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("yt_dlp.YoutubeDL", UnavailableMockYDL):
                with self.assertRaises(YouTubeDownloadError) as ctx:
                    download_youtube_video("https://www.youtube.com/watch?v=jNQXAC9IVRw", Path(tmp))
                self.assertEqual(ctx.exception.category, DownloadErrorCategory.VIDEO_UNAVAILABLE)

    def test_network_timeout_exponential_backoff(self) -> None:
        import yt_dlp

        attempts = {"n": 0}

        class TimeoutMockYDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                attempts["n"] += 1
                raise yt_dlp.utils.DownloadError("Read timed out")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("yt_dlp.YoutubeDL", TimeoutMockYDL), patch("yt_shorts.downloader.time.sleep") as mock_sleep:
                with self.assertRaises(YouTubeDownloadError) as ctx:
                    download_youtube_video("https://www.youtube.com/watch?v=jNQXAC9IVRw", Path(tmp))
                self.assertEqual(ctx.exception.category, DownloadErrorCategory.NETWORK_TIMEOUT)
                self.assertEqual(attempts["n"], 3)
                self.assertEqual([c.args[0] for c in mock_sleep.call_args_list], [1, 2])

    def test_concurrent_downloads_and_validation(self) -> None:
        test_case = self

        class SuccessMockYDL:
            def __init__(self, opts):
                self.outtmpl = opts["outtmpl"]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                vid = url.split("v=")[-1]
                target = Path(self.outtmpl.replace("%(id)s", vid).replace("%(ext)s", "mp4"))
                test_case._make_sample_mp4(target, duration_sec=6)
                return {"id": vid, "title": f"Title {vid}", "duration": 6.0}

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch("yt_dlp.YoutubeDL", SuccessMockYDL):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    f1 = pool.submit(download_youtube_video, "https://www.youtube.com/watch?v=vidA", out_dir, None, False)
                    f2 = pool.submit(download_youtube_video, "https://www.youtube.com/watch?v=vidB", out_dir, None, False)
                    r1, r2 = f1.result(), f2.result()
            self.assertNotEqual(r1.video_path, r2.video_path)
            self.assertTrue(r1.video_path.exists())
            self.assertTrue(r2.video_path.exists())
            self.assertEqual(list(out_dir.glob("yt_dl_*")), [])

    def test_direct_upload_workflow_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "_cache"
            cache.mkdir(parents=True, exist_ok=True)
            uploaded = self._make_sample_mp4(cache / "sample_upload.mp4", duration_sec=16)

            result = run_pipeline(
                local_video_path=uploaded,
                output_root=root,
                num_clips=1,
                min_sec=15.0,
                max_sec=20.0,
                layout="blur",
                burn_captions=False,
                cleanup_cache=True,
            )
            self.assertEqual(len(result.clips), 1)
            clip_path = result.clips[0].file_path
            self.assertTrue(clip_path.exists())
            self.assertGreater(clip_path.stat().st_size, 0)
            self.assertGreater(validate_video_file(clip_path, resolve_ffmpeg_exe()), 10.0)
            self.assertFalse(uploaded.exists())


if __name__ == "__main__":
    unittest.main()
