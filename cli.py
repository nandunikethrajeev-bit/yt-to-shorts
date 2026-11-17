"""CLI entrypoint for converting long YouTube videos into 15-30s Shorts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from yt_shorts.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Convert a long YouTube video into 15-30s vertical (9:16) Shorts."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--clips", type=int, default=3, help="Number of Shorts to generate (default: 3)")
    parser.add_argument("--min-sec", type=float, default=15.0, help="Minimum clip length in seconds (default: 15)")
    parser.add_argument("--max-sec", type=float, default=30.0, help="Maximum clip length in seconds (default: 30)")
    parser.add_argument(
        "--layout",
        choices=["blur", "crop"],
        default="blur",
        help="Vertical 9:16 layout: 'blur' (full frame + blurred bg) or 'crop' (center crop)",
    )
    parser.add_argument("--no-captions", action="store_true", help="Disable burned-in captions")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory (default: ./output)")
    parser.add_argument("--model", default=None, help="Gemini model (default: gemini-3.5-flash-lite)")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser name for YouTube cookies (e.g., brave, chrome, edge, firefox)",
    )
    parser.add_argument("--cookies", type=Path, default=None, help="Path to exported cookies.txt file")
    parser.add_argument(
        "--quality",
        type=int,
        choices=[720, 1080],
        default=1080,
        help="Max download height in pixels (720 or 1080, default: 1080)",
    )

    args = parser.parse_args()
    if args.min_sec <= 0 or args.max_sec < args.min_sec:
        parser.error("--max-sec must be >= --min-sec > 0")

    result = run_pipeline(
        url=args.url,
        output_root=args.out,
        num_clips=args.clips,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        layout=args.layout,
        burn_captions=not args.no_captions,
        model=args.model,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies,
        max_height=args.quality,
        on_progress=lambda msg: print(f"[yt-shorts] {msg}", flush=True),
    )

    print(f"\nGenerated {len(result.clips)} Shorts in {result.output_dir}:")
    for clip in result.clips:
        print(
            f"  {clip.index}. {clip.file_path.name} "
            f"[{clip.start_time:.1f}s - {clip.end_time:.1f}s, {clip.duration:.1f}s] "
            f"(score {clip.viral_score}/100) - {clip.hook_title}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
