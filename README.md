# YouTube to Shorts (15–30s)

A 100% free, local tool that converts long YouTube videos into 15–30 second vertical (`9:16`, `1080x1920`) Shorts clips using `yt-dlp`, `ffmpeg` (via `imageio-ffmpeg`), and the free tier of the Gemini API (`google-genai`).

## Setup (Windows)

```powershell
cd "C:\Users\Niketh Rajeev\.gemini\antigravity\scratch\yt-to-shorts"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your free Gemini API key from [Google AI Studio](https://aistudio.google.com/):

```powershell
Copy-Item .env.example .env
```

> **Note:** If you leave `GEMINI_API_KEY` blank, the tool automatically falls back to YouTube's "Most Replayed" heatmap (or evenly spaced highlights) at zero cost.

## Usage

### Web UI (Streamlit)
```powershell
streamlit run app.py
```

### Command Line (CLI)
```powershell
python cli.py "https://www.youtube.com/watch?v=VIDEO_ID" --clips 3 --min-sec 15 --max-sec 30 --layout blur
```

Options:
- `--clips`: Number of Shorts to generate (default `3`)
- `--min-sec` / `--max-sec`: Clip duration range in seconds (default `15`–`30`)
- `--layout`: `blur` (full 16:9 frame centered on blurred 9:16 background) or `crop` (center-cropped 9:16)
- `--no-captions`: Skip burning in subtitles
- `--model`: Gemini model (`gemini-3.5-flash-lite` default, or `gemini-4-hydrogen`)
- `--out`: Output folder (default `./output`)
