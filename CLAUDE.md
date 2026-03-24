# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Kunstwerk generates parallel subtitle videos for opera performances. It downloads audio, separates vocals, transcribes them, aligns the transcription with a libretto, and renders a video with synchronized original-language and translated subtitles. It powers the [Kunstwerk YouTube channel](https://www.youtube.com/@kunstwerk-opera).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (download → separate → transcribe → video)
python kunstwerk.py configs/your_config.yaml

# Skip steps you've already done
python kunstwerk.py configs/your_config.yaml --skip-download
python kunstwerk.py configs/your_config.yaml --skip-transcribe
python kunstwerk.py configs/your_config.yaml --skip-download --skip-transcribe

# Copyright test (blank video with audio, no transcription)
python kunstwerk.py configs/your_config.yaml --copyright-test

# Translate a libretto (uses Claude API)
python translate.py configs/your_config.yaml en

# Generate image prompts from a libretto (sliding-window LLM)
python anime/libretto_promptgen.py libretti/some_libretto_en.txt --opera-title "Title" --style opera_anime

# Audio download & vocal separation only
./separate.sh configs/your_config.yaml
```

External tools required: FFmpeg, yt-dlp, Demucs.

## Architecture

**Pipeline flow** (orchestrated by `kunstwerk.py`):

1. **Download & Separate** (`separate.sh` → `download_spotify.py` or yt-dlp → Demucs) — fetches audio tracks, separates vocals from accompaniment
2. **Transcribe** (`transcribe_elevenlabs.py` then `transcribe.py`) — produces word-level timestamps via ElevenLabs and OpenAI Whisper APIs
3. **Align & Generate Video** (`make_video.py`) — aligns transcription to libretto using DP/Levenshtein, pairs with translation, renders frames

**Key data flow:**
- YAML config (`configs/`) → `OperaConfig` dataclass (`config_parser.py`)
- Audio files (`audio/{prefix}/`) → JSON transcriptions (`transcribed/{prefix}_transcribed/`)
- Libretto text files (`libretti/{prefix}_{lang}.txt`) + transcription → `AlignedWord` list
- Aligned words + translation pairs → video frames → MP4 (`output/`)

**Core modules:**
- `align.py` — `AlignedWord` dataclass, Levenshtein-based word similarity, DP text alignment, deserialization of transcription JSON
- `make_video.py` — the main script that wires alignment, interpolation, monotonicity enforcement, pair splitting, and video creation together (run as a script, not imported)
- `video_gen/` — video generation subsystem: `config/video_config.py` (VideoConfig dataclass), `frame/generator.py` (frame creation), `text/formatting.py` (bold/italic styling), `video/creator.py` (MoviePy-based MP4 output)
- `translate.py` — chunk-based libretto translation via Claude API with line-count validation

**Anime/image generation** (`anime/`):
- `libretto_promptgen.py` — sliding-window LLM pass over libretto to generate image prompts; supports Claude and Kimi providers; uses style templates from `anime/styles/`
- `image_batch_runner.py`, `nano_banana.py`, `imagefx.py`, `seedream4.py`, `kimi_k2.py` — various image generation backends

## Config Format

Opera configs live in `configs/*.yaml`. Key fields: `title`, `file_prefix`, `language` (ISO-639-1), `start_idx`/`end_idx` (scene range), `overture_indices`, `characters`, `secondary_color`, `video_width`/`video_height`, `font_size`, `res_divisor`, `playlist_url` or `spotify_url`, `translation_file`.

## API Keys

Requires environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Optional: `FIREWORKS_API_KEY` (for Kimi provider), Spotify credentials for `download_spotify.py`.

## Conventions

- Libretti are stored as plain text in `libretti/` with naming `{prefix}_{lang}.txt`; paragraphs are separated by blank lines. Source and translation files must have the same number of paragraph blocks.
- Audio files follow `audio/{prefix}/{index}.m4a` numbering (zero-padded to 2 digits).
- Transcription JSON follows OpenAI's `TranscriptionVerbose` schema.
- No formal test suite; development/exploration happens in `notebooks/`.
