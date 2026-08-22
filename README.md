# Kunstwerk

Tooling for generating parallel-subtitle videos of full-length operas — original language on the left, translation on the right, word-synced to the singing. This is the repo behind the [YouTube channel of the same name](https://www.youtube.com/@kunstwerk-opera); example output: [Tristan und Isolde with parallel subtitles](https://www.youtube.com/watch?v=2R6lTcdJoCk).

## How it works

One YAML config in, one video out:

1. **Libretto** — `fetch_libretto.py` pulls the libretto (and its translation, when available) from librettoarchive.com into a simple blank-line-separated text format; `translate.py` fills in a missing translation with Claude, block-for-block.
2. **Audio** — `download_album.py` resolves a recording to its YouTube Music album (auto-generated art-tracks, in order) and downloads it with yt-dlp; `demucs` separates the vocals.
3. **Transcription** — ElevenLabs Scribe (with OpenAI Whisper as fallback) transcribes the vocal stems with word timestamps; `detect_instrumental.py` flags purely orchestral tracks so their hallucinated transcripts get ignored.
4. **Alignment + render** — `make_video.py` aligns the transcript to the libretto (Needleman-Wunsch over Levenshtein similarity), interpolates timings, pairs source/translation blocks and renders the video with moviepy; it also prints YouTube chapter markers.

## Prerequisites

- Python 3.10+, `ffmpeg`, ImageMagick (`convert`)
- The [Claude Code CLI](https://claude.com/claude-code) (`claude`), logged in — translation runs through it
- `.env` with `ELEVENLABS_API_KEY` and `OPENAI_API_KEY`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

A minimal config (`configs/<opera>.yaml`):

```yaml
title: CARMEN
file_prefix: carmen
language: fr                      # language sung (ISO-639-1)
album_query: Bizet Carmen Abbado London Symphony Orchestra   # or album_url: https://music.youtube.com/browse/MPREb_...
res_divisor: 4                    # 4 = quick 960x540 preview, 1 = final 4K
```

Everything else — number of tracks, which tracks are instrumental, character names for bold formatting, the translation file — is derived automatically; see `CLAUDE.md` for the full field list and how each is derived. Any field can still be set explicitly.

```bash
python kunstwerk.py configs/carmen.yaml                       # the whole pipeline
python kunstwerk.py configs/carmen.yaml --stop-after libretto # fetch + translate, then stop to eyeball the text
python kunstwerk.py configs/carmen.yaml --skip-download --skip-transcribe   # re-render only
python kunstwerk.py configs/carmen.yaml --copyright-test      # download the audio and make a black-screen video to test YouTube Content ID before investing in the full render
```

Every stage is idempotent — it skips work whose output already exists — so re-running after a failure picks up where it left off. Each stage is also its own script (`fetch_libretto.py`, `translate.py`, `separate.sh`, `download_album.py`, `detect_instrumental.py`, `transcribe_elevenlabs.py`, `transcribe.py`, `make_video.py`); run them individually when you want to intervene, e.g. hand-edit `aligned_words_<prefix>.csv` between alignment and render.

Outputs land in `output/<prefix>-<res_divisor>.mp4`.

## Project structure

- `kunstwerk.py` — orchestrator
- `config_parser.py` — config loading and derived fields
- `fetch_libretto.py`, `translate.py` — libretto acquisition
- `download_album.py`, `separate.sh`, `detect_instrumental.py` — audio acquisition and separation
- `transcribe_elevenlabs.py`, `transcribe.py` — transcription
- `align.py` — transcript ↔ libretto alignment
- `make_video.py`, `video_gen/` — frame generation and rendering
- `libretti/`, `configs/` — per-opera inputs
