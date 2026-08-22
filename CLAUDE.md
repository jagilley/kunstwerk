# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Kunstwerk** — the tooling behind the [Kunstwerk YouTube channel](https://www.youtube.com/@kunstwerk-opera): full-length opera recordings rendered as videos with parallel subtitles (original language left, translation right), word-synced to the singing. The git repo is named `gotterdammerung` for historical reasons; the project is Kunstwerk.

Completed operas (libretti + transcripts present): Ring cycle (rheingold, walkure, siegfried, gotterdammerung), tristan, zauberflote, boheme, giovanni, butterfly, carmen. Only the last few have `configs/*.yaml`; older ones were produced from the per-opera notebooks in `notebooks/`, which predate and duplicate `make_video.py`.

## Environment & commands

- Python 3.10 venv at `.venv/` (`source .venv/bin/activate`; `pip install -r requirements.txt`). `demucs` and `yt-dlp` are installed in the venv. `AGENTS.md` is a symlink to this file.
- **Always run from the repo root** — every script uses relative paths (`configs/`, `libretti/`, `audio/`, `sep/`, `transcribed/`, `output/`).
- API keys live in `.env` (gitignored), loaded via `python-dotenv`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`, `ASSEMBLYAI_API_KEY`, `DEEPINFRA_API_KEY`. Spotify download additionally needs `SPOTIFY_CLIENT_ID`/`SPOTIFY_CLIENT_SECRET`.
- System deps: `ffmpeg` and ImageMagick (`convert` — required by moviepy `TextClip`), both via Homebrew.
- No test suite and no linter are configured.

```bash
# Full pipeline for one opera
python kunstwerk.py configs/carmen.yaml                      # download+separate → transcribe → video
python kunstwerk.py configs/carmen.yaml --skip-download      # audio/ and sep/ already exist
python kunstwerk.py configs/carmen.yaml --skip-download --skip-transcribe   # just re-render video
python kunstwerk.py configs/carmen.yaml --copyright-test     # audio-only black video to probe YouTube Content ID

# Individual stages
./separate.sh configs/carmen.yaml                 # yt-dlp/Spotify download → audio/<prefix>/NN.m4a, then demucs → sep/<prefix>_sep/htdemucs/NN/vocals.m4a
python transcribe_elevenlabs.py configs/carmen.yaml   # primary transcriber (ElevenLabs scribe_v1)
python transcribe.py configs/carmen.yaml              # OpenAI whisper-1; fills any tracks ElevenLabs skipped
python translate.py configs/carmen.yaml en [--force]  # Claude translation → libretti/<prefix>_en.txt
python make_video.py configs/carmen.yaml              # align + render → output/<prefix>-<res_divisor>.mp4
```

Use `res_divisor: 4` in a config for fast 960×540 preview renders; `res_divisor: 1` is the final 4K output. Output filename encodes it: `output/<prefix>-<res_divisor>.mp4`.

## Architecture

`kunstwerk.py` is a thin orchestrator that shells out to each stage script; each stage is independently runnable and idempotent-ish (transcribers skip tracks whose JSON already exists; `translate.py` refuses to overwrite without `--force`). Everything is keyed off `configs/<opera>.yaml`, parsed into `OperaConfig` by `config_parser.py` (`parse_yaml.py` is the bash-friendly subset used by `separate.sh`). Config fields: `title`, `file_prefix`, `language` (ISO-639-1), `start_idx`/`end_idx`, `overture_indices`, `characters`, `secondary_color` (X11 color name), `video_width`/`video_height`, `font_size`, `res_divisor`, `translation_file`, and `playlist_url` or `spotify_url`.

```
configs/<opera>.yaml
  ├─ separate.sh ─────────► audio/<prefix>/NN.m4a            (playlist_url via yt-dlp, or spotify_url via download_spotify.py: Spotify → Odesli/song.link → YouTube art-track)
  │                          └─ demucs two-stems ──► sep/<prefix>_sep/htdemucs/NN/vocals.m4a
  ├─ transcribe_elevenlabs.py, transcribe.py ──► transcribed/<prefix>_transcribed/NN.json
  ├─ translate.py ──────────► libretti/<prefix>_<lang>.txt
  └─ make_video.py ─────────► aligned_words_<prefix>.csv, output/<prefix>-<res>.mp4, YouTube chapter list (stdout)
```

**Transcript JSON shape** is always OpenAI's `TranscriptionVerbose` (`duration`, `language`, `text`, `words[{word,start,end}]`). `classes.py` defines dataclass twins so the ElevenLabs/AssemblyAI transcribers emit the same shape, and `align.deserialize_transcription_from_file` reads them all back as the OpenAI pydantic model. Per-track times are relative; `convert_file_times_to_absolute_times` shifts them using each track's `duration`.

**Track indexing**: tracks are 1-based, zero-padded to 2 digits (`01.m4a`). `start_idx`/`end_idx` are used as `range(start_idx, end_idx)` — **`end_idx` is exclusive**, so it's (number of tracks + 1). `overture_indices` are 1-based track numbers whose transcripts are blanked (purely instrumental). The transcribers ignore `start_idx` and always loop from 1.

**`make_video.py`** is the heart and is a top-to-bottom script (no `main()`), in this order:
1. Load transcripts, blank overtures, concatenate into one absolute-time word list.
2. `align_transcription_with_libretto` — Needleman-Wunsch-style DP over transcript words vs. whitespace-split libretto words with Levenshtein similarity (`word_similarity` in `align.py`), producing `AlignedWord(word, start, end)` with `None` times for unmatched libretto words. Optional `markers = [(timestamp, phrase), ...]` ground-truth anchors (parsed by `parse_timestamp_and_phrase`) give a scoring bonus to pin drifting sections; it's empty by default and was the manual knob used when alignment went off the rails.
3. Writes `aligned_words_<prefix>.csv` and immediately reads it back — this is the hook for hand-editing timings — then `interpolate_word_timings` spreads times across untimed runs ≤ 20 s.
4. Pairs source and translation libretti block-by-block (blank-line separated) — **block counts must match or it raises** — and `split_long_pairs` splits blocks > 15 lines at parenthesis-safe points so they fit on screen.
5. `video_gen/`: `create_frames` pre-renders one RGB frame per line pair (moviepy `TextClip`, Baskerville; character names bold, parenthetical stage directions italic, font auto-shrinks to fit), maps each `1/fps` timestamp to a line index (fps=4, 8 s `text_timeout` after the last sung word, title card for the first 10 s), then `enforce_monotonicity` (never show an earlier line after a later one) and `interpolate_frames` (fill short gaps between identical lines). `create_parallel_text_video` writes the mp4 with concatenated original audio.
6. `generate_audio_timestamps` prints per-track chapter markers (first sung line of each track) to paste into the YouTube description.

`show_plots = True` at the top of `make_video.py` pops up blocking matplotlib windows of alignment coverage; set it to `False` for unattended runs.

**Libretto text format** (`libretti/<prefix>_<lang>.txt`): plain text, blocks separated by blank lines; character names in CAPS on their own line; stage directions in parentheses. `characters:` in the YAML must list names exactly as they appear (the code also tries upper/lower variants) — they drive bold formatting (matched as `name + "\n"` prefix of a block) and get stripped from chapter titles. The translation file must mirror the source block-for-block; `translate.py` enforces newline parity per chunk and retries up to 5 times to get it.

## Gotchas and legacy

- `anime/`, `stories/`, `generated_images/`, `seedream_playground_outputs/`, `pyproject.toml` and `*.egg-info/` are gitignored on purpose: they belong to a separate private project that is being moved out of this public repo. Don't commit, document, or depend on them here.
- Large data is gitignored (`audio/`, `sep/`, `transcribed/`, `output/`, `images/`, `*.csv`, `*.mp4`, `*.png`). Locally, `transcribed/` and `aligned_words_*.csv` exist for most operas but `audio/` and `sep/` are empty — re-rendering any video requires re-downloading audio first.
- `kunstwerk.py` references `args.minimal` and `minimal_copyright_test.py`, neither of which exists; `--copyright-test` will crash at that branch until it's cleaned up.
- Dead or superseded: `align.py`'s `__main__` (hardcoded rheingold, 3-digit indices), `parse_libretto.py` (HTML-table libretti), `transcribe_ass.py` (AssemblyAI, hardcoded tristan), `elevenlabs_example.py`. `notebooks/*.ipynb` are the original per-opera workflows and duplicate `make_video.py` cell-for-cell.
- `.claude/skills/` was copied in from another project; several skills (`writeup`, `subagent-instructions`, `document-history*`) reference conventions and files (Modal jobs, `experiments/CLAUDE.md`, `STRUCTURE.md`) that don't exist here.
