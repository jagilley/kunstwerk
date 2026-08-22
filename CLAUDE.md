# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Kunstwerk** — the tooling behind the [Kunstwerk YouTube channel](https://www.youtube.com/@kunstwerk-opera): full-length opera recordings rendered as videos with parallel subtitles (original language left, translation right), word-synced to the singing. The git repo is named `gotterdammerung` for historical reasons; the project is Kunstwerk.

The goal is a **headless pipeline**: one small YAML in, one video out, no hand-prepared inputs. Completed operas (libretti + transcripts present): Ring cycle (rheingold, walkure, siegfried, gotterdammerung), tristan, zauberflote, boheme, giovanni, butterfly, carmen. Only the last few have `configs/*.yaml`; older ones were produced from the per-opera notebooks in `notebooks/`, which predate and duplicate `make_video.py`.

## Environment & commands

- Python 3.10 venv at `.venv/` (`source .venv/bin/activate`; `pip install -r requirements.txt`). `demucs` and `yt-dlp` are installed in the venv. `AGENTS.md` is a symlink to this file. This Mac has 8 GB RAM — keep that in mind for anything that holds whole operas in memory.
- **Always run from the repo root** — every script uses relative paths (`configs/`, `libretti/`, `audio/`, `sep/`, `transcribed/`, `output/`).
- API keys live in `.env` (gitignored), loaded via `python-dotenv`: `OPENAI_API_KEY`, `ELEVENLABS_API_KEY` (transcription), plus unused legacy ones. **Claude is not called through an API key**: `translate.py` shells out to the Claude Code CLI (`claude -p`), which is billed to the subscription the CLI is logged into, and strips `ANTHROPIC_API_KEY` from that subprocess's environment. `ANTHROPIC_API_KEY` is deliberately commented out in `.env`; keep it that way. If `translate.py` says "Not logged in", run `claude` once and `/login`.
- System deps: `ffmpeg` and ImageMagick (`convert` — required by moviepy `TextClip`), both via Homebrew; the Claude Code CLI (`claude`) for translation.
- No test suite and no linter are configured.

```bash
# Full pipeline for one opera (every stage is idempotent — re-run after a failure and it resumes)
python kunstwerk.py configs/carmen.yaml                       # libretto → translate → download+separate → detect instrumental → transcribe → video
python kunstwerk.py configs/carmen.yaml --stop-after libretto # fetch + translate only, then eyeball libretti/
python kunstwerk.py configs/carmen.yaml --skip-download --skip-transcribe   # just re-render
python kunstwerk.py configs/carmen.yaml --copyright-test      # download only → audio-only black video to probe YouTube Content ID

# Individual stages
python fetch_libretto.py configs/carmen.yaml        # librettoarchive.com → libretti/<prefix>_<lang>.txt (+ translation if the site has it)
python translate.py configs/carmen.yaml [de] [--force]   # Claude (via `claude -p`) → libretti/<prefix>_<translation_language>.txt
./separate.sh configs/carmen.yaml                   # download_album.py → audio/<prefix>/NN.m4a, then demucs → sep/<prefix>_sep/htdemucs/NN/vocals.m4a
python download_album.py configs/carmen.yaml        # download only (album_url / album_query / playlist_url)
python detect_instrumental.py configs/carmen.yaml   # vocals-vs-mix energy → sep/<prefix>_sep/instrumental.json (replaces hand-typed overture_indices)
python transcribe_elevenlabs.py configs/carmen.yaml # primary transcriber (ElevenLabs scribe_v1)
python transcribe.py configs/carmen.yaml            # OpenAI whisper-1; fills any tracks ElevenLabs skipped
python make_video.py configs/carmen.yaml            # align + render → output/<prefix>-<res_divisor>.mp4
python config_parser.py configs/carmen.yaml         # show what a config resolves to (derived fields included)
```

Use `res_divisor: 4` in a config for fast 960×540 preview renders; `res_divisor: 1` is the final 4K output. Output filename encodes it: `output/<prefix>-<res_divisor>.mp4`. `KUNSTWERK_SHOW_PLOTS=1` re-enables the blocking matplotlib alignment plots in `make_video.py` (off by default so it can run unattended).

## Configs

A config needs only `title`, `file_prefix`, `language` (ISO-639-1 of the sung language) and an audio source. Everything else is optional and, when absent, **derived from what's on disk** by `config_parser.py` (`OperaConfig`, lazy `cached_property`s — derivations happen on first access because the inputs don't exist yet when the orchestrator starts):

| field | default / derivation |
|---|---|
| `album_url` / `album_query` / `playlist_url` | audio source, tried in that order by `download_album.py` (`spotify_url` is dead — Odesli's public API was shut down) |
| `libretto_url` | explicit librettoarchive.com page; else `fetch_libretto.py` fuzzy-matches `title` against the site catalog (~70 operas; some titles are English, e.g. "The Magic Flute" — use `libretto_url` when the lookup fails) |
| `translation_language` | `en` |
| `translation_file` | `<prefix>_<translation_language>.txt` |
| `start_idx` / `end_idx` | `1` / number of `audio/<prefix>/NN.m4a` files + 1 (falls back to counting `transcribed/`); **`end_idx` is exclusive** |
| `overture_indices` | `sep/<prefix>_sep/instrumental.json` written by `detect_instrumental.py`; `[]` with a warning if neither exists |
| `characters` | ALL-CAPS first lines of blocks in **both** libretti (speaker labels and act headings), plus anything listed — the old hand lists missed e.g. `JOSÉ`, `CHŒUR`, `SOLDIERS` |
| `secondary_color`, `video_width`/`video_height`, `font_size`, `res_divisor` | `Silver`, 3840×2160, 96, 1 |

Explicit values always win, so the older configs keep working unchanged.

## Architecture

`kunstwerk.py` is a thin orchestrator that shells out to each stage script (with the venv's bin dir prepended to `PATH`, so it works from cron/launchd). Stages are independently runnable and idempotent: `fetch_libretto.py` keeps existing output files (and no-ops when both exist) unless `--force`, `translate.py` skips an existing translation unless `--force`, `download_album.py` skips tracks already on disk (`--replace` re-fetches stale ones *and* deletes their stems/transcripts so nothing downstream reuses them), transcribers skip tracks whose JSON exists. Cheap, likely-to-fail stages (libretto lookup, translation, album resolution) run before the expensive ones (demucs, transcription, render).

```
configs/<opera>.yaml
  ├─ fetch_libretto.py ──────► libretti/<prefix>_<lang>.txt (+ <prefix>_<translation_language>.txt when the site has it, block parity guaranteed)
  ├─ translate.py ───────────► libretti/<prefix>_<translation_language>.txt (Claude via `claude -p`, line-parity validated, per-block fallback)
  ├─ separate.sh
  │    ├─ download_album.py ──► audio/<prefix>/NN.m4a + audio/<prefix>/tracks.json (YouTube Music album → ordered art-track playlist, via yt-dlp; fails loudly on gaps)
  │    └─ demucs two-stems ───► sep/<prefix>_sep/htdemucs/NN/vocals.m4a
  ├─ detect_instrumental.py ─► sep/<prefix>_sep/instrumental.json
  ├─ transcribe_elevenlabs.py, transcribe.py ──► transcribed/<prefix>_transcribed/NN.json
  └─ make_video.py ─────────► aligned_words_<prefix>.csv, output/<prefix>-<res>.mp4, YouTube chapter list (stdout)
```

**Audio sourcing.** There is no Spotify/Odesli any more. `download_album.py` takes, in priority order, `track_urls` (a YAML list of video URLs/ids in track order — for hand-assembled recordings; survives playlist rot), `album_url` (a `music.youtube.com/browse/MPREb_…`, `OLAK5uy_…` playlist, or ordinary YouTube playlist URL), `album_query` (free text → YouTube Music album search via yt-dlp, no API keys; candidates are scored — penalising highlights/excerpts/suites/"sung in X", preferring complete-opera durations/track counts and title matches — the table and pick are printed; `--dry-run` to audit, `--pick N` to override, `--strict` to fail if any query word is absent from the pick), or legacy `playlist_url`. It writes `audio/<prefix>/tracks.json` (ids, titles, durations, `complete`), skips files already on disk, refuses to mix recordings (a file is stale if `tracks.json` recorded a different video id for that index; `--replace` re-fetches only those) and exits non-zero on any unavailable track or count mismatch — downstream assumes a contiguous, complete set. Caveats: among complete recordings of one opera the scores are close and the tie-break is popularity, so put conductor/singer surnames in `album_query` (orchestra names rarely appear in YT Music metadata) or pin `album_url`; different editions (e.g. Carmen with dialogue vs. Guiraud recitatives) won't match a given libretto. The complete Abbado/LSO Carmen doesn't exist on YouTube Music as an album, hence `track_urls` in `configs/carmen.yaml`. `tracks.json` keeps the track titles; they contain each number's incipit and are the intended source of automatic alignment anchors (not wired up yet). yt-dlp must run from a residential IP (this Mac) — cloud runners get bot-checked. `separate.sh` runs demucs only on tracks without a `vocals.m4a` yet, so restarts are cheap; demucs on this CPU is ~1× realtime (3–4 h per opera) — a GPU (Modal) step is the obvious future win.

**Transcript JSON shape** is always OpenAI's `TranscriptionVerbose` (`duration`, `language`, `text`, `words[{word,start,end}]`). `classes.py` defines dataclass twins so the ElevenLabs/AssemblyAI transcribers emit the same shape, and `align.deserialize_transcription_from_file` reads them all back as the OpenAI pydantic model. Per-track times are relative; `convert_file_times_to_absolute_times` shifts them using each track's `duration`.

**Track indexing**: tracks are 1-based, zero-padded to 2 digits (`01.m4a`). `start_idx`/`end_idx` are used as `range(start_idx, end_idx)` — **`end_idx` is exclusive**. `overture_indices` are 1-based track numbers whose transcripts are blanked (purely instrumental — ElevenLabs hallucinates fluent text on orchestral tracks, up to hundreds of words, so word density cannot be used to detect them). The transcribers ignore `start_idx` and always loop from 1.

**Instrumental detection** (`detect_instrumental.py`): per 0.5 s frame, `vocals_rms / mix_rms` from the demucs stem vs. the original; a frame "has vocals" above 0.2 (~-14 dB); `vocal_frac` = share of non-silent frames with vocals; a track is instrumental iff `vocal_frac < 0.10` and it has < 30 s of detected vocals (the second clause guards CD splits that glue a prelude to a sung passage). Calibrated on Carmen: prelude/entr'actes 0.00–0.015, quietest sung track (boys' chorus) 0.53, spoken dialogue 1.0 — a ~5–7× margin either way, stable across frame length / ratio / silence-floor sweeps. It only scores tracks that have a `vocals.{m4a,wav}` stem, so the cache is complete only after demucs has run over every track (the pipeline guarantees that order; `config_parser.py` warns when the cache doesn't cover the configured track range). All metrics, including a whole-track `rms_ratio_db` cross-check, are kept in `sep/<prefix>_sep/instrumental.json`.

**`make_video.py`** is the heart and is a top-to-bottom script (no `main()`), in this order:
1. Load transcripts, blank overtures, concatenate into one absolute-time word list.
2. `align.align_transcription_with_libretto` — Needleman-Wunsch-style DP over transcript words vs. whitespace-split libretto words with Levenshtein similarity, producing `AlignedWord(word, start, end)` with `None` times for unmatched libretto words. It is vectorised (rapidfuzz similarity in row chunks, numpy prefix-max per row, int8 backtrack): a full opera (~30k × 10k words) aligns in ~15 s and <0.5 GB; the old pure-Python version in this file needed ~10 GB and swapped. Optional `markers = [(timestamp, phrase), ...]` ground-truth anchors (parsed by `parse_timestamp_and_phrase`) give a scoring bonus to pin drifting sections; empty by default and the manual knob used when alignment went off the rails. Typical raw coverage is 30–60% of libretto words timed; the transcript is ~3× longer than the libretto because of repeats the libretto elides as "etc."/"ecc.".
3. Writes `aligned_words_<prefix>.csv` and immediately reads it back — the hook for hand-editing timings — then `interpolate_word_timings` spreads times across untimed runs ≤ 20 s.
4. Pairs source and translation libretti block-by-block (blank-line separated) — **block counts must match or it raises** — and `split_long_pairs` splits blocks > 15 lines at parenthesis-safe points so they fit on screen.
5. `video_gen/`: `create_frames` pre-renders one RGB frame per line pair (moviepy `TextClip`, Baskerville; character names bold, parenthetical stage directions italic, font auto-shrinks to fit), maps each `1/fps` timestamp to a line index (fps=4, 8 s `text_timeout` after the last sung word, title card for the first 10 s), then `enforce_monotonicity` (never show an earlier line after a later one) and `interpolate_frames` (fill short gaps between identical lines). `create_parallel_text_video` writes the mp4 with concatenated original audio. Known hot spots: every pre-rendered frame is kept in RAM (fine at `res_divisor: 4`, ~15 GB at 4K — 4K renders swap on this machine) and `get_active_line_idx` is O(words) per timestamp.
6. `generate_audio_timestamps` prints per-track chapter markers (first sung line of each track) to paste into the YouTube description.

**Libretto text format** (`libretti/<prefix>_<lang>.txt`): plain text, blocks separated by blank lines; speaker names in CAPS on their own line at the top of a block; stage directions in parentheses. Bold formatting matches `name + "\n"` as a block prefix (`video_gen/text/formatting.py`). The translation file must mirror the source block-for-block; `fetch_libretto.py` guarantees that when it fetches both (pairing the site's two single-language pages of the same edition row by row; mismatched blank-line placement is repaired by merging into a neighbour and logged as a WARNING; `--strict-parity` disables that), and `translate.py` enforces newline parity per chunk with retries and a per-block fallback. librettoarchive.com's bilingual pages are paywalled — don't point `libretto_url` at `_libretto_<L1>_<L2>` URLs.

## Gotchas and legacy

- `anime/`, `stories/`, `generated_images/`, `seedream_playground_outputs/`, `pyproject.toml` and `*.egg-info/` are gitignored on purpose: they belong to a separate private project that is being moved out of this public repo. Don't commit, document, or depend on them here.
- Large data is gitignored (`audio/`, `sep/`, `transcribed/`, `output/`, `images/`, `*.csv`, `*.mp4`, `*.png`). Locally, `transcribed/` and `aligned_words_*.csv` exist for most operas but `audio/` and `sep/` are mostly empty — re-rendering any video requires re-downloading audio first. The old hand-pasted libretti differ slightly from what `fetch_libretto.py` produces (block boundaries, a few typo fixes); leave them alone — the existing CSVs/markers are tied to them.
- `configs/giovanni.yaml` and `configs/butterfly.yaml` point at the same, now-dead YouTube playlist; `configs/tristan.yaml` has a placeholder URL and no translation file. They need an `album_query`/`album_url` before they can be re-run.
- yt-dlp has deprecated Python 3.10; the venv will need bumping before long.
- Dead or superseded: `align.py`'s `__main__` and `align_texts` (hardcoded rheingold, 3-digit indices), `parse_libretto.py` (HTML-table libretti from the old murashev.com layout), `transcribe_ass.py` (AssemblyAI, hardcoded tristan), `elevenlabs_example.py`, `convert_transcription.py`. `notebooks/*.ipynb` are the original per-opera workflows and duplicate `make_video.py` cell-for-cell.
- `.claude/skills/` was copied in from another project; several skills (`writeup`, `subagent-instructions`, `document-history*`) reference conventions and files (Modal jobs, `experiments/CLAUDE.md`, `STRUCTURE.md`) that don't exist here.
