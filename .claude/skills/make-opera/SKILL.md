---
description: Produce a new Kunstwerk opera video from just the opera's name — settle the recording and libretto with Jasper, run the pipeline, QA the result, hand over the files. Invoke for "/make-opera <opera>", "make <opera>", "let's do Tosca next", etc.
---

# /make-opera <opera> [recording hints, a YouTube Music album URL, or "just pick"]

You're producing a finished parallel-subtitle video (original language left, translation right, word-synced) for an opera, end to end. Read `CLAUDE.md` first — it documents every stage, config field, derived value and gotcha; this skill is about the judgment calls around that machinery, and what to do while it runs. Everything runs from the repo root with `.venv/bin/python`. Budget: ~1 hour wall clock, well under $1 of API + Modal (profile `chromatic`).

**Deliverables:** `configs/<prefix>.yaml`; `libretti/<prefix>_<lang>.txt` + translation; `output/<prefix>_copyright_test.mp4` (early); `output/<prefix>-1.mp4` (4K); `output/<prefix>-alignment-report.json`; `output/<prefix>-chapters.txt`; a short handover; a merged PR with the config + libretti (audio/stems/transcripts/video are gitignored data).

## 1. Pin the opera down (minutes, do this before spending anything)

Composer, canonical title, sung language (ISO-639-1), a short ascii `file_prefix`. Then the two availability checks, cheapest first:

- **Libretto**: `python fetch_libretto.py --list-catalog` — librettoarchive.com is the only wired source (~70 operas). If the opera isn't there, stop and tell Jasper; don't download anything. Some entries are English-titled ("The Magic Flute") — `libretto_url` handles that. Note which translation languages the site offers; if there's no English, `translate.py` (Claude, via the subscription) fills in, which is fine.
- **Edition**: the libretto and the recording must be the same version of the work — Carmen with spoken dialogue vs. Guiraud recitatives, Don Carlo(s) four/five acts, Boris, Hoffmann, Ariadne… When a work has versions, decide which one you're making and check the recording's track titles against the libretto's numbers before committing.

## 2. Choose the recording with Jasper

He wants to supervise this (he cares about which recording it is), so unless he named one or said "just pick":

1. Draft `configs/<prefix>.yaml` with `album_query: <composer> <opera> [conductor/singers]` and run `python download_album.py configs/<prefix>.yaml --dry-run`. It searches YouTube Music albums and prints a scored table; read it the way a record-shop regular would. Complete recordings of one opera score within a point or two of each other and the tie-break is popularity; orchestra names don't appear in YT Music metadata (singer/conductor surnames do); anything "sung in English/German" or "Highlights" is already penalised.
2. Shortlist two or three: cast, conductor, year, track count (per-number tracks beat per-act megatracks for chapters *and* alignment), total duration, channel. Say which you'd choose and why. Ask Jasper to confirm (`AskUserQuestion` is fine — it's his call). If a famous recording he wants isn't on YouTube Music as an album (the Abbado Carmen wasn't), a hand-made `track_urls` list or a YouTube playlist URL also works — see CLAUDE.md.
3. Pin the choice as `album_url` (not the query) in the config, so re-runs can't drift to a different recording.

## 3. Run the pipeline — you hold the waits

You are a top-level session: long jobs go in **background Bash** (`run_in_background`) and the harness wakes you when they finish. Don't poll logs, and don't delegate waiting to subagents — subagents can't idle for free and die on long waits (we've watched it happen). Subagents are good for bounded side-quests: researching a recording's cast/edition, checking a libretto page, reviewing frames.

The two halves are independent; run them concurrently:

- **Text**: `python kunstwerk.py configs/<prefix>.yaml --stop-after libretto` — fetch + translation (5–10 min).
- **Audio**: `python kunstwerk.py configs/<prefix>.yaml --skip-libretto --stop-after transcribe` — download, demucs on Modal, instrumental detection, Scribe + quality gate + Whisper fallback (10–20 min).

As soon as `audio/<prefix>/tracks.json` says `complete: true`, build the probe: `python kunstwerk.py configs/<prefix>.yaml --copyright-test --skip-libretto --skip-download` → `output/<prefix>_copyright_test.mp4`. Tell Jasper it's ready so he can upload it privately and check Content ID while the rest runs — that's the whole point of the probe: learn about a block before an hour of rendering.

When both halves are done: `python kunstwerk.py configs/<prefix>.yaml --skip-libretto --skip-download --skip-transcribe`. That runs the `align` stage (tripwire) and then the render. 4K (`res_divisor: 1`, the deliverable) takes ~30 min and ~1.5 GB; a `res_divisor: 4` preview takes ~7 min if you want eyes on it first. Don't run two renders of the same prefix at once (they share the CSV and output paths).

## 4. Read the tripwire before you spend the render

`kunstwerk` prints `Alignment OK` or a `REVIEW NEEDED` banner from `output/<prefix>-alignment-report.json`. Look at `review_reasons` and `notes`: black-screen fraction, coverage, longest gap, per-track coverage, ASR holes after the fallback, anchors found. Use judgment rather than thresholds alone — a 12%-black Carmen is fine to ship; 40% black or a 10-minute gap is not. Levers, in order: is a track an ensemble the ASR can't handle (note it, move on); did an anchor land wrong (`markers` in `make_video.py`); did translation parity break (block counts in the error); is the recording's edition different from the libretto (back to step 1). Say what you decided and why in the handover.

## 5. QA the video

`ffprobe` duration ≈ sum of track durations, 3840×2160. Grab frames with `ffmpeg -ss <t> -frames:v 1` at the title card (≈5 s), three or four sung moments (take timestamps from `output/<prefix>-chapters.txt` + 30–60 s), and near the end; look at them: two columns, speaker names bold, stage directions italic, text fits the frame, the translation is the matching passage. Skim `chapters.txt` — each line should read like a number's first words, not a stage direction.

## 6. Hand over

Tell Jasper: the files; the recording (cast/conductor/year, track count) and libretto edition; the alignment summary (coverage, black %, longest gap, weak tracks); the chapter list; what's manual (upload probe privately → check claims → upload the final, paste chapters into the description); time and cost actually used; anything you'd do differently. Then open and merge a PR per `/open-pr-and-merge` with `configs/<prefix>.yaml` and `libretti/<prefix>_*.txt`.

## Things that bite

- `end_idx` is exclusive; tracks are 1-based `NN`; `config_parser.py` derives `end_idx`, `overture_indices`, `characters`, `translation_file` — don't hand-type them.
- Modal must see `MODAL_PROFILE=chromatic` (the scripts set it themselves). `ELEVENLABS_API_KEY` is the only paid API the pipeline needs; the OpenAI key is optional and currently has no credits; Claude runs via the CLI on the subscription.
- yt-dlp needs this Mac's residential IP; it deprecated Python 3.10 (warnings are fine).
- Everything is idempotent: re-running a stage after a failure resumes. `--force` exists where overwriting matters (`fetch_libretto.py`, `translate.py`, `transcribe_tracks.py`).
- `KUNSTWERK_MAX_DURATION=600 python make_video.py …` renders only the first 10 minutes — handy to check typography at 4K without the full encode.
