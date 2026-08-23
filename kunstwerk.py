#!/usr/bin/env python3
"""Kunstwerk orchestrator: one config in, one subtitled opera video out.

    python kunstwerk.py configs/carmen.yaml
    python kunstwerk.py configs/carmen.yaml --skip-download --skip-transcribe   # just re-render
    python kunstwerk.py configs/carmen.yaml --stop-after libretto               # fetch + translate, then stop
    python kunstwerk.py configs/carmen.yaml --copyright-test                    # audio-only probe for Content ID
    python kunstwerk.py configs/carmen.yaml --skip-download --skip-transcribe --stop-after align   # just the alignment check
    python kunstwerk.py configs/carmen.yaml --strict-alignment                  # refuse to render a bad alignment

Stages, in order (each is its own script and idempotent — it skips work whose
output already exists — so re-running after a failure is safe):

  libretto    fetch_libretto.py      -> libretti/<prefix>_<lang>.txt (+ translation if the site has it)
              translate.py           -> libretti/<prefix>_<translation_language>.txt (Claude), if still missing
  download    separate.sh            -> audio/<prefix>/NN.m4a (download_album.py) and sep/<prefix>_sep/ (demucs)
              detect_instrumental.py -> sep/<prefix>_sep/instrumental.json (which tracks are purely orchestral)
  transcribe  transcribe_tracks.py   -> transcribed/<prefix>_transcribed/NN.json (+ quality.json); ElevenLabs Scribe,
              quality gate (coverage vs detected singing, loops), Modal Whisper / whisper-1 fallback fused into gaps
  align       make_video.py --align-only -> aligned_words_<prefix>.csv + output/<prefix>-alignment-report.json;
              prints a loud REVIEW NEEDED when the alignment looks bad (--strict-alignment makes that fatal)
  video       make_video.py          -> output/<prefix>-<res_divisor>.mp4
  publish     publish_metadata.py    -> output/<prefix>-youtube.txt (title, credits, cast, named chapters)
              and output/<prefix>-chapters.txt — everything the YouTube upload form asks for

The cheap, most-likely-to-fail stages (libretto lookup, translation, album
resolution) run before the expensive ones (demucs, transcription, render).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from config_parser import parse_opera_config

STAGES = ["libretto", "download", "transcribe", "align", "video", "publish"]


def _env() -> dict:
    """Subprocess environment with this interpreter's bin dir first on PATH, so
    `demucs`, `yt-dlp`, `ffmpeg` etc. resolve to the venv even when kunstwerk.py
    is launched from cron/launchd without the venv activated."""
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    return env


def run(cmd: str, error_msg: str, ok_codes=(0,)) -> int:
    print(f"\n$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, env=_env())
    if result.returncode not in ok_codes:
        raise RuntimeError(f"{error_msg} (exit code {result.returncode})")
    return result.returncode


def py(script: str) -> str:
    return f"{sys.executable} {script}"


def check_alignment(config, strict: bool) -> None:
    """Read make_video.py --align-only's report and shout if it needs review."""
    path = f"output/{config.file_prefix}-alignment-report.json"
    if not os.path.exists(path):
        print(f"WARNING: no alignment report at {path}", flush=True)
        return
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    reasons = report.get("review_reasons", [])
    summary = (f"{report['coverage_raw_alnum']:.0%} of words timed, "
               f"{report['black_frac']:.0%} of sung time blank, "
               f"longest gap {report['longest_gap_s'] / 60:.1f} min, "
               f"anchors on {report['tracks_anchored']}/{report['tracks_sung']} sung tracks")
    if not reasons:
        print(f"Alignment OK: {summary}", flush=True)
        return
    banner = "!" * 78
    print(f"\n{banner}\n!!! REVIEW NEEDED: alignment for {config.title} looks bad ({summary})", flush=True)
    for r in reasons:
        print(f"!!!   - {r}", flush=True)
    for n in report.get("notes", []):
        print(f"!!!   note: {n}", flush=True)
    print(f"!!!   details: {path}; fix with markers in make_video.py or re-transcribe the listed tracks\n{banner}\n", flush=True)
    if strict:
        raise RuntimeError("alignment needs review and --strict-alignment is set: " + "; ".join(reasons))


def process_opera(config_path: Path, args) -> None:
    config = parse_opera_config(str(config_path))
    print(f"Processing {config.title} ({config_path})")

    # --- libretto -----------------------------------------------------------
    if not args.skip_libretto:
        print("\n=== Libretto ===")
        if os.path.exists(config.libretto_path):
            print(f"{config.libretto_path} exists, skipping fetch")
        else:
            run(f"{py('fetch_libretto.py')} {config_path}", "Failed to fetch libretto")
        if os.path.exists(config.translation_path):
            print(f"{config.translation_path} exists, skipping translation")
        else:
            run(f"{py('translate.py')} {config_path}", "Failed to translate libretto")
    if args.stop_after == "libretto":
        return

    # --- download (+ separate) ---------------------------------------------
    if args.copyright_test:
        print("\n=== Copyright test (download only, no separation/transcription) ===")
        if not args.skip_download:
            run(f"{py('download_album.py')} {config_path}", "Failed to download audio")
        run(f"{py('copyright_test.py')} {config_path}", "Failed to generate copyright test video")
        return

    if not args.skip_download:
        print("\n=== Downloading and separating audio ===")
        run(f"./separate.sh {config_path}", "Failed to download/separate audio")
        run(f"{py('detect_instrumental.py')} {config_path}", "Failed to detect instrumental tracks")
    if args.stop_after == "download":
        return

    # --- transcribe ----------------------------------------------------------
    if not args.skip_transcribe:
        print("\n=== Transcribing audio ===")
        # exit 2 = some sung tracks still fail the quality gate after the fallback; the
        # alignment tripwire will name them, so warn and carry on.
        rc = run(f"{py('transcribe_tracks.py')} {config_path}", "Failed to transcribe audio", ok_codes=(0, 2))
        if rc == 2:
            print("WARNING: some tracks still have poor transcripts (see transcribed/<prefix>_transcribed/quality.json)", flush=True)
    if args.stop_after == "transcribe":
        return

    # --- align (cheap; the tripwire runs before the expensive render) --------
    print("\n=== Aligning transcript with libretto ===")
    run(f"{py('make_video.py')} {config_path} --align-only", "Failed to align transcript with libretto")
    check_alignment(config, strict=args.strict_alignment)
    if args.stop_after == "align":
        return

    # --- video ---------------------------------------------------------------
    print("\n=== Generating video ===")
    run(f"{py('make_video.py')} {config_path}", "Failed to generate video")
    print(f"\nOutput: output/{config.file_prefix}-{config.res_divisor}.mp4")
    if args.stop_after == "video":
        return

    # --- publish metadata (cheap; the last thing standing between an mp4 and an upload)
    print("\n=== Publish metadata ===")
    run(f"{py('publish_metadata.py')} {config_path}", "Failed to write publish metadata")
    print(f"Paste-ready title/description/chapters: output/{config.file_prefix}-youtube.txt")


def main():
    parser = argparse.ArgumentParser(description="Generate parallel-subtitle videos for operas")
    parser.add_argument("config", help="Path to the opera configuration YAML file")
    parser.add_argument("--skip-libretto", action="store_true", help="Skip libretto fetch/translation")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, separation and instrumental detection")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription")
    parser.add_argument("--stop-after", choices=STAGES, help="Stop after this stage")
    parser.add_argument("--copyright-test", action="store_true",
                        help="Download the audio and build a black-screen video to probe YouTube Content ID; no separation/transcription/render")
    parser.add_argument("--strict-alignment", action="store_true",
                        help="Fail instead of just warning when the alignment report says REVIEW NEEDED")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        process_opera(config_path, args)
        print("\nDone.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
