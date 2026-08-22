#!/usr/bin/env python3
"""Kunstwerk orchestrator: one config in, one subtitled opera video out.

    python kunstwerk.py configs/carmen.yaml
    python kunstwerk.py configs/carmen.yaml --skip-download --skip-transcribe   # just re-render
    python kunstwerk.py configs/carmen.yaml --stop-after libretto               # fetch + translate, then stop
    python kunstwerk.py configs/carmen.yaml --copyright-test                    # audio-only probe for Content ID

Stages, in order (each is its own script and idempotent — it skips work whose
output already exists — so re-running after a failure is safe):

  libretto    fetch_libretto.py      -> libretti/<prefix>_<lang>.txt (+ translation if the site has it)
              translate.py           -> libretti/<prefix>_<translation_language>.txt (Claude), if still missing
  download    separate.sh            -> audio/<prefix>/NN.m4a (download_album.py) and sep/<prefix>_sep/ (demucs)
              detect_instrumental.py -> sep/<prefix>_sep/instrumental.json (which tracks are purely orchestral)
  transcribe  transcribe_elevenlabs.py, transcribe.py -> transcribed/<prefix>_transcribed/NN.json
  video       make_video.py          -> output/<prefix>-<res_divisor>.mp4 + YouTube chapter list on stdout

The cheap, most-likely-to-fail stages (libretto lookup, translation, album
resolution) run before the expensive ones (demucs, transcription, render).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from config_parser import parse_opera_config

STAGES = ["libretto", "download", "transcribe", "video"]


def _env() -> dict:
    """Subprocess environment with this interpreter's bin dir first on PATH, so
    `demucs`, `yt-dlp`, `ffmpeg` etc. resolve to the venv even when kunstwerk.py
    is launched from cron/launchd without the venv activated."""
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    return env


def run(cmd: str, error_msg: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, env=_env())
    if result.returncode != 0:
        raise RuntimeError(f"{error_msg} (exit code {result.returncode})")


def py(script: str) -> str:
    return f"{sys.executable} {script}"


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
        run(f"{py('transcribe_elevenlabs.py')} {config_path}", "Failed to transcribe audio using ElevenLabs")
        run(f"{py('transcribe.py')} {config_path}", "Failed to transcribe audio using OpenAI")
    if args.stop_after == "transcribe":
        return

    # --- video ---------------------------------------------------------------
    print("\n=== Generating video ===")
    run(f"{py('make_video.py')} {config_path}", "Failed to generate video")
    print(f"\nOutput: output/{config.file_prefix}-{config.res_divisor}.mp4")


def main():
    parser = argparse.ArgumentParser(description="Generate parallel-subtitle videos for operas")
    parser.add_argument("config", help="Path to the opera configuration YAML file")
    parser.add_argument("--skip-libretto", action="store_true", help="Skip libretto fetch/translation")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, separation and instrumental detection")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription")
    parser.add_argument("--stop-after", choices=STAGES, help="Stop after this stage")
    parser.add_argument("--copyright-test", action="store_true",
                        help="Download the audio and build a black-screen video to probe YouTube Content ID; no separation/transcription/render")
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
