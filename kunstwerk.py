#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional
from config_parser import parse_opera_config

def run_command(cmd: str, error_msg: str) -> None:
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"{error_msg} (exit code: {result.returncode})")

def process_opera(
    config_path: str, 
    skip_download: bool = False, 
    skip_transcribe: bool = False
) -> None:
    """Main function to process an opera from start to finish"""
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Processing opera using config: {config_path}")
    
    config = parse_opera_config(config_path)


    # Step 1: Download and separate audio
    if not skip_download:
        print("\n=== Downloading and separating audio ===")
        run_command(
            f"./separate.sh {config_path}",
            "Failed to download/separate audio"
        )

    # Step 2: Transcribe audio
    if not skip_transcribe:
        print("\n=== Transcribing audio ===")
        run_command(
            f"python transcribe_elevenlabs.py {config_path}",
            "Failed to transcribe audio using Elevenlabs"
        )
        run_command(
            f"python transcribe.py {config_path}",
            "Failed to transcribe audio using OpenAI"
        )

    # Step 3: Generate video
    print("\n=== Generating video ===")
    run_command(
        f"python make_video.py {config_path}",
        "Failed to generate video"
    )

def main():
    parser = argparse.ArgumentParser(description="Generate parallel subtitle videos for operas")
    parser.add_argument("config", help="Path to the opera configuration YAML file")
    parser.add_argument("--skip-download", action="store_true", 
                      help="Skip downloading and separating audio (use existing files)")
    parser.add_argument("--skip-transcribe", action="store_true",
                      help="Skip transcription (use existing transcription files)")
    
    args = parser.parse_args()

    try:
        process_opera(
            args.config, 
            args.skip_download, 
            args.skip_transcribe
        )
        print("\nSuccess! Opera processing completed.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
