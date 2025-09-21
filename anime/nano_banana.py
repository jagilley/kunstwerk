"""
Batch image generation for Google's Gemini Image model using prompts.

Dependencies: pip install google-genai
Env: GEMINI_API_KEY must be set
"""

import argparse
import mimetypes
import os
from pathlib import Path
from typing import Iterable, List
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()


def save_binary_file(file_path: Path, data: bytes) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(data)
    print(f"Saved: {file_path}")


def read_prompts(path: Path, limit: int | None = None) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:limit] if limit else lines


def generate_for_prompt(
    client: genai.Client,
    model: str,
    prompt: str,
    out_dir: Path,
    prefix: str,
    index: int,
) -> None:
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    cfg = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])  # TEXT for any safety/textual notes

    variant = 0
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=cfg):
        try:
            cand = chunk.candidates[0]
            parts = cand.content.parts if cand and cand.content else None
            if not parts:
                continue
            part0 = parts[0]
            if getattr(part0, "inline_data", None) and getattr(part0.inline_data, "data", None):
                inline_data = part0.inline_data
                data_buffer: bytes = inline_data.data
                ext = mimetypes.guess_extension(inline_data.mime_type) or ".png"
                file_path = out_dir / f"{prefix}_{index:03d}_{variant:02d}{ext}"
                save_binary_file(file_path, data_buffer)
                variant += 1
            elif getattr(chunk, "text", None):
                # Streamed textual notes
                print(chunk.text)
        except Exception:
            # Skip malformed chunks
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from prompts using Gemini")
    parser.add_argument("--prompts", type=str, default="anime/prompts.txt", help="Path to prompts TXT (one per line)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-image-preview", help="Gemini model name")
    parser.add_argument("--outdir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--prefix", type=str, default="frame", help="Filename prefix")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of prompts")

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    model = args.model
    out_dir = Path(args.outdir)
    prompts_path = Path(args.prompts)
    prompts = read_prompts(prompts_path, args.limit)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt}")
        generate_for_prompt(client, model, prompt, out_dir, args.prefix, i)


if __name__ == "__main__":
    main()
