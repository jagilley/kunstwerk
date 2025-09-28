"""
Batch image generation for Google's Gemini Image model using prompts.

Dependencies: pip install google-genai
Env: GEMINI_API_KEY must be set
"""

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import List, Optional, Dict
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
    ref_images: Optional[List[Path]] = None,
) -> None:
    parts: List[types.Part] = [types.Part.from_text(text=prompt)]
    # Attach any provided reference images (character cards) as additional parts
    if ref_images:
        # Give the model an explicit instruction to honor attached references
        try:
            ref_names = ", ".join(p.stem for p in ref_images)
        except Exception:
            ref_names = "character references"
        parts.append(
            types.Part.from_text(
                text=(
                    "Use the attached character reference images as strict identity/style guides "
                    f"for: {ref_names}. Match faces, hair, outfit, and overall design."
                )
            )
        )
        for ipath in ref_images:
            try:
                with open(ipath, "rb") as f:
                    image_bytes = f.read()
                mime = mimetypes.guess_type(str(ipath))[0] or "image/png"
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
            except Exception as e:
                print(f"Warning: failed to read reference image {ipath}: {e}")

    contents = [types.Content(role="user", parts=parts)]
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
    parser.add_argument("--prompts", type=str, default=None, help="Path to prompts TXT (one per line)")
    parser.add_argument("--jsonl", type=str, default=None, help="Path to JSONL with field 'final_prompt' and optional 'characters' array")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-image-preview", help="Gemini model name")
    parser.add_argument("--outdir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--prefix", type=str, default="frame", help="Filename prefix")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of prompts")
    parser.add_argument("--character-card-dir", type=str, default=None, help="Directory of character card images to attach by name (uses JSONL 'characters')")

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    model = args.model
    out_dir = Path(args.outdir)
    # Build a map of available character card images by filename stem
    card_map: Dict[str, Path] = {}
    if args.character_card_dir:
        cdir = Path(args.character_card_dir)
        if cdir.exists() and cdir.is_dir():
            for fn in os.listdir(cdir):
                p = cdir / fn
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in {".png", ".jpg", ".jpeg", ".webp"}:
                    card_map[p.stem.lower()] = p
            if card_map:
                print(f"Character cards loaded: {len(card_map)}")
        else:
            print(f"character-card-dir not found or not a dir: {cdir}")

    def pick_ref_images(char_names: Optional[List[str]]) -> List[Path]:
        if not char_names or not card_map:
            return []
        refs: List[Path] = []
        for name in char_names:
            key = str(name).strip().lower()
            # try exact stem match first
            if key in card_map:
                refs.append(card_map[key])
                continue
            # relaxed matching: remove non-alnum
            simple = "".join(ch for ch in key if ch.isalnum())
            # best-effort lookup
            found = None
            for stem, path in card_map.items():
                s2 = "".join(ch for ch in stem if ch.isalnum())
                if s2 == simple or s2.startswith(simple) or simple.startswith(s2):
                    found = path
                    break
            if found:
                refs.append(found)
            else:
                print(f"Note: no card found for character '{name}'")
        # de-duplicate while preserving order
        seen = set()
        uniq: List[Path] = []
        for p in refs:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        return uniq

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            raise RuntimeError(f"JSONL not found: {jsonl_path}")
        records: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    records.append(json.loads(ln))
                except Exception as e:
                    print(f"Skipping bad JSONL line: {e}")
        if args.limit is not None:
            records = records[: args.limit]
        print(f"Loaded {len(records)} records from {jsonl_path}")
        total = len(records)
        for i, rec in enumerate(records, 1):
            prompt = (
                rec.get("final_prompt")
                or rec.get("prompt")
                or rec.get("description")
                or ""
            )
            if not prompt:
                print(f"Record {i}: missing prompt; skipping")
                continue
            chars = rec.get("characters") if isinstance(rec, dict) else None
            if not isinstance(chars, list):
                chars = None
            ref_imgs = pick_ref_images(chars)
            print(f"\n[{i}/{total}] {prompt}")
            if ref_imgs:
                print("  + attaching character cards: " + ", ".join(p.stem for p in ref_imgs))
            generate_for_prompt(client, model, prompt, out_dir, args.prefix, i, ref_images=ref_imgs)
        return

    # Fallback to plain TXT prompts
    prompts_path = Path(args.prompts or "anime/prompts.txt")
    prompts = read_prompts(prompts_path, args.limit)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt}")
        generate_for_prompt(client, model, prompt, out_dir, args.prefix, i)


if __name__ == "__main__":
    main()
