"""Common batch generation helpers to support pluggable image models."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence


@dataclass
class GenerationRequest:
    prompt: str
    index: int
    total: int
    out_dir: Path
    prefix: str
    style_guide: Optional[str]
    ref_images: List[Path]
    record: Optional[dict]


def _ensure_unique_path(file_path: Path) -> Path:
    if not file_path.exists():
        return file_path
    stem = file_path.stem
    suffix = file_path.suffix
    counter = 1
    while True:
        candidate = file_path.with_name(f"{stem}_{counter:02d}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def save_binary_file(file_path: Path, data: bytes) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    target_path = _ensure_unique_path(file_path)
    if target_path != file_path:
        print(f"Note: {file_path.name} exists; saving as {target_path.name}")
    with open(target_path, "wb") as f:
        f.write(data)
    print(f"Saved: {target_path}")


class ImageModelProvider(Protocol):
    """Minimal interface providers must implement to plug into the batch runner."""

    @staticmethod
    def add_model_arguments(parser: argparse.ArgumentParser) -> None:
        """Register provider-specific CLI arguments."""

    def setup(self) -> None:
        """Perform any provider-specific setup prior to generation."""

    def generate(self, request: GenerationRequest) -> None:
        """Generate images for a single request."""


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to prompts TXT (one per line)",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to JSONL with field 'final_prompt' and optional 'characters' and 'global_style' fields",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="generated_images",
        help="Output directory",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of prompts",
    )
    parser.add_argument(
        "--character-card-dir",
        type=str,
        default=None,
        help="Directory of character card images to attach by name (uses JSONL 'characters')",
    )
    parser.add_argument(
        "--style-guide",
        type=str,
        default=None,
        help="Optional path to a style guide JSON or TXT; if JSONL is used, falls back to the record/global style",
    )


def load_character_cards(directory: Optional[str]) -> Dict[str, Path]:
    card_map: Dict[str, Path] = {}
    if not directory:
        return card_map
    cdir = Path(directory)
    if not cdir.exists() or not cdir.is_dir():
        print(f"character-card-dir not found or not a dir: {cdir}")
        return card_map
    for fn in os.listdir(cdir):
        p = cdir / fn
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".webp"}:
            card_map[p.stem.lower()] = p
    if card_map:
        print(f"Character cards loaded: {len(card_map)}")
    return card_map


def read_prompts(path: Path, limit: Optional[int]) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:limit] if limit else lines


def load_jsonl_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    print("Skipping JSONL entry that is not an object")
            except Exception as e:
                print(f"Skipping bad JSONL line: {e}")
    return records


def pick_ref_images(char_names: Optional[Sequence[str]], card_map: Dict[str, Path]) -> List[Path]:
    if not char_names or not card_map:
        return []
    refs: List[Path] = []
    for name in char_names:
        key = str(name).strip().lower()
        if not key:
            continue
        if key in card_map:
            refs.append(card_map[key])
            continue
        simple = "".join(ch for ch in key if ch.isalnum())
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
    seen = set()
    uniq: List[Path] = []
    for p in refs:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def load_style_from_path(path: Path) -> Optional[str]:
    try:
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            if suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            obj = json.loads(ln)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            gs = obj.get("global_style")
                            if isinstance(gs, str) and gs.strip():
                                return gs.strip()
                return None
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for key in ("guide", "global_style", "short_suffix"):
                    val = obj.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                return None
            return None
        return path.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def gather_generation_requests(args: argparse.Namespace) -> List[GenerationRequest]:
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card_map = load_character_cards(args.character_card_dir)

    provided_style_guide: Optional[str] = None
    if args.style_guide:
        provided_style_guide = load_style_from_path(Path(args.style_guide))

    requests: List[GenerationRequest] = []

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            raise RuntimeError(f"JSONL not found: {jsonl_path}")
        records = load_jsonl_records(jsonl_path)
        if args.limit is not None:
            records = records[: args.limit]
        print(f"Loaded {len(records)} records from {jsonl_path}")
        jsonl_style_guide: Optional[str] = None
        if not provided_style_guide:
            for rec in records:
                gs = rec.get("global_style")
                if isinstance(gs, str) and gs.strip():
                    jsonl_style_guide = gs.strip()
                    break
        for rec in records:
            prompt = (
                rec.get("final_prompt")
                or rec.get("prompt")
                or rec.get("description")
                or ""
            )
            if not isinstance(prompt, str) or not prompt.strip():
                print("Record missing prompt; skipping")
                continue
            chars = rec.get("characters") if isinstance(rec, dict) else None
            if not isinstance(chars, Sequence) or isinstance(chars, (str, bytes)):
                chars = None
            ref_imgs = pick_ref_images(chars, card_map)
            style_for_this: Optional[str] = provided_style_guide or jsonl_style_guide or rec.get("global_style")
            if isinstance(style_for_this, str):
                style_for_this = style_for_this.strip() or None
            requests.append(
                GenerationRequest(
                    prompt=prompt.strip(),
                    index=0,
                    total=0,
                    out_dir=out_dir,
                    prefix=args.prefix,
                    style_guide=style_for_this,
                    ref_images=ref_imgs,
                    record=rec,
                )
            )
    else:
        prompts_path = Path(args.prompts or "anime/prompts.txt")
        prompts = read_prompts(prompts_path, args.limit)
        print(f"Loaded {len(prompts)} prompts from {prompts_path}")
        for prompt in prompts:
            requests.append(
                GenerationRequest(
                    prompt=prompt,
                    index=0,
                    total=0,
                    out_dir=out_dir,
                    prefix=args.prefix,
                    style_guide=provided_style_guide,
                    ref_images=[],
                    record=None,
                )
            )

    total = len(requests)
    for idx, req in enumerate(requests, 1):
        req.index = idx
        req.total = total
    return requests


def run_batch(provider: ImageModelProvider, requests: Iterable[GenerationRequest]) -> None:
    provider.setup()
    for req in requests:
        print(f"\n[{req.index}/{req.total}] {req.prompt}")
        if req.ref_images:
            print("  + attaching character cards: " + ", ".join(p.stem for p in req.ref_images))
        if req.style_guide:
            print("  + applying global style guide")
        provider.generate(req)
