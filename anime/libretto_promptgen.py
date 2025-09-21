#!/usr/bin/env python3
"""
Generate image prompts from a libretto using a sliding-window LLM pass.

Outputs both a JSONL of structured prompts and a plain TXT of final
prompts usable by anime/nano_banana.py.

Provider: Anthropics Claude by default (requires ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import anthropic
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PromptSuggestion:
    title: str
    description: str
    shot: Optional[str] = None
    time_of_day: Optional[str] = None
    lighting: Optional[str] = None
    palette: Optional[str] = None
    characters: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    nsfw: Optional[bool] = False
    source_window_index: Optional[int] = None

    def to_prompt(self, style_suffix: str) -> str:
        parts = [self.description.strip()]
        # fold additional cinematography cues into the description if present
        extras: List[str] = []
        if self.shot:
            extras.append(f"{self.shot.strip()} shot")
        if self.time_of_day:
            extras.append(self.time_of_day.strip())
        if self.lighting:
            extras.append(f"{self.lighting.strip()} lighting")
        if self.palette:
            # Palette may be a string or a list; normalize to a readable string
            if isinstance(self.palette, list):  # type: ignore[unreachable]
                pal = ", ".join(
                    [str(p).strip() for p in self.palette if str(p).strip()]
                )
            else:
                pal = str(self.palette).strip()
            if pal:
                extras.append(f"color palette: {pal}")
        if extras:
            parts.append(", ".join(extras))
        if style_suffix:
            parts.append(style_suffix.strip())
        # Join with commas where appropriate but end as a single line
        return ", ".join([p for p in parts if p]).replace("\n", " ")


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_blocks(text: str) -> List[str]:
    """Split by blank lines into blocks (paragraphs/stage directions)."""
    # Normalize line endings and strip trailing whitespace from lines
    normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    # Split on two or more newlines to get text blocks
    blocks = re.split(r"\n{2,}", normalized.strip())
    # Remove empty blocks
    return [b for b in blocks if b.strip()]


def sliding_windows(blocks: List[str], window: int, stride: int) -> Iterable[Tuple[int, List[str]]]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    i = 0
    n = len(blocks)
    while i < n:
        yield i, blocks[i : min(i + window, n)]
        if i + window >= n:
            break
        i += stride


def _extract_json(s: str) -> Any:
    """Extract the first JSON value (array or object) from a string.

    Handles code fences and leading/trailing text.
    """
    # Try to locate fenced json
    m = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", s, flags=re.S)
    if m:
        s = m.group(1)
    # Fallback: find first array/object
    else:
        m = re.search(r"(\[.*\]|\{.*\})", s, flags=re.S)
        if m:
            s = m.group(1)
    return json.loads(s)


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 2000) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package not installed. Please pip install anthropic.")
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if not client.api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # anthropic SDK returns a list of content parts
    return msg.content[0].text  # type: ignore[index]


PROMPT_TEMPLATE = (
    "You are an expert storyboard artist for opera-to-anime adaptation. "
    "Given a libretto fragment from the opera '{opera_title}', propose up to {max_per_window} visually distinct still-image ideas that would make compelling anime frames.\n\n"
    "Requirements:\n"
    "- Each idea must be a single, concrete cinematic moment (no sequences).\n"
    "- Be literal to the text while compositing a strong shot (framing, action, mood).\n"
    "- Prefer stage directions for visuals, but you may summarize dialogue into action.\n"
    "- Return ONLY a JSON array of objects with keys: title, description, shot, time_of_day, lighting, palette, characters (array), tags (array), nsfw (boolean).\n"
    "- The 'description' must be one sentence that fully describes the frame. Do NOT include the opera title.\n"
    "- Avoid repeating essentially the same shot within this window.\n\n"
    "Libretto fragment (blocks, preserve line breaks):\n"
    "---\n"
    "{fragment}\n"
    "---\n"
)


def generate_prompts_for_window(
    blocks: List[str],
    window_index: int,
    opera_title: str,
    max_per_window: int,
) -> List[PromptSuggestion]:
    fragment = "\n\n".join(blocks)
    prompt = PROMPT_TEMPLATE.format(
        opera_title=opera_title,
        max_per_window=max_per_window,
        fragment=fragment,
    )
    raw = call_anthropic(prompt)
    data = _extract_json(raw)
    suggestions: List[PromptSuggestion] = []
    if not isinstance(data, list):
        raise ValueError("LLM did not return a JSON array")
    for item in data:
        if not isinstance(item, dict):
            continue
        suggestions.append(
            PromptSuggestion(
                title=str(item.get("title", "")).strip() or "untitled",
                description=str(item.get("description", "")).strip(),
                shot=(item.get("shot") or None),
                time_of_day=(item.get("time_of_day") or None),
                lighting=(item.get("lighting") or None),
                palette=(item.get("palette") or None),
                characters=item.get("characters") or None,
                tags=item.get("tags") or None,
                nsfw=bool(item.get("nsfw", False)),
                source_window_index=window_index,
            )
        )
    return suggestions


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:80]


def aggregate_suggestions(
    all_windows: List[List[PromptSuggestion]],
    max_total: Optional[int] = None,
) -> List[PromptSuggestion]:
    seen: set[str] = set()
    out: List[PromptSuggestion] = []
    for window_suggestions in all_windows:
        for s in window_suggestions:
            key = slugify(s.title or s.description)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if max_total and len(out) >= max_total:
                return out
    return out


def derive_defaults_from_path(libretto_path: Path) -> Tuple[str, str]:
    name = libretto_path.stem  # e.g., gotterdammerung_en
    # Try to extract a nice title
    title = name.split("_")[0].replace("-", " ").title()
    style_suffix = f"Still from {title}, cinematic artistic anime"
    return title, style_suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate anime image prompts from a libretto via sliding window LLM")
    parser.add_argument("libretto", type=str, help="Path to libretto text file (e.g., libretti/gotterdammerung_en.txt)")
    parser.add_argument("--opera-title", type=str, default=None, help="Opera title override for prompting")
    parser.add_argument("--style-suffix", type=str, default=None, help="Suffix appended to each prompt (e.g., 'Still from …, cinematic artistic anime')")
    parser.add_argument("--window", type=int, default=10, help="Window size in blocks")
    parser.add_argument("--stride", type=int, default=6, help="Stride in blocks")
    parser.add_argument("--max-per-window", type=int, default=2, help="Max suggestions per window")
    parser.add_argument("--max-total", type=int, default=None, help="Optional global cap on suggestions")
    parser.add_argument("--out-jsonl", type=str, default=None, help="Output JSONL path for structured prompts")
    parser.add_argument("--out-txt", type=str, default=None, help="Output TXT path of final prompts (one per line)")
    parser.add_argument("--start-line", type=int, default=None, help="Optional 1-based start line (inclusive) of libretto slice")
    parser.add_argument("--end-line", type=int, default=None, help="Optional 1-based end line (inclusive) of libretto slice")
    parser.add_argument("--dry-run", action="store_true", help="Print suggestions and exit (no files)")

    args = parser.parse_args()

    libretto_path = Path(args.libretto)
    if not libretto_path.exists():
        print(f"Libretto not found: {libretto_path}")
        sys.exit(1)

    text = read_text(libretto_path)

    # Optional line slicing (1-based inclusive indices)
    if args.start_line is not None or args.end_line is not None:
        lines = text.split("\n")
        total_lines = len(lines)
        start = max(1, args.start_line) if args.start_line is not None else 1
        end = min(total_lines, args.end_line) if args.end_line is not None else total_lines
        if start <= 0 or end <= 0:
            print("start-line and end-line must be positive (1-based)")
            sys.exit(1)
        if end < start:
            print(f"Invalid range: end-line ({end}) < start-line ({start})")
            sys.exit(1)
        text = "\n".join(lines[start - 1 : end])
        print(f"Slicing lines {start}..{end} of {total_lines}")

    blocks = split_into_blocks(text)
    opera_title, default_style_suffix = derive_defaults_from_path(libretto_path)
    opera_title = args.opera_title or opera_title
    style_suffix = args.style_suffix or default_style_suffix

    # Determine output paths
    out_dir = Path("output/prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = libretto_path.stem  # e.g., gotterdammerung_en
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else out_dir / f"{base_name}.jsonl"
    out_txt = Path(args.out_txt) if args.out_txt else out_dir / f"{base_name}.txt"

    print(f"Blocks: {len(blocks)} | window={args.window} stride={args.stride}")

    all_window_suggestions: List[List[PromptSuggestion]] = []
    for widx, win_blocks in sliding_windows(blocks, args.window, args.stride):
        try:
            suggestions = generate_prompts_for_window(
                win_blocks, widx, opera_title, args.max_per_window
            )
        except Exception as e:
            print(f"Window {widx}: LLM error: {e}")
            continue
        all_window_suggestions.append(suggestions)
        print(f"Window {widx}: {len(suggestions)} suggestions")

    final_suggestions = aggregate_suggestions(all_window_suggestions, args.max_total)
    print(f"Total unique suggestions: {len(final_suggestions)}")

    if args.dry_run:
        for i, s in enumerate(final_suggestions, 1):
            print(f"\n[{i}] {s.title}\n{ s.to_prompt(style_suffix)}")
        return

    # Write JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in final_suggestions:
            rec = asdict(s)
            rec["final_prompt"] = s.to_prompt(style_suffix)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write TXT
    with open(out_txt, "w", encoding="utf-8") as f:
        for s in final_suggestions:
            f.write(s.to_prompt(style_suffix) + "\n")

    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
