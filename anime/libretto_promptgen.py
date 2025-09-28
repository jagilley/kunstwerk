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
from typing import Any, Iterable, List, Optional, Tuple
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
    "You are composing frame-by-frame shot ideas for a single coherent anime film adaptation of the opera '{opera_title}'. "
    "Given the libretto fragment, propose up to {max_per_window} plausible film stills — frames that look like they were captured mid-scene from one continuous production.\n\n"
    "Global Film Style — apply consistently across ALL frames in this session:\n"
    "{global_style}\n\n"
    "Requirements (film stills, not posters):\n"
    "- Each idea is a single captured frame (no multi-beat sequences, no montage).\n"
    "- Be literal to the libretto; prefer stage directions and diegetic action over abstract symbolism.\n"
    "- Favor grounded, production-plausible staging and camera work; avoid impossible physics or floating cameras.\n"
    "- Describe the shot as it appears on screen (framing, action, mood), not as instructions.\n"
    "- Keep continuity with the Global Film Style (palette, lighting character, lens feel, era, texture).\n"
    "- Return ONLY a JSON array of objects with keys: title, description, shot, time_of_day, lighting, palette, characters (array), tags (array), nsfw (boolean).\n"
    "- The 'description' must be ONE sentence that fully describes the frame. Do NOT include the opera title.\n"
    "- Avoid repeating essentially the same shot within this window.\n\n"
    "Character cards (if provided):\n"
    "- Available character reference images (by filename stem) will be listed below.\n"
    "- When relevant, set the 'characters' array to the exact names from this list (filename stem).\n"
    "- Downstream image generation will attach the matching references — keep designs consistent and avoid reinventing looks.\n\n"
    "Libretto fragment (blocks, preserve line breaks):\n"
    "---\n"
    "{fragment}\n"
    "---\n"
)

# A separate prompt used to derive a single coherent style guide for all frames.
STYLE_GUIDE_TEMPLATE = (
    "You are the supervising art director for a single anime film adaptation of the opera '{opera_title}'. "
    "From the libretto excerpt below, derive a cohesive visual style guide that keeps all generated frames looking like they belong to ONE movie.\n\n"
    "Output STRICTLY a JSON object with keys:\n"
    "- short_suffix: a concise one-line style tail to append to every image prompt (avoid artist names).\n"
    "- guide: a compact paragraph (3–6 lines) describing the film's art direction: palette, lighting, lens/shot feel, texture/grain, era/format (e.g., 16:9 1080p), production vibe (generic, not name-dropping living artists), and any continuity notes.\n\n"
    "Constraints:\n"
    "- Do NOT reference specific living artists.\n"
    "- Favor practical, production-plausible details (camera/lens feeling, color grading, film/scan texture, background style).\n"
    "- Keep it applicable across the whole story; do not lock to a single scene.\n\n"
    "Libretto excerpt for style inference:\n"
    "---\n"
    "{fragment}\n"
    "---\n"
)


def generate_prompts_for_window(
    blocks: List[str],
    window_index: int,
    opera_title: str,
    max_per_window: int,
    available_characters: Optional[List[str]] = None,
    global_style: Optional[str] = None,
) -> List[PromptSuggestion]:
    fragment = "\n\n".join(blocks)
    prompt = PROMPT_TEMPLATE.format(
        opera_title=opera_title,
        max_per_window=max_per_window,
        fragment=fragment,
        global_style=(global_style or "(No explicit style — keep continuity plausible and grounded)")
    )
    # If character cards are available, append a short list the model can reference
    if available_characters:
        prompt += "\nAvailable character cards (filename stems):\n"
        prompt += ", ".join(sorted(available_characters)) + "\n\n"
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
    # Conservative, generic default; will be replaced by auto style if enabled
    style_suffix = (
        f"film still from a single {title} anime adaptation, 16:9, grounded composition, subtle film grain, cohesive palette, no text overlays, no credits"
    )
    return title, style_suffix


def generate_global_style(
    all_blocks: List[str],
    opera_title: str,
    style_blocks: int,
) -> Tuple[str, str]:
    """Use the first N blocks to infer a global film style.

    Returns (short_suffix, guide_paragraph).
    """
    if style_blocks <= 0:
        style_blocks = min(12, len(all_blocks)) or len(all_blocks)
    excerpt = "\n\n".join(all_blocks[:style_blocks])
    style_prompt = STYLE_GUIDE_TEMPLATE.format(opera_title=opera_title, fragment=excerpt)
    raw = call_anthropic(style_prompt, max_tokens=1000)
    try:
        obj = _extract_json(raw)
        if isinstance(obj, dict):
            short_suffix = str(obj.get("short_suffix") or "").strip()
            guide = str(obj.get("guide") or "").strip()
            if short_suffix and guide:
                return short_suffix, guide
    except Exception:
        pass
    # Fallback minimal style if parsing fails
    return (
        "cohesive anime film still, 16:9, grounded staging, subtle grain, consistent color grading",
        "A single cohesive anime film look: 16:9 frame, neutral yet cinematic color grading, soft key with gentle fill, practical production staging, and subtle film-grain texture for continuity.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate anime image prompts from a libretto via sliding window LLM")
    parser.add_argument("libretto", type=str, help="Path to libretto text file (e.g., libretti/gotterdammerung_en.txt)")
    parser.add_argument("--opera-title", type=str, default=None, help="Opera title override for prompting")
    parser.add_argument("--style-suffix", type=str, default=None, help="Suffix appended to each prompt (overrides auto style)")
    parser.add_argument("--no-auto-style", action="store_true", help="Disable LLM-derived global style guide and use default/manual suffix")
    parser.add_argument("--style-blocks", type=int, default=18, help="How many initial text blocks to sample for global style inference")
    parser.add_argument("--style-out", type=str, default=None, help="Optional path to save the inferred style guide JSON")
    parser.add_argument("--window", type=int, default=10, help="Window size in blocks")
    parser.add_argument("--stride", type=int, default=6, help="Stride in blocks")
    parser.add_argument("--max-per-window", type=int, default=2, help="Max suggestions per window")
    parser.add_argument("--max-total", type=int, default=None, help="Optional global cap on suggestions")
    parser.add_argument("--out-jsonl", type=str, default=None, help="Output JSONL path for structured prompts")
    parser.add_argument("--out-txt", type=str, default=None, help="Output TXT path of final prompts (one per line)")
    parser.add_argument("--start-line", type=int, default=None, help="Optional 1-based start line (inclusive) of libretto slice")
    parser.add_argument("--end-line", type=int, default=None, help="Optional 1-based end line (inclusive) of libretto slice")
    parser.add_argument("--dry-run", action="store_true", help="Print suggestions and exit (no files)")
    parser.add_argument(
        "--character-card-dir",
        type=str,
        default=None,
        help="Directory containing character reference images (filenames indicate character names)",
    )

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

    # Stage 1: derive a global style guide (unless disabled or manually provided)
    short_style_suffix: Optional[str] = None
    style_guide_paragraph: Optional[str] = None
    if not args.no_auto_style and not args.style_suffix:
        try:
            short_style_suffix, style_guide_paragraph = generate_global_style(blocks, opera_title, args.style_blocks)
            print("Inferred global style guide.")
        except Exception as e:
            print(f"Warning: failed to infer style guide, falling back to defaults: {e}")
    # Final suffix applied to every prompt
    style_suffix = args.style_suffix or short_style_suffix or default_style_suffix

    # Determine output paths
    out_dir = Path("output/prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = libretto_path.stem  # e.g., gotterdammerung_en
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else out_dir / f"{base_name}.jsonl"
    out_txt = Path(args.out_txt) if args.out_txt else out_dir / f"{base_name}.txt"

    # Optional: list available character cards (by filename stem)
    available_characters: Optional[List[str]] = None
    if args.character_card_dir:
        card_dir = Path(args.character_card_dir)
        if card_dir.exists() and card_dir.is_dir():
            stems: List[str] = []
            for fn in os.listdir(card_dir):
                p = card_dir / fn
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in {".png", ".jpg", ".jpeg", ".webp"}:
                    stems.append(p.stem)
            if stems:
                available_characters = sorted(set(stems))
                print(f"Character cards found ({len(available_characters)}): {', '.join(available_characters)}")
        else:
            print(f"character-card-dir not found or not a dir: {card_dir}")

    print(f"Blocks: {len(blocks)} | window={args.window} stride={args.stride}")

    all_window_suggestions: List[List[PromptSuggestion]] = []
    # Build a compact global style string for in-prompt continuity
    global_style_for_prompt = style_guide_paragraph or style_suffix

    for widx, win_blocks in sliding_windows(blocks, args.window, args.stride):
        try:
            suggestions = generate_prompts_for_window(
                win_blocks,
                widx,
                opera_title,
                args.max_per_window,
                available_characters=available_characters,
                global_style=global_style_for_prompt,
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
            if style_guide_paragraph:
                rec["global_style"] = style_guide_paragraph
            elif style_suffix:
                rec["global_style"] = style_suffix
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write TXT
    with open(out_txt, "w", encoding="utf-8") as f:
        for s in final_suggestions:
            f.write(s.to_prompt(style_suffix) + "\n")

    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_txt}")

    # Optionally save the style guide separately for reference
    if args.style_out:
        style_out = Path(args.style_out)
        style_out.parent.mkdir(parents=True, exist_ok=True)
        style_payload = {
            "opera_title": opera_title,
            "short_suffix": style_suffix,
            "guide": style_guide_paragraph or global_style_for_prompt,
            "source": str(libretto_path),
            "blocks_sampled": None if args.no_auto_style else args.style_blocks,
        }
        with open(style_out, "w", encoding="utf-8") as sf:
            json.dump(style_payload, sf, ensure_ascii=False, indent=2)
        print(f"Wrote {style_out}")


if __name__ == "__main__":
    main()
