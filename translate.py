#!/usr/bin/env python3
"""Translate a libretto with Claude, block-for-block.

    python translate.py configs/carmen.yaml            # -> libretti/<prefix>_<translation_language>.txt
    python translate.py configs/carmen.yaml de --force # explicit target language, overwrite

Claude is called through the Claude Code CLI in headless mode (`claude -p`),
so the work is billed to the Claude subscription the CLI is logged into, not
to an API key. ANTHROPIC_API_KEY is deliberately stripped from the subprocess
environment so the CLI can't silently fall back to API billing. Log in once
with `claude` → `/login` if you see "Not logged in".

The output must mirror the source block-for-block (blank-line separated) and
line-for-line, because make_video.py pairs the two files by block and raises
on a mismatch. Each chunk of blocks is validated for newline parity and
retried; a chunk that still fails falls back to translating its blocks one at
a time, which almost always restores parity.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Tuple

from tqdm import tqdm

from config_parser import OperaConfig, parse_opera_config

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_WORKERS = 4  # concurrent `claude -p` calls; each has a few seconds of CLI startup

LANGUAGE_NAMES = {
    "en": "English", "de": "German", "it": "Italian", "fr": "French", "es": "Spanish",
    "ru": "Russian", "cs": "Czech", "pt": "Portuguese", "ja": "Japanese", "zh": "Chinese",
}


def find_claude_binary() -> str:
    candidates = [os.getenv("KUNSTWERK_CLAUDE_BIN"), shutil.which("claude"), os.path.expanduser("~/.local/bin/claude")]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("Claude Code CLI not found — install it or set KUNSTWERK_CLAUDE_BIN")


def chunk_text(text: str, chunk_size: int = 12) -> Iterator[Tuple[List[str], int]]:
    """Split text into chunks of roughly chunk_size blocks (double-newline separated)."""
    blocks = text.split("\n\n")
    for i in range(0, len(blocks), chunk_size):
        yield blocks[i:i + chunk_size], i


def create_translation_prompt(text: str, source_lang: str, target_lang: str, opera_title: str) -> str:
    src = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt = LANGUAGE_NAMES.get(target_lang, target_lang)
    return f"""You are an expert translator of opera libretti from {src} to {tgt}.
You are translating "{opera_title}". Translate the following passage, keeping its poetic and dramatic qualities while staying accurate. It is imperative that you preserve all line breaks and blank lines exactly as in the original: there must be a 1:1 correspondence between each line of the original and each line of the translation, including character names (translate or transliterate them as is conventional, keeping them in capitals on their own line) and stage directions in parentheses.

Here is the text to translate:

{text}

Reply with only the translation, with the same line-break structure, and nothing else — no preamble, no code fences."""


def validate_translation(source: str, translation: str) -> bool:
    """Newline parity — the same number of lines means the blocks will pair up."""
    return source.count("\n") == translation.count("\n")


class ClaudeCLI:
    """Thin wrapper over `claude -p` (headless Claude Code)."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: int = 600):
        self.binary = find_claude_binary()
        self.model = model
        self.timeout = timeout
        # No API key, and run from an empty directory so no CLAUDE.md gets pulled into context.
        self.env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        self.cwd = tempfile.mkdtemp(prefix="kunstwerk-claude-")

    def ask(self, prompt: str) -> str:
        cmd = [self.binary, "-p", "--no-session-persistence", "--model", self.model, "--tools", ""]
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=self.env,
                                cwd=self.cwd, timeout=self.timeout)
        out = result.stdout.strip()
        if result.returncode != 0 or "Not logged in" in out or "Not logged in" in result.stderr:
            raise RuntimeError(
                f"claude -p failed (exit {result.returncode}): {out or result.stderr.strip()}\n"
                f"If it says 'Not logged in', run `claude` and `/login` once on this machine."
            )
        # Strip an accidental code fence
        if out.startswith("```"):
            out = out.strip("`").strip()
            if out.lower().startswith("text\n"):
                out = out[5:]
        return out.strip()


def translate_text(
    client: ClaudeCLI, text: str, source_lang: str, target_lang: str, opera_title: str, max_attempts: int = 5
) -> str:
    """Translate one piece of text, retrying until newline parity holds."""
    last = ""
    for _ in range(max_attempts):
        last = client.ask(create_translation_prompt(text, source_lang, target_lang, opera_title))
        if validate_translation(text, last):
            return last
    raise ValueError(
        f"Failed to get a translation with matching line structure after {max_attempts} attempts.\n"
        f"Source:\n{text}\nLast translation:\n{last}"
    )


def translate_chunk(
    client: ClaudeCLI, chunk: List[str], source_lang: str, target_lang: str, opera_title: str, max_attempts: int = 5
) -> List[str]:
    """Translate a chunk of blocks; returns one translated block per source block."""
    if not any(b.strip() for b in chunk):
        return list(chunk)
    joined = "\n\n".join(chunk).strip()
    try:
        out = translate_text(client, joined, source_lang, target_lang, opera_title, max_attempts).split("\n\n")
        if len(out) == len(chunk):
            return out
    except ValueError:
        pass
    # Fallback: block by block. Smaller units make line parity far easier to hit.
    # Blocks can carry leading/trailing newlines (from runs of 3+ newlines in the source);
    # translate the stripped core and re-attach them so the structure is preserved exactly.
    print(f"  chunk failed parity check; retrying its {len(chunk)} blocks individually", file=sys.stderr)
    out = []
    for block in chunk:
        core = block.strip()
        if not core:
            out.append(block)
            continue
        lead = block[: len(block) - len(block.lstrip())]
        trail = block[len(block.rstrip()):]
        out.append(lead + translate_text(client, core, source_lang, target_lang, opera_title, max_attempts) + trail)
    return out


def translate_libretto(
    config: OperaConfig, target_lang: str, force: bool = False, model: str = DEFAULT_MODEL, workers: int = DEFAULT_WORKERS
) -> Path:
    source_path = Path(config.libretto_path)
    # The configured translation file for the default target language (so a custom
    # `translation_file:` is honoured); <prefix>_<lang>.txt for any other language.
    if target_lang == config.translation_language:
        target_path = Path(config.translation_path)
    else:
        target_path = Path(f"libretti/{config.file_prefix}_{target_lang}.txt")
    if target_path.exists() and not force:
        print(f"Translation already exists at {target_path}. Use --force to overwrite.")
        return target_path
    if not source_path.exists():
        raise FileNotFoundError(f"Source libretto not found: {source_path} (run fetch_libretto.py first)")

    client = ClaudeCLI(model=model)
    source_text = source_path.read_text(encoding="utf-8")
    source_blocks = source_text.split("\n\n")
    chunks = [c for c, _ in chunk_text(source_text)]

    def work(chunk: List[str]) -> List[str]:
        return translate_chunk(client, chunk, config.language, target_lang, config.title)

    translated_blocks: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for out in tqdm(ex.map(work, chunks), total=len(chunks), desc=f"Translating {config.file_prefix} -> {target_lang}"):
            translated_blocks.extend(out)

    if len(translated_blocks) != len(source_blocks):
        raise ValueError(
            f"Block count mismatch after translation: {len(source_blocks)} source vs {len(translated_blocks)} translated"
        )

    target_path.write_text("\n\n".join(translated_blocks), encoding="utf-8")
    print(f"Wrote {target_path} ({len(translated_blocks)} blocks)")
    return target_path


def main():
    parser = argparse.ArgumentParser(description="Translate an opera libretto with Claude (via the Claude Code CLI)")
    parser.add_argument("config", help="Path to the opera configuration YAML file")
    parser.add_argument("target_lang", nargs="?", help="Target language code (default: translation_language from the config, else 'en')")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing translation file")
    parser.add_argument("--model", default=os.getenv("KUNSTWERK_TRANSLATE_MODEL", DEFAULT_MODEL), help=f"Claude model (default {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Concurrent claude calls (default {DEFAULT_WORKERS})")
    args = parser.parse_args()

    config = parse_opera_config(args.config)
    target_lang = args.target_lang or config.translation_language
    try:
        translate_libretto(config, target_lang, args.force, args.model, args.workers)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
