#!/usr/bin/env python3
"""publish_metadata.py — the copy-pastable YouTube package for one opera.

    python publish_metadata.py configs/figaro.yaml

Writes output/<prefix>-youtube.txt: the video title, then the full description
with recording credits, cast and named chapter timestamps — everything that has
to be pasted into the YouTube upload form, in the order the form asks for it.
Also writes output/<prefix>-chapters.txt on its own, since that is the part that
gets re-pasted most often.

Chapters are named after the *recording's own track titles*, not the first sung
line: a track title reads "Act II: No. 13, Venite inginocchiatevi", which is what
a listener wants in a chapter list, whereas the first aligned words of a track are
as likely to be a stage direction or the tail of a recitative. The timestamps are
cumulative track durations, taken from the transcripts (the same numbers
make_video.py used to lay out the video) and falling back to the audio itself.

The credits cannot be derived from the audio — YouTube Music tags a track with a
featured artist, not a role — so they live in the config under `credits:`, and
this script only formats them. A config with no `credits:` block still produces a
correct title and chapter list; the recording section is simply omitted.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from typing import List, Optional

from config_parser import parse_opera_config

# ISO-639-1 for the languages this project has libretti in.
LANGUAGE_NAMES = {
    "it": "Italian", "de": "German", "fr": "French", "en": "English",
    "ru": "Russian", "cz": "Czech", "cs": "Czech", "es": "Spanish", "hu": "Hungarian",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get((code or "").lower(), (code or "").upper())


def hms(seconds: float) -> str:
    """YouTube chapter timestamp: m:ss, or h:mm:ss past the hour."""
    s = int(round(seconds))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ---------------------------------------------------------------------------
# track titles -> chapter names
# ---------------------------------------------------------------------------

def strip_common_prefix(titles: List[str]) -> List[str]:
    """Drop the boilerplate every track title of one album repeats.

    YouTube Music titles a track "Mozart: Le nozze di Figaro, K. 492, Act II: No.
    13, Venite inginocchiatevi" — the work is named 79 times over. The shared
    opening is exactly the longest common prefix across the album, so take it on a
    word boundary (never mid-word) and trim the punctuation left dangling.
    """
    if not titles:
        return []
    words = [t.split() for t in titles]

    def key(w: str) -> str:
        # Compare words ignoring trailing punctuation: the same album writes the
        # work's catalogue number "K. 492:" before an overture and "K. 492," before
        # an act, and a strict match would stop one word short and leave "492:"
        # glued to the front of every chapter.
        return w.rstrip(",:;.-–—").lower()

    n = 0
    while n < min(len(w) for w in words) and len({key(w[n]) for w in words}) == 1:
        n += 1
    if n == 0:
        return [t.strip() for t in titles]
    out = [" ".join(w[n:]).strip() for w in words]
    # A prefix that swallowed a whole title (a track named only by the work) is
    # useless — keep the originals rather than emit a blank chapter.
    if any(not t for t in out):
        return [t.strip() for t in titles]
    return [t.lstrip(" ,:;-–—.").strip() for t in out]


def tidy_title(title: str) -> str:
    """Undo the cosmetic damage YouTube does to long track titles.

    YouTube caps a track title around 100 characters and appends an ellipsis, which
    leaves the last clause hanging mid-phrase ('… Che soave zeffiretto "Letter').
    That trailing ellipsis is hard evidence the title was cut, so only then is it
    safe to drop the dangling remainder — a title that arrived intact keeps every
    quote and dash it was written with.
    """
    title = re.sub(r"\s+", " ", title).strip()
    truncated = bool(re.search(r"(\.\.\.+|…)\s*$", title))
    title = re.sub(r"[\s.]*(\.\.\.+|…)\s*$", "", title).strip()
    if truncated:
        if title.count('"') % 2 == 1:                  # a quote opened and never closed
            title = title[: title.rfind('"')].strip()
        title = re.sub(r"\s+[-–—]\s+\S.*$", "", title)  # a clause begun after " - "
    return title.rstrip(" ,;:-–—")


def track_durations(cfg) -> List[float]:
    """Per-track seconds for tracks [start_idx, end_idx), preferring the transcript
    durations make_video.py laid the video out with."""
    durs = []
    for i in range(cfg.start_idx, cfg.end_idx):
        path = f"{cfg.transcribed_dir}/{i:02d}.json"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = (json.load(f) or {}).get("duration")
            if d:
                durs.append(float(d))
                continue
        audio = f"{cfg.audio_dir}/{i:02d}.m4a"
        if not os.path.exists(audio):
            raise SystemExit(
                f"No duration for track {i:02d}: neither {path} nor {audio} exists. "
                f"Run the transcribe stage (or at least the download) first."
            )
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", audio],
            capture_output=True, text=True, check=True,
        )
        durs.append(float(out.stdout.strip()))
    return durs


def chapter_lines(cfg) -> List[str]:
    """`<timestamp> <name>` per track, starting at 0:00 (YouTube rejects a chapter
    list whose first entry is not 0:00)."""
    tracks_json = f"{cfg.audio_dir}/tracks.json"
    n = cfg.end_idx - cfg.start_idx
    if os.path.exists(tracks_json):
        with open(tracks_json, encoding="utf-8") as f:
            all_tracks = (json.load(f) or {}).get("tracks") or []
        titles = [t.get("title") or "" for t in all_tracks[cfg.start_idx - 1: cfg.end_idx - 1]]
        names = [tidy_title(t) for t in strip_common_prefix(titles)]
    else:
        print(f"WARNING: {tracks_json} not found — falling back to numbered chapters", file=sys.stderr)
        names = []
    if len(names) != n or not all(names):
        names = [f"Track {i}" for i in range(cfg.start_idx, cfg.end_idx)]

    lines, t = [], 0.0
    for name, dur in zip(names, track_durations(cfg)):
        lines.append(f"{hms(t)} {name}")
        t += dur
    return lines


# ---------------------------------------------------------------------------
# title + description
# ---------------------------------------------------------------------------

def video_title(cfg) -> str:
    return (f"{cfg.display_title} - full opera with "
            f"{language_name(cfg.translation_language)}/{language_name(cfg.language)} libretto")


def _cast_pairs(cast) -> List[tuple]:
    """Accept `- role: X / singer: Y` mappings, `- "Role: Singer"` strings, or a
    plain `Role: Singer` mapping."""
    pairs = []
    if isinstance(cast, dict):
        return [(str(r), str(s)) for r, s in cast.items()]
    for entry in cast or []:
        if isinstance(entry, dict):
            if "role" in entry or "singer" in entry:
                pairs.append((str(entry.get("role", "")), str(entry.get("singer", ""))))
            else:
                pairs.extend((str(r), str(s)) for r, s in entry.items())
        elif isinstance(entry, str) and ":" in entry:
            role, _, singer = entry.partition(":")
            pairs.append((role.strip(), singer.strip()))
        elif isinstance(entry, str):
            pairs.append(("", entry.strip()))
    return [(r, s) for r, s in pairs if r or s]


def description(cfg, chapters: List[str]) -> str:
    c = cfg.credits or {}
    src, tr = language_name(cfg.language), language_name(cfg.translation_language)
    parts: List[str] = []

    headline = cfg.display_title
    if c.get("composer"):
        headline += f" — {c['composer']}"
    parts.append(headline)

    blurb = (f"The complete opera, with the {src} libretto and its {tr} translation "
             f"side by side, synchronised to the singing.")
    if c.get("librettist"):
        blurb += f" Libretto by {c['librettist']}."
    parts.append(blurb)

    rec = []
    for key, label in (("conductor", None), ("orchestra", None), ("chorus", None)):
        if c.get(key):
            rec.append(f"{c[key]}, {key}" if key == "conductor" else str(c[key]))
    tail = ", ".join(str(c[k]) for k in ("label", "recorded", "year") if c.get(k))
    if tail:
        rec.append(tail)
    if rec:
        parts.append("RECORDING\n" + "\n".join(rec))

    pairs = _cast_pairs(c.get("cast"))
    if pairs:
        width = max(len(r) for r, _ in pairs)
        parts.append("CAST\n" + "\n".join(
            (f"{r.ljust(width)}  {s}".rstrip() if r else s) for r, s in pairs))

    parts.append("CHAPTERS\n" + "\n".join(chapters))
    return "\n\n".join(parts) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/<opera>.yaml")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="also write the package to stdout")
    args = ap.parse_args(argv)

    cfg = parse_opera_config(args.config)
    os.makedirs("output", exist_ok=True)

    chapters = chapter_lines(cfg)
    title = video_title(cfg)
    body = description(cfg, chapters)
    package = f"TITLE\n{title}\n\nDESCRIPTION\n{body}"

    ch_path = f"output/{cfg.file_prefix}-chapters.txt"
    yt_path = f"output/{cfg.file_prefix}-youtube.txt"
    with open(ch_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chapters) + "\n")
    with open(yt_path, "w", encoding="utf-8") as f:
        f.write(package)

    if args.show:
        print(package)
    print(f"wrote {yt_path} ({len(chapters)} chapters) and {ch_path}")
    if not cfg.credits:
        print(f"NOTE: no `credits:` block in {args.config} — the description has no "
              f"recording or cast section. Add one to name the conductor, orchestra and singers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
