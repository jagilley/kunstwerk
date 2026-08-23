#!/usr/bin/env python3
"""Compare transcript sets for one opera (ASR bake-off / quality-gate metrics).

    python asr_bakeoff.py configs/carmen.yaml scribe=transcribed/carmen_transcribed \
        whisperv2=transcribed/carmen_whisperv2_transcribed whisperv3=transcribed/carmen_whisperv3_transcribed
    python asr_bakeoff.py configs/carmen.yaml --fuse scribe=... whisperv2=... --out transcribed/carmen_fused_transcribed
    python asr_bakeoff.py configs/carmen.yaml --reports scribe=out/scribe.json whisperv2=out/v2.json ...

Per track and per provider: real (alphanumeric) word count, distinct-word ratio, loop
fraction (what align.collapse_transcript_loops removes), longest hole between words, and
coverage — the fraction of the track's *vocal* time (0.5 s frames where the demucs stem
carries energy relative to the mix, same test as detect_instrumental.py) lying within
`--near` seconds of a kept word — plus the fraction of words that land in frames with no
vocal energy at all ("spurious", a hallucination signal).

--fuse writes a naive fused set: the first provider's (cleaned, de-looped) words, plus the
second provider's words inside the first's holes longer than --gap seconds.

--reports tabulates saved make_video.py --align-only reports (output/<prefix>-alignment-report.json
copied aside per provider).
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from align import collapse_transcript_loops, normalise_for_matching
from classes import TranscriptionWord
from detect_instrumental import FRAME_S, SILENCE_DB, SR, VOCAL_RATIO_THRESHOLD, _decode, _frame_rms


def load_words(path: Path) -> Tuple[List[TranscriptionWord], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    words = [TranscriptionWord(start=float(w["start"]), end=float(w["end"]), word=str(w["word"]))
             for w in (d.get("words") or [])]
    return words, float(d.get("duration") or 0.0)


def clean_words(words: List[TranscriptionWord]) -> Tuple[List[TranscriptionWord], int, int]:
    """Drop empty/punctuation tokens, then collapse hallucination loops (what make_video.py does)."""
    real = [w for w in words if normalise_for_matching(w.word)]
    kept, n_loop = collapse_transcript_loops(real)
    return kept, len(words) - len(real), n_loop


def vocal_mask(mix_path: Optional[Path], stem_path: Path) -> np.ndarray:
    """Boolean per-FRAME_S-frame vocal-activity mask (detect_instrumental's test; stem-only
    energy test if the mix is missing)."""
    voc = _decode(str(stem_path))
    frame_len = int(SR * FRAME_S)
    voc_rms = _frame_rms(voc, frame_len)
    silence_lin = 10 ** (SILENCE_DB / 20.0)
    if mix_path is None or not mix_path.exists():
        return voc_rms > 10 ** (-40.0 / 20.0)
    mix = _decode(str(mix_path))
    n = min(len(mix), len(voc))
    mix_rms = _frame_rms(mix[:n], frame_len)
    voc_rms = voc_rms[: len(mix_rms)]
    active = mix_rms > silence_lin
    ratio = np.where(active, voc_rms / np.maximum(mix_rms, 1e-9), 0.0)
    return active & (ratio > VOCAL_RATIO_THRESHOLD)


def masks_for(prefix: str, tracks: List[int], cache: Path) -> Dict[int, np.ndarray]:
    cached: Dict[str, List[int]] = {}
    if cache.exists():
        cached = json.loads(cache.read_text())
    out: Dict[int, np.ndarray] = {}
    dirty = False
    for k in tracks:
        nn = str(k).zfill(2)
        if nn in cached:
            out[k] = np.array(cached[nn], dtype=bool)
            continue
        stem = None
        for cand in (Path("sep") / f"{prefix}_sep" / "htdemucs" / nn / "vocals.m4a",
                     Path("sep") / f"{prefix}_sep" / "htdemucs" / nn / "vocals.wav"):
            if cand.exists():
                stem = cand
                break
        if stem is None:
            continue
        mix = Path("audio") / prefix / f"{nn}.m4a"
        m = vocal_mask(mix if mix.exists() else None, stem)
        out[k] = m
        cached[nn] = [int(x) for x in m]
        dirty = True
    if dirty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached))
    return out


def covered_intervals(words: List[TranscriptionWord], near: float, dur: float) -> List[Tuple[float, float]]:
    ivals = sorted((max(0.0, w.start - near), min(dur, w.end + near)) for w in words)
    merged: List[List[float]] = []
    for a, b in ivals:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def track_metrics(words_raw: List[TranscriptionWord], dur: float, mask: Optional[np.ndarray], near: float) -> Dict:
    kept, n_empty, n_loop = clean_words(words_raw)
    n_real = len(kept) + n_loop
    toks = [normalise_for_matching(w.word) for w in kept]
    distinct = len(set(toks)) / len(toks) if toks else 0.0
    # holes between kept words, including track edges
    starts = np.array(sorted(w.start for w in kept), dtype=float)
    edges = np.concatenate(([0.0], starts, [dur])) if len(starts) else np.array([0.0, dur])
    longest_hole = float(np.diff(edges).max()) if dur > 0 else 0.0
    ivals = covered_intervals(kept, near, dur)
    covered_s = sum(b - a for a, b in ivals)
    m = {
        "tokens": len(words_raw), "real_words": n_real, "kept_words": len(kept),
        "loop_dropped": n_loop, "loop_frac": round(n_loop / n_real, 3) if n_real else 0.0,
        "distinct_ratio": round(distinct, 3),
        "longest_hole_s": round(longest_hole, 1),
        "coverage_dur": round(covered_s / dur, 3) if dur > 0 else 0.0,
        "coverage_vocal": None, "vocal_s": None, "spurious_frac": None,
    }
    if mask is not None and len(mask):
        n_frames = len(mask)
        mids = (np.arange(n_frames) + 0.5) * FRAME_S
        cov = np.zeros(n_frames, dtype=bool)
        for a, b in ivals:
            cov |= (mids >= a) & (mids < b)
        n_vocal = int(mask.sum())
        m["vocal_s"] = round(n_vocal * FRAME_S, 1)
        m["coverage_vocal"] = round(float((cov & mask).sum() / n_vocal), 3) if n_vocal else None
        # words whose midpoint sits in a frame with no vocal energy (±1 frame tolerance)
        if kept:
            wm = np.array([(w.start + w.end) / 2 for w in kept])
            fi = np.clip((wm / FRAME_S).astype(int), 0, n_frames - 1)
            padded = np.concatenate(([False], mask, [False]))
            near_vocal = padded[fi] | padded[fi + 1] | padded[fi + 2]
            m["spurious_frac"] = round(float((~near_vocal).mean()), 3)
        else:
            m["spurious_frac"] = 0.0
    return m


def fmt_row(name: str, m: Dict, width: int = 10) -> str:
    def f(x, pct=False):
        if x is None:
            return "-"
        return f"{x:.0%}" if pct else (f"{x:.2f}" if isinstance(x, float) else str(x))
    return (f"{name:<{width}} {f(m['real_words']):>6} {f(m['kept_words']):>6} {f(m['loop_frac'], True):>6} "
            f"{f(m['distinct_ratio']):>6} {f(m['coverage_vocal'], True):>7} {f(m['coverage_dur'], True):>6} "
            f"{f(m['spurious_frac'], True):>6} {f(m['longest_hole_s']):>7}")


HEADER = f"{'':<10} {'real':>6} {'kept':>6} {'loop%':>6} {'dist':>6} {'cov_voc':>7} {'cov_dur':>6} {'spur%':>6} {'hole_s':>7}"


def compare(prefix: str, tracks: List[int], instrumental: List[int], providers: Dict[str, Path], near: float,
            masks: Dict[int, np.ndarray], per_track: bool) -> Dict:
    results: Dict[str, Dict[int, Dict]] = {p: {} for p in providers}
    for p, d in providers.items():
        for k in tracks:
            f = d / f"{str(k).zfill(2)}.json"
            if not f.exists():
                continue
            words, dur = load_words(f)
            results[p][k] = track_metrics(words, dur, masks.get(k), near)
            results[p][k]["duration_s"] = round(dur, 1)
    sung = [k for k in tracks if k not in instrumental]
    if per_track:
        for k in tracks:
            tag = " (instrumental)" if k in instrumental else ""
            print(f"\n--- track {str(k).zfill(2)}{tag}")
            print(HEADER)
            for p in providers:
                if k in results[p]:
                    print(fmt_row(p, results[p][k]))
    # totals over sung tracks
    print(f"\n=== totals over {len(sung)} sung tracks (coverage weighted by vocal seconds) ===")
    print(HEADER)
    totals: Dict[str, Dict] = {}
    for p in providers:
        rows = [results[p][k] for k in sung if k in results[p]]
        if not rows:
            continue
        real = sum(r["real_words"] for r in rows)
        kept = sum(r["kept_words"] for r in rows)
        loops = sum(r["loop_dropped"] for r in rows)
        vs = [(r["coverage_vocal"], r["vocal_s"]) for r in rows if r["coverage_vocal"] is not None]
        cov_v = sum(c * v for c, v in vs) / sum(v for _, v in vs) if vs else None
        dur = sum(r["duration_s"] for r in rows)
        cov_d = sum(r["coverage_dur"] * r["duration_s"] for r in rows) / dur if dur else 0.0
        sp = [(r["spurious_frac"], r["kept_words"]) for r in rows if r["spurious_frac"] is not None]
        spur = sum(s * n for s, n in sp) / max(1, sum(n for _, n in sp)) if sp else None
        # distinct ratio over the pooled vocabulary is not meaningful; report mean of per-track
        dist = float(np.mean([r["distinct_ratio"] for r in rows]))
        hole = max(r["longest_hole_s"] for r in rows)
        totals[p] = {"real_words": real, "kept_words": kept, "loop_frac": loops / real if real else 0.0,
                     "distinct_ratio": dist, "coverage_vocal": cov_v, "coverage_dur": cov_d,
                     "spurious_frac": spur, "longest_hole_s": hole,
                     "tracks": len(rows), "tracks_missing": len([k for k in sung if k not in results[p]]),
                     "empty_tracks": [k for k in sung if k in results[p] and results[p][k]["kept_words"] == 0],
                     "low_coverage_tracks": [k for k in sung if k in results[p] and results[p][k]["coverage_vocal"] is not None
                                             and results[p][k]["coverage_vocal"] < 0.6]}
        print(fmt_row(p, totals[p]))
    for p, t in totals.items():
        extra = []
        if t["empty_tracks"]:
            extra.append("empty: " + ",".join(str(k).zfill(2) for k in t["empty_tracks"]))
        if t["low_coverage_tracks"]:
            extra.append("cov_voc<60%: " + ",".join(str(k).zfill(2) for k in t["low_coverage_tracks"]))
        if t["tracks_missing"]:
            extra.append(f"missing {t['tracks_missing']} track files")
        if extra:
            print(f"  {p}: " + "; ".join(extra))
    return {"per_track": {p: {str(k).zfill(2): v for k, v in r.items()} for p, r in results.items()}, "totals": totals}


def fuse(tracks: List[int], base_dir: Path, other_dir: Path, out_dir: Path, gap: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_added_total = 0
    for k in tracks:
        nn = str(k).zfill(2)
        bf, of = base_dir / f"{nn}.json", other_dir / f"{nn}.json"
        if not bf.exists():
            print(f"  {nn}: no base transcript — skipped", file=sys.stderr)
            continue
        base_raw = json.loads(bf.read_text(encoding="utf-8"))
        base_words, dur = load_words(bf)
        kept, _, _ = clean_words(base_words)
        other_words: List[TranscriptionWord] = []
        if of.exists():
            ow, odur = load_words(of)
            other_words, _, _ = clean_words(ow)
            if not dur:
                dur = odur
        starts = sorted(w.start for w in kept)
        holes: List[Tuple[float, float]] = []
        edges = [0.0] + starts + [dur]
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a > gap:
                holes.append((a, b))
        added = [w for w in other_words if any(a < w.start < b for a, b in holes)]
        fused = sorted(kept + added, key=lambda w: (w.start, w.end))
        n_added_total += len(added)
        doc = {
            "duration": dur,
            "language": base_raw.get("language"),
            "text": " ".join(w.word for w in fused),
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in fused],
            "segments": None,
        }
        (out_dir / f"{nn}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {nn}: {len(kept)} base words + {len(added)} from {other_dir.name} in {len(holes)} hole(s) > {gap:.0f}s")
    print(f"fused set written to {out_dir}/ ({n_added_total} words added)")


def summarise_reports(reports: Dict[str, Path]) -> None:
    print(f"{'provider':<10} {'cov_raw':>8} {'cov_alnum':>9} {'cov_interp':>10} {'black%':>7} {'gaps>60s':>8} {'longest':>8} {'loops':>6} {'words':>6}  verdict")
    per_track_black: Dict[str, Dict[str, float]] = {}
    for p, f in reports.items():
        r = json.loads(f.read_text(encoding="utf-8"))
        verdict = "OK" if r.get("ok") else "REVIEW: " + "; ".join(r.get("review_reasons", []))
        print(f"{p:<10} {r['coverage_raw']:>8.1%} {r['coverage_raw_alnum']:>9.1%} {r['coverage_interp']:>10.1%} "
              f"{r['black_frac']:>7.1%} {r['gaps_over_threshold']:>8} {r['longest_gap_s'] / 60:>7.1f}m "
              f"{r.get('transcript_dropped', {}).get('loops', 0):>6} {r['transcript_words']:>6}  {verdict}")
        per_track_black[p] = {k: v["black_frac"] for k, v in r["per_track"].items() if not v.get("instrumental")}
    tracks = sorted({k for d in per_track_black.values() for k in d}, key=int)
    print("\nper-track black fraction (sung tracks):")
    print(f"{'track':<6} " + " ".join(f"{p:>10}" for p in reports))
    for k in tracks:
        print(f"{str(k).zfill(2):<6} " + " ".join(f"{per_track_black[p].get(k, float('nan')):>10.0%}" for p in reports))


def main() -> int:
    ap = argparse.ArgumentParser(description="ASR bake-off metrics for one opera")
    ap.add_argument("config")
    ap.add_argument("providers", nargs="*", help="tag=dir pairs (transcript dirs; or report files with --reports)")
    ap.add_argument("--near", type=float, default=5.0, help="a word covers ±this many seconds (default 5)")
    ap.add_argument("--per-track", action="store_true", help="print the per-track tables too")
    ap.add_argument("--json", help="write all metrics to this JSON file")
    ap.add_argument("--fuse", action="store_true", help="fuse the first provider with the second into --out")
    ap.add_argument("--out", help="output dir for --fuse")
    ap.add_argument("--gap", type=float, default=8.0, help="--fuse: fill holes longer than this (s)")
    ap.add_argument("--reports", action="store_true", help="providers are tag=alignment-report.json; tabulate them")
    ap.add_argument("--mask-cache", default=None, help="JSON cache for the per-track vocal masks")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    prefix = cfg["file_prefix"]
    start_idx = int(cfg.get("start_idx", 1))
    end_idx = int(cfg.get("end_idx") or (max(int(p.name) for p in Path("sep", f"{prefix}_sep", "htdemucs").glob("[0-9][0-9]")) + 1))
    tracks = list(range(start_idx, end_idx))
    instrumental = list(cfg.get("overture_indices") or [])
    inst_json = Path("sep") / f"{prefix}_sep" / "instrumental.json"
    if not instrumental and inst_json.exists():
        instrumental = json.loads(inst_json.read_text()).get("indices", [])

    providers: Dict[str, Path] = {}
    for spec in args.providers:
        tag, _, d = spec.partition("=")
        providers[tag] = Path(d)

    if args.reports:
        summarise_reports(providers)
        return 0
    if args.fuse:
        if len(providers) != 2 or not args.out:
            print("--fuse needs exactly two tag=dir providers and --out", file=sys.stderr)
            return 1
        (base, other) = list(providers.values())
        fuse(tracks, base, other, Path(args.out), args.gap)
        return 0
    if not providers:
        print("give at least one tag=dir provider", file=sys.stderr)
        return 1
    cache = Path(args.mask_cache) if args.mask_cache else Path("sep") / f"{prefix}_sep" / "vocal_mask_cache.json"
    masks = masks_for(prefix, tracks, cache)
    res = compare(prefix, tracks, instrumental, providers, args.near, masks, args.per_track)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
