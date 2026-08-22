#!/usr/bin/env python
"""Detect purely instrumental tracks (preludes, entr'actes, intermezzi) automatically.

Transcribers hallucinate words on instrumental audio, which wrecks the
libretto alignment in make_video.py, so those tracks' transcripts must be
blanked.  Historically that was the hand-entered ``overture_indices`` list in
configs/<opera>.yaml; this module derives the same list from the audio.

Method: for every track, compare the demucs ``vocals`` stem against the
original mix, frame by frame (FRAME_S-second frames).  A frame "has vocals"
when the vocals stem carries at least VOCAL_RATIO_THRESHOLD of the mix's RMS
in that frame (i.e. the stem is within ~-14 dB of the mix).  ``vocal_frac`` is
the fraction of non-silent frames that have vocals; a track is instrumental
when vocal_frac < INSTRUMENTAL_VOCAL_FRAC.

Calibration (Carmen, demucs htdemucs two-stems, 2026-08):
    instrumental  01 prelude 0.015, 13 entr'acte 0.015, 23 intermezzo 0.000,
                  30 aragonaise 0.000          (whole-track vocals/mix: -24..-53 dB)
    sung          03 boys' chorus 0.53, 31 chorus 0.62, 32 march+chorus 0.72,
                  20 flower song 0.79, 27 Micaela aria 0.82, 05 habanera 0.98,
                  10 spoken dialogue 1.00      (whole-track vocals/mix: -4.5..0 dB)
    => threshold 0.10 sits ~7x above the worst instrumental and ~5x below the
    quietest sung track; the gap is stable across frame length (0.1-2 s),
    ratio threshold (0.05-0.5) and silence floor (-70..-40 dBFS).
    Transcript word density is NOT a usable substitute: ElevenLabs
    hallucinated 228 words (127 wpm) on entr'acte 13 and only 34 on sung
    chorus 31.

Usage:
    python detect_instrumental.py configs/carmen.yaml
        -> writes sep/<file_prefix>_sep/instrumental.json and prints indices
    python detect_instrumental.py --prefix carmen      (skip the yaml)

Importable:
    detect_instrumental_tracks(file_prefix) -> list[int]   (computes + writes cache)
    load_instrumental_indices(file_prefix)  -> list[int] | None  (reads cache)

Requires ``ffmpeg`` on PATH (already a pipeline dependency); reads m4a or wav.
Always run from the repo root (relative audio/ and sep/ paths, like every
other stage script).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import yaml

# ---- tunables ---------------------------------------------------------------
SR = 16000                      # decode sample rate (plenty for an energy measure)
FRAME_S = 0.5                   # analysis frame length, seconds
SILENCE_DB = -50.0              # mix frames quieter than this (dBFS RMS) are ignored
VOCAL_RATIO_THRESHOLD = 0.2     # vocals_rms / mix_rms above this => frame has vocals (~ -14 dB)
INSTRUMENTAL_VOCAL_FRAC = 0.10  # track is instrumental if vocal_frac < this ...
MAX_INSTRUMENTAL_VOCAL_S = 30.0 # ... and it has less than this many seconds of detected vocals
                                # (guards mixed tracks: a long prelude + a short sung passage must not be blanked)
# ----------------------------------------------------------------------------


def _decode(path: str, sr: int = SR) -> np.ndarray:
    """Decode any ffmpeg-readable audio file to mono float32 at ``sr``."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def _frame_rms(y: np.ndarray, frame_len: int) -> np.ndarray:
    n = len(y) // frame_len
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = y[: n * frame_len].reshape(n, frame_len).astype(np.float64)
    return np.sqrt(np.mean(frames**2, axis=1))


def _to_db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-12)))


def compute_track_metrics(mix_path: str, vocals_path: str) -> dict:
    """Compare the vocals stem against the mix and return per-track metrics."""
    mix = _decode(mix_path)
    voc = _decode(vocals_path)
    n = min(len(mix), len(voc))
    mix, voc = mix[:n], voc[:n]

    frame_len = int(SR * FRAME_S)
    mix_rms = _frame_rms(mix, frame_len)
    voc_rms = _frame_rms(voc, frame_len)

    silence_lin = 10 ** (SILENCE_DB / 20.0)
    active = mix_rms > silence_lin
    n_active = int(active.sum())

    if n_active == 0:
        vocal_frac = 0.0
        ratio_frames = np.zeros(0)
    else:
        ratio_frames = voc_rms[active] / mix_rms[active]
        vocal_frac = float(np.mean(ratio_frames > VOCAL_RATIO_THRESHOLD))

    mix_total_rms = float(np.sqrt(np.mean(mix.astype(np.float64) ** 2))) if n else 0.0
    voc_total_rms = float(np.sqrt(np.mean(voc.astype(np.float64) ** 2))) if n else 0.0
    rms_ratio = voc_total_rms / mix_total_rms if mix_total_rms > 0 else 0.0

    return {
        "duration_s": round(n / SR, 2),
        "active_s": round(n_active * FRAME_S, 2),
        "vocal_frac": round(vocal_frac, 4),
        "vocal_s": round(vocal_frac * n_active * FRAME_S, 2),
        "rms_ratio": round(rms_ratio, 4),
        "rms_ratio_db": round(_to_db(rms_ratio), 2),
        "ratio_p50": round(float(np.median(ratio_frames)), 4) if len(ratio_frames) else 0.0,
        "ratio_p90": round(float(np.percentile(ratio_frames, 90)), 4) if len(ratio_frames) else 0.0,
        "instrumental": bool(vocal_frac < INSTRUMENTAL_VOCAL_FRAC
                             and vocal_frac * n_active * FRAME_S < MAX_INSTRUMENTAL_VOCAL_S),
    }


def _audio_dir(file_prefix: str) -> str:
    return os.path.join("audio", file_prefix)


def _sep_dir(file_prefix: str) -> str:
    return os.path.join("sep", f"{file_prefix}_sep")


def cache_path(file_prefix: str) -> str:
    return os.path.join(_sep_dir(file_prefix), "instrumental.json")


def _find_vocals(file_prefix: str, track: str) -> str | None:
    base = os.path.join(_sep_dir(file_prefix), "htdemucs", track)
    for ext in ("m4a", "wav"):
        p = os.path.join(base, f"vocals.{ext}")
        if os.path.exists(p):
            return p
    return None


def iter_track_pairs(file_prefix: str):
    """Yield (track_id, mix_path, vocals_path) for every track that has both files."""
    adir = _audio_dir(file_prefix)
    if not os.path.isdir(adir):
        return
    exts = (".m4a", ".wav", ".mp3", ".flac", ".opus", ".webm")
    mixes: dict[str, str] = {}
    for fname in sorted(os.listdir(adir)):
        stem, ext = os.path.splitext(fname)
        if not stem.isdigit() or ext.lower() not in exts:
            continue
        # one file per track; prefer the earliest extension in `exts`
        if stem not in mixes or exts.index(ext.lower()) < exts.index(os.path.splitext(mixes[stem])[1].lower()):
            mixes[stem] = fname
    for stem in sorted(mixes):
        vocals = _find_vocals(file_prefix, stem)
        if vocals is None:
            continue
        yield stem, os.path.join(adir, mixes[stem]), vocals


def detect_instrumental_tracks(file_prefix: str, verbose: bool = True) -> list[int]:
    """Compute metrics for every separated track, write the cache, return 1-based indices."""
    tracks: dict[str, dict] = {}
    for track, mix_path, vocals_path in iter_track_pairs(file_prefix):
        m = compute_track_metrics(mix_path, vocals_path)
        tracks[track] = m
        if verbose:
            flag = "INSTRUMENTAL" if m["instrumental"] else ""
            print(
                f"{track}: vocal_frac={m['vocal_frac']:.3f} vocal_s={m['vocal_s']:6.1f}/"
                f"{m['active_s']:6.1f}  rms_ratio={m['rms_ratio_db']:6.1f} dB  {flag}",
                file=sys.stderr,
            )
    indices = sorted(int(t) for t, m in tracks.items() if m["instrumental"])
    payload = {
        "indices": indices,
        "params": {
            "frame_s": FRAME_S,
            "silence_db": SILENCE_DB,
            "vocal_ratio_threshold": VOCAL_RATIO_THRESHOLD,
            "instrumental_vocal_frac": INSTRUMENTAL_VOCAL_FRAC,
            "max_instrumental_vocal_s": MAX_INSTRUMENTAL_VOCAL_S,
        },
        "tracks": tracks,
    }
    os.makedirs(_sep_dir(file_prefix), exist_ok=True)
    with open(cache_path(file_prefix), "w") as f:
        json.dump(payload, f, indent=2)
    return indices


def load_instrumental_indices(file_prefix: str) -> list[int] | None:
    """Return cached 1-based instrumental indices, or None if no cache exists."""
    p = cache_path(file_prefix)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        data = json.load(f)
    return [int(i) for i in data.get("indices", [])]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Detect purely instrumental tracks by comparing the demucs vocals stem "
        "against the original mix; writes sep/<file_prefix>_sep/instrumental.json."
    )
    ap.add_argument("config", nargs="?", help="configs/<opera>.yaml (provides file_prefix)")
    ap.add_argument("--prefix", help="file_prefix to use instead of reading a yaml")
    ap.add_argument("-q", "--quiet", action="store_true", help="don't print per-track metrics")
    args = ap.parse_args(argv)

    if args.prefix:
        file_prefix = args.prefix
    elif args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        file_prefix = cfg["file_prefix"]
    else:
        ap.error("need a config.yaml or --prefix")

    indices = detect_instrumental_tracks(file_prefix, verbose=not args.quiet)
    print(f"instrumental indices for {file_prefix}: {indices}")
    print(f"wrote {cache_path(file_prefix)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
