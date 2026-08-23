#!/usr/bin/env python3
"""Transcribe every track of an opera with a quality gate and a fallback ASR.

    python transcribe_tracks.py configs/carmen.yaml              # transcribe what's missing, repair what's bad
    python transcribe_tracks.py configs/carmen.yaml --assess     # just grade the transcripts on disk
    python transcribe_tracks.py configs/carmen.yaml --fallback openai   # whisper-1 instead of Modal Whisper
    python transcribe_tracks.py configs/carmen.yaml --force      # redo everything from scratch

Per track:
  1. primary ASR = ElevenLabs Scribe on sep/<prefix>_sep/htdemucs/NN/vocals.m4a
     (an existing transcribed/<prefix>_transcribed/NN.json is reused as the primary);
  2. grade it: real-word count, hallucination loops (align.collapse_transcript_loops),
     and coverage — seconds of the track within COVER_RADIUS_S of a word, compared
     with the seconds of singing the demucs stem actually contains (`vocal_s` from
     sep/<prefix>_sep/instrumental.json, written by detect_instrumental.py);
  3. if it fails (empty / looped / holey), run the fallback ASR on that track and
     **fuse**: keep the primary's (de-looped) words and add the fallback's words only
     inside the primary's gaps (> GAP_S with a margin), so the two never overlap;
  4. write transcribed/<prefix>_transcribed/NN.json in the usual TranscriptionVerbose
     shape, keep every provider's raw output under .../providers/<provider>/NN.json
     (so re-runs never call an API twice), and a per-track quality.json.

Fallbacks: `modal` (faster-whisper on an L4 via transcribe_modal.py — credit-free,
~$0.07 per opera), `openai` (whisper-1; needs credits), `none`.
Exit 0 if every sung track passes the gate in the end, 2 if some are still bad
(the alignment tripwire will report them too) — the pipeline continues either way.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from align import collapse_transcript_loops, normalise_for_matching
from classes import TranscriptionWord
from config_parser import OperaConfig, parse_opera_config

load_dotenv()

# ---- gate thresholds ---------------------------------------------------------
COVER_RADIUS_S = 4.0       # a second is "covered" if a real word starts within this of it
MIN_COVERAGE = 0.70        # covered seconds / expected sung seconds below this => holey
MAX_LOOP_FRAC = 0.50       # share of words dropped as loops above this => looped
MIN_REAL_WORDS = 5         # fewer real words than this on a sung track => empty
MIN_VOCAL_S_TO_JUDGE = 20  # don't judge tracks with (almost) no singing
GAP_S = 8.0                # primary gaps longer than this get filled from the fallback
GAP_MARGIN_S = 1.0         # ... leaving this margin so the two never overlap

ELEVENLABS_LANGS = {"en": "eng", "it": "ita", "de": "deu", "fr": "fra", "es": "spa", "ru": "rus", "cs": "ces", "pt": "por"}


# ---- transcript shape ----------------------------------------------------------
def words_from_dict(d: dict) -> List[TranscriptionWord]:
    return [TranscriptionWord(start=float(w["start"]), end=float(w["end"]), word=str(w["word"])) for w in (d.get("words") or [])]


def transcript_dict(words: List[TranscriptionWord], duration: float, language: str, text: Optional[str] = None) -> dict:
    words = sorted(words, key=lambda w: (w.start, w.end))
    return {
        "duration": float(duration),
        "language": language,
        "text": text if text is not None else " ".join(w.word for w in words if w.word.strip()),
        "segments": None,
        "words": [{"start": w.start, "end": w.end, "word": w.word} for w in words],
    }


def is_real(word: str) -> bool:
    return any(c.isalnum() for c in word)


# ---- quality gate -----------------------------------------------------------
@dataclass
class Grade:
    verdict: str                 # ok | empty | looped | holey | instrumental | unknown
    real_words: int
    distinct_ratio: float
    loop_frac: float
    covered_s: float
    expected_s: Optional[float]  # vocal_s from the detector, or None
    coverage: Optional[float]
    gaps: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict in ("ok", "instrumental", "unknown")

    def short(self) -> str:
        cov = f"{self.coverage:.0%}" if self.coverage is not None else "n/a"
        return f"{self.verdict:12s} words={self.real_words:5d} loops={self.loop_frac:.0%} cover={cov}"


def gaps_in(words: List[TranscriptionWord], duration: float, gap_s: float = GAP_S) -> List[Tuple[float, float]]:
    starts = sorted(w.start for w in words if is_real(w.word))
    gaps, prev = [], 0.0
    for s in starts:
        if s - prev > gap_s:
            gaps.append((prev, s))
        prev = max(prev, s)
    if duration - prev > gap_s:
        gaps.append((prev, duration))
    return gaps


def grade(words: List[TranscriptionWord], duration: float, vocal_s: Optional[float], instrumental: bool = False) -> Grade:
    real = [w for w in words if is_real(w.word)]
    deloop, dropped = collapse_transcript_loops(words)
    loop_frac = dropped / len(real) if real else 0.0
    norm = [normalise_for_matching(w.word) for w in real]
    norm = [t for t in norm if t]
    distinct = len(set(norm)) / len(norm) if norm else 0.0
    # coverage of the track by (de-looped, real) words
    covered = set()
    for w in deloop:
        if is_real(w.word):
            lo, hi = int(max(0.0, w.start - COVER_RADIUS_S)), int(min(duration, w.start + COVER_RADIUS_S))
            covered.update(range(lo, hi + 1))
    covered_s = float(len(covered))
    expected = float(vocal_s) if vocal_s is not None else None
    coverage = (covered_s / expected) if expected else None
    gaps = gaps_in(deloop, duration)

    if instrumental:
        verdict = "instrumental"
    elif expected is not None and expected < MIN_VOCAL_S_TO_JUDGE:
        verdict = "ok"
    elif len(real) < MIN_REAL_WORDS:
        verdict = "empty"
    elif loop_frac > MAX_LOOP_FRAC:
        verdict = "looped"
    elif coverage is not None and coverage < MIN_COVERAGE:
        verdict = "holey"
    elif coverage is None and duration > 60 and covered_s / duration < MIN_COVERAGE * 0.8:
        verdict = "holey"   # no detector cache: judge against the whole track, leniently
    else:
        verdict = "ok"
    return Grade(verdict, len(real), distinct, loop_frac, covered_s, expected, coverage, gaps)


# ---- known ASR hallucinations ---------------------------------------------------
# Whisper's "credits" hallucinations on quiet/odd audio, by language. Matched on
# normalised word sequences; any word inside a match is dropped from fallback output.
HALLUCINATION_PHRASES = [
    "Sous-titrage", "Sous-titres", "Sous-titrage ST' 501", "Sous-titrage MFP", "Amara.org",
    "Merci d'avoir regardé", "Abonnez-vous", "Untertitel", "Untertitelung des ZDF", "Sottotitoli",
    "Sottotitoli e revisione", "Subtítulos", "Subtitles by", "Thanks for watching", "Thank you for watching",
    "Transcription par", "Traduction par",
]
# Matched on a flattened stream of normalised sub-tokens, so it doesn't matter whether an
# ASR emits "Sous-titrage" as one word or two ("Sous", "titrage").
_HALLUCINATION_TOKENS = [normalise_for_matching(ph).split() for ph in HALLUCINATION_PHRASES]


def strip_known_hallucinations(words: List[TranscriptionWord]) -> Tuple[List[TranscriptionWord], int]:
    flat: List[Tuple[int, str]] = []   # (word index, sub-token)
    for k, w in enumerate(words):
        for t in normalise_for_matching(w.word).split():
            flat.append((k, t))
    toks = [t for _, t in flat]
    drop_words = set()
    for p in _HALLUCINATION_TOKENS:
        n = len(p)
        if n == 0:
            continue
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == p:
                drop_words.update(flat[j][0] for j in range(i, i + n))
    kept = [w for k, w in enumerate(words) if k not in drop_words]
    return kept, len(drop_words)


# ---- fusion -------------------------------------------------------------------
def fuse(primary: List[TranscriptionWord], fallback: List[TranscriptionWord], duration: float) -> Tuple[List[TranscriptionWord], int]:
    """Primary (de-looped) words, plus fallback words strictly inside the primary's gaps."""
    base, _ = collapse_transcript_loops(primary)
    gaps = gaps_in(base, duration)
    added = []
    for g0, g1 in gaps:
        lo, hi = g0 + GAP_MARGIN_S, g1 - GAP_MARGIN_S
        if hi <= lo:
            continue
        added.extend(w for w in fallback if lo <= w.start <= hi and w.end <= hi + GAP_MARGIN_S)
    merged = sorted(list(base) + added, key=lambda w: (w.start, w.end))
    return merged, len(added)


# ---- providers ----------------------------------------------------------------
def audio_duration(path: str) -> float:
    import librosa
    return float(librosa.get_duration(path=path))


def run_elevenlabs(path: str, language: str, max_retries: int = 3) -> Optional[dict]:
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    lang = ELEVENLABS_LANGS.get(language)
    duration = audio_duration(path)
    data = Path(path).read_bytes()
    last_err = None
    for attempt in range(max_retries):
        try:
            tr = client.speech_to_text.convert(file=data, model_id="scribe_v1", language_code=lang,
                                               tag_audio_events=False, diarize=False)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
            continue
        words = [TranscriptionWord(start=w.start, end=w.end, word=w.text)
                 for w in (tr.words or []) if getattr(w, "type", "word") == "word"]
        # Scribe returns words for every sung track it understands; an empty result is a
        # failure we want the fallback for, not a retry loop.
        return transcript_dict(words, duration, language, text=tr.text or "")
    print(f"    ElevenLabs failed after {max_retries} attempts: {last_err}", file=sys.stderr)
    return None


def run_openai(path: str, language: str) -> Optional[dict]:
    from openai import OpenAI
    client = OpenAI()
    duration = audio_duration(path)
    try:
        with open(path, "rb") as f:
            tr = client.audio.transcriptions.create(file=f, model="whisper-1", language=language,
                                                    response_format="verbose_json", timestamp_granularities=["word"])
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "insufficient_quota" in msg or "credit_balance_exhausted" in msg:
            print("    OpenAI: the key has no credits (https://platform.openai.com/settings/organization/billing/)", file=sys.stderr)
            return None
        raise
    words = [TranscriptionWord(start=w.start, end=w.end, word=w.word) for w in (tr.words or [])]
    return transcript_dict(words, duration, language, text=tr.text or "")


def run_modal_batch(paths: List[str], language: str, model: Optional[str]) -> List[Optional[dict]]:
    try:
        from transcribe_modal import transcribe_paths_modal
    except ImportError as e:
        print(f"    Modal Whisper unavailable ({e}); install modal / check transcribe_modal.py", file=sys.stderr)
        return [None] * len(paths)
    kwargs = {"model": model} if model else {}
    results = transcribe_paths_modal(paths, language, **kwargs)
    out = []
    for p, r in zip(paths, results):
        if r is None:
            out.append(None)
            continue
        words = words_from_dict(r)
        out.append(transcript_dict(words, float(r.get("duration") or audio_duration(p)), language, text=r.get("text")))
    return out


# ---- driver -------------------------------------------------------------------
def stem_path(config: OperaConfig, idx: int) -> Optional[str]:
    for ext in ("m4a", "wav"):
        p = f"{config.sep_dir}/htdemucs/{idx:02d}/vocals.{ext}"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def load_detector(config: OperaConfig) -> dict:
    p = f"{config.sep_dir}/instrumental.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe all tracks with a quality gate and fallback ASR")
    ap.add_argument("config")
    ap.add_argument("--assess", action="store_true", help="only grade the transcripts on disk")
    ap.add_argument("--fallback", choices=["modal", "openai", "none"], default=os.getenv("KUNSTWERK_ASR_FALLBACK", "modal"))
    ap.add_argument("--model", default=os.getenv("KUNSTWERK_WHISPER_MODEL"), help="Whisper model for the Modal fallback (transcribe_modal.py's default if unset)")
    ap.add_argument("--force", action="store_true", help="ignore existing final transcripts and provider outputs")
    ap.add_argument("--no-fuse", action="store_true", help="replace a failing primary with the fallback instead of fusing")
    args = ap.parse_args()

    config = parse_opera_config(args.config)
    out_dir = Path(config.transcribed_dir)
    prov_dir = out_dir / "providers"
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = load_detector(config)
    det_tracks = detector.get("tracks", {})
    instrumental = set(int(i) for i in detector.get("indices", []))
    if not det_tracks:
        print("note: no sep/<prefix>_sep/instrumental.json — grading against track duration instead of detected singing")

    def provider_path(name: str, idx: int) -> Path:
        return prov_dir / name / f"{idx:02d}.json"

    def save(path: Path, d: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    def load(path: Path) -> Optional[dict]:
        if path.exists() and not args.force:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    tracks = list(range(config.start_idx, config.end_idx))
    quality: Dict[str, dict] = {}
    primaries: Dict[int, dict] = {}
    grades: Dict[int, Grade] = {}

    # ---- 1. primary -------------------------------------------------------------
    print(f"=== {config.title}: {len(tracks)} tracks, fallback={args.fallback} ===")
    for idx in tracks:
        final_path = out_dir / f"{idx:02d}.json"
        d = load(final_path) or load(provider_path("elevenlabs", idx))
        src = "existing" if d else None
        if d is None and not args.assess:
            stem = stem_path(config, idx)
            if stem is None:
                print(f"  {idx:02d}: no vocals stem under {config.sep_dir}/htdemucs/{idx:02d}/ — run separate.sh first", file=sys.stderr)
                continue
            print(f"  {idx:02d}: ElevenLabs ...", end="", flush=True)
            d = run_elevenlabs(stem, config.language)
            print(" done" if d else " FAILED")
            if d:
                save(provider_path("elevenlabs", idx), d)
                src = "elevenlabs"
        if d is None:
            continue
        primaries[idx] = d
        vocal_s = det_tracks.get(f"{idx:02d}", {}).get("vocal_s") if det_tracks else None
        g = grade(words_from_dict(d), float(d.get("duration") or 0), vocal_s, instrumental=idx in instrumental)
        grades[idx] = g
        quality[f"{idx:02d}"] = {"primary": src, "primary_grade": g.__dict__ | {"gaps": [[round(a, 1), round(b, 1)] for a, b in g.gaps]}}

    # ---- 2. fallback + fusion for failing tracks ---------------------------------
    failing = [i for i, g in grades.items() if not g.ok]
    print(f"\nprimary grades: {sum(g.ok for g in grades.values())}/{len(grades)} ok; failing: {[f'{i:02d}' for i in failing]}")
    for idx in tracks:
        if idx in grades:
            print(f"  {idx:02d}: {grades[idx].short()}")

    if failing and not args.assess and args.fallback != "none":
        need = [i for i in failing if load(provider_path(f"{args.fallback}", i)) is None]
        results: Dict[int, Optional[dict]] = {i: load(provider_path(args.fallback, i)) for i in failing}
        if need:
            print(f"\n=== fallback {args.fallback} on {[f'{i:02d}' for i in need]} ===")
            if args.fallback == "modal":
                paths = [stem_path(config, i) for i in need]
                if any(p is None for p in paths):
                    print("  some stems are missing; skipping those", file=sys.stderr)
                ok_idx = [i for i, p in zip(need, paths) if p]
                outs = run_modal_batch([stem_path(config, i) for i in ok_idx], config.language, args.model)
                for i, r in zip(ok_idx, outs):
                    results[i] = r
            else:
                for i in need:
                    print(f"  {i:02d}: whisper-1 ...", end="", flush=True)
                    r = run_openai(stem_path(config, i), config.language)
                    print(" done" if r else " FAILED")
                    results[i] = r
            for i, r in results.items():
                if r is not None and i in need:
                    save(provider_path(args.fallback, i), r)

        print("\n=== fusion ===")
        for idx in failing:
            r = results.get(idx)
            d = primaries[idx]
            vocal_s = det_tracks.get(f"{idx:02d}", {}).get("vocal_s") if det_tracks else None
            if r is None:
                quality[f"{idx:02d}"]["fallback"] = {"provider": args.fallback, "result": "unavailable"}
                print(f"  {idx:02d}: fallback unavailable; keeping primary ({grades[idx].verdict})")
                continue
            fb_words, n_halluc = strip_known_hallucinations(words_from_dict(r))
            duration = float(d.get("duration") or r.get("duration") or 0)
            fb_grade = grade(fb_words, duration, vocal_s)
            if args.no_fuse:
                fused, added = fb_words, len(fb_words)
            else:
                fused, added = fuse(words_from_dict(d), fb_words, duration)
            fused_grade = grade(fused, duration, vocal_s)
            # keep whichever of fused / fallback-alone grades better (fusion can't hurt coverage,
            # but if the primary was pure garbage the fallback alone is cleaner)
            choice = "fused"
            if grades[idx].verdict in ("empty", "looped") and (fb_grade.coverage or 0) >= (fused_grade.coverage or 0) - 1e-9:
                fused, fused_grade, choice = fb_words, fb_grade, "fallback"
            primaries[idx] = transcript_dict(fused, duration, config.language)
            grades[idx] = fused_grade
            quality[f"{idx:02d}"]["fallback"] = {
                "provider": args.fallback, "grade": fb_grade.__dict__ | {"gaps": len(fb_grade.gaps)},
                "choice": choice, "words_added": added, "hallucinated_words_dropped": n_halluc,
                "final_grade": fused_grade.__dict__ | {"gaps": len(fused_grade.gaps)},
            }
            print(f"  {idx:02d}: {choice:8s} fallback {fb_grade.short()} -> final {fused_grade.short()}")

    # ---- 3. write finals + quality report -----------------------------------------
    if not args.assess:
        for idx, d in primaries.items():
            save(out_dir / f"{idx:02d}.json", d)
    still_bad = [f"{i:02d}" for i, g in grades.items() if not g.ok]
    summary = {
        "tracks": len(tracks), "graded": len(grades), "ok": len(grades) - len(still_bad), "still_bad": still_bad,
        "thresholds": {"min_coverage": MIN_COVERAGE, "max_loop_frac": MAX_LOOP_FRAC, "min_real_words": MIN_REAL_WORDS,
                       "cover_radius_s": COVER_RADIUS_S, "gap_s": GAP_S},
        "fallback": args.fallback,
    }
    with open(out_dir / "quality.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "tracks": quality}, f, indent=2, ensure_ascii=False)
    print(f"\n{summary['ok']}/{summary['graded']} tracks pass the gate"
          + (f"; still bad: {still_bad}" if still_bad else "") + f"  (details: {out_dir}/quality.json)")
    return 2 if still_bad else 0


if __name__ == "__main__":
    sys.exit(main())
