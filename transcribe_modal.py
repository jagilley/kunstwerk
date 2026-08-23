#!/usr/bin/env python3
"""Whisper transcription of vocal stems on Modal GPUs (faster-whisper / CTranslate2).

    python transcribe_modal.py configs/carmen.yaml --out-dir transcribed/carmen_whisperv2_transcribed
    python transcribe_modal.py configs/carmen.yaml --out-dir ... --model large-v3 --tracks 03,24
    python transcribe_modal.py configs/carmen.yaml --out-dir ... --no-vad --beam-size 1

Reads sep/<prefix>_sep/htdemucs/NN/vocals.{m4a,wav} (falls back to audio/<prefix>/NN.m4a
when no stem exists), transcodes each to 16 kHz mono FLAC locally, sends all tracks of
the opera up in one Modal batch (one `.map` over an L4 class that loads the model once
per container) and writes transcribed/<out-dir>/NN.json in the pipeline's transcript
shape (OpenAI `TranscriptionVerbose`: duration, language, text, words[{word,start,end}],
segments: null). `duration` is the decoded audio length, not the last word. Word strings
are stripped of leading/trailing punctuation (like whisper-1's word timestamps; ElevenLabs
emits punctuation as separate tokens that align.py drops anyway).

A sidecar `<out-dir>/_asr_meta.json` records model, decoding options, per-segment
avg_logprob / no_speech_prob / compression_ratio and GPU seconds per track — provenance
and raw material for a quality gate; nothing downstream reads it.

Importable: `transcribe_paths_modal(paths, language, model="large-v2") -> list[dict]` does
the same for an arbitrary list of local audio paths in one Modal batch (same order).

Nothing is stored on Modal: audio goes up as function arguments, transcripts come back as
return values. Model weights for BAKED_MODELS are baked into the image; any other
faster-whisper model name (e.g. large-v3-turbo, medium) is fetched from Hugging Face on
container start. The Modal workspace is `chromatic` (MODAL_PROFILE is set to it unless
KUNSTWERK_MODAL_PROFILE says otherwise). Cost: a 2 h opera is ~5 GPU-minutes on an L4.

Exit codes: 0 ok; 1 failure; 3 Modal unavailable (not installed / no token).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Must be set before `import modal` — the client reads the profile from the environment.
os.environ.setdefault("MODAL_PROFILE", os.getenv("KUNSTWERK_MODAL_PROFILE", "chromatic"))

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

import yaml

APP_NAME = "kunstwerk-whisper"
GPU = os.getenv("KUNSTWERK_MODAL_GPU", "L4")
MAX_CONTAINERS = int(os.getenv("KUNSTWERK_MODAL_MAX_CONTAINERS", "6"))
DEFAULT_MODEL = "large-v2"
BAKED_MODELS = ("large-v2", "large-v3")   # weights baked into the image; others download on cold start
SR = 16000                                # whisper's sample rate; audio is transcoded to this before upload
COMPUTE_TYPE = "float16"

# Decoding defaults, tuned for sung material on demucs vocal stems:
#  - condition_on_previous_text=False: each 30 s window is decoded on its own, so a
#    hallucination loop ("la la la ...") cannot propagate past one window — the classic
#    cure for Whisper locking up on melismas.
#  - a short temperature ladder: the default (0..1.0) fallback on failed compression-ratio
#    / logprob checks tends to produce confident nonsense on singing; 0.4 is plenty.
#  - vad_filter=True: Silero VAD drops silences > 1 s before decoding, which is where
#    Whisper invents text on the near-silent instrumental stretches of a vocals stem. A
#    low threshold + generous padding so quiet singing is not eaten.
#  - hallucination_silence_threshold: with word timestamps, drop words that sit after
#    > 2 s of silence when the segment looks hallucinated.
DEFAULT_OPTIONS: Dict = {
    "beam_size": 5,
    "best_of": 5,
    "temperature": [0.0, 0.2, 0.4],
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    # Silero VAD is trained on speech and throws away sung passages: on Carmen it cut a
    # chorus track from 109 words to 26 and a sextet to 0. Off by default; --vad to enable.
    "vad_filter": False,
    "vad_parameters": {"threshold": 0.3, "min_silence_duration_ms": 1000, "speech_pad_ms": 400},
    "hallucination_silence_threshold": 2.0,
    "initial_prompt": None,
}

if modal is not None:
    image = (
        # cuDNN 9 + CUDA 12 runtime, which ctranslate2 >= 4.5 needs; python added by Modal.
        modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
        .apt_install("ffmpeg")
        .pip_install("faster-whisper==1.1.1", "huggingface_hub>=0.26", "requests")
        # Bake the weights into the image so containers don't fetch ~3 GB on cold start.
        .run_commands(*[
            f"python -c \"from faster_whisper.utils import download_model; download_model('{m}')\""
            for m in BAKED_MODELS
        ])
    )
    app = modal.App(APP_NAME)

    @app.cls(image=image, gpu=GPU, timeout=60 * 60, retries=1, max_containers=MAX_CONTAINERS, scaledown_window=60)
    class Transcriber:
        model_name: str = modal.parameter(default=DEFAULT_MODEL)

        @modal.enter()
        def load(self):
            from faster_whisper import WhisperModel

            t0 = time.time()
            self.model = WhisperModel(self.model_name, device="cuda", compute_type=COMPUTE_TYPE)
            self.load_s = time.time() - t0

        @modal.method()
        def transcribe(self, name: str, audio: bytes, language: str, options: dict) -> dict:
            """Audio bytes (any ffmpeg-readable container; we send 16 kHz mono FLAC) ->
            {"transcript": <our JSON shape>, "meta": {...}}."""
            import io

            from faster_whisper import decode_audio

            t0 = time.time()
            samples = decode_audio(io.BytesIO(audio), sampling_rate=SR)
            duration = float(len(samples)) / SR
            opts = dict(options)
            if opts.get("initial_prompt") is None:
                opts.pop("initial_prompt", None)
            if isinstance(opts.get("temperature"), list):
                opts["temperature"] = tuple(opts["temperature"])
            segments_iter, info = self.model.transcribe(
                samples, language=language, task="transcribe", word_timestamps=True, **opts)
            words: List[dict] = []
            texts: List[str] = []
            seg_meta: List[dict] = []
            for seg in segments_iter:
                texts.append(seg.text.strip())
                seg_meta.append({
                    "start": round(seg.start, 2), "end": round(seg.end, 2),
                    "avg_logprob": round(seg.avg_logprob, 3),
                    "no_speech_prob": round(seg.no_speech_prob, 3),
                    "compression_ratio": round(seg.compression_ratio, 3),
                    "temperature": seg.temperature,
                    "n_words": len(seg.words or []),
                })
                for w in seg.words or []:
                    token = _clean_word(w.word)
                    if not token:
                        continue
                    words.append({"word": token, "start": round(float(w.start), 3), "end": round(float(w.end), 3)})
            transcript = {
                "duration": duration,
                "language": language,
                "text": " ".join(t for t in texts if t),
                "words": words,
                "segments": None,
            }
            meta = {
                "name": name,
                "model": self.model_name,
                "gpu_s": round(time.time() - t0, 2),
                "model_load_s": round(self.load_s, 1),
                "audio_s": round(duration, 2),
                "vad_speech_s": round(info.duration_after_vad, 2) if getattr(info, "duration_after_vad", None) is not None else None,
                "language_probability": round(float(info.language_probability), 3) if info.language_probability is not None else None,
                "n_segments": len(seg_meta),
                "n_words": len(words),
                "segments": seg_meta,
            }
            return {"transcript": transcript, "meta": meta}


_PUNCT_EDGE = re.compile(r"^\W+|\W+$", re.UNICODE)


def _clean_word(w: str) -> str:
    """Strip whitespace and leading/trailing punctuation; keep internal apostrophes/hyphens."""
    return _PUNCT_EDGE.sub("", w.strip())


def audio_payload(path: str) -> bytes:
    """Transcode any audio file to 16 kHz mono 16-bit FLAC bytes (what whisper wants,
    ~1/10 the size of a 44.1 kHz stereo wav; lossless at that rate)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
           "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", "-f", "flac", "-"]
    res = subprocess.run(cmd, capture_output=True, check=True)
    return res.stdout


def _modal_ready() -> Optional[str]:
    """None if Modal is importable and has a token for the profile, else an error string."""
    if modal is None:
        return "modal is not installed in this environment (pip install modal)"
    try:
        modal.config._check_config()  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        return f"Modal is not configured for profile {os.environ['MODAL_PROFILE']}: {e}"
    return None


def merged_options(overrides: Optional[dict] = None) -> dict:
    opts = json.loads(json.dumps(DEFAULT_OPTIONS))  # deep copy, JSON-safe
    for k, v in (overrides or {}).items():
        if k == "vad_parameters" and isinstance(v, dict):
            opts["vad_parameters"].update(v)
        else:
            opts[k] = v
    return opts


def transcribe_paths_modal(
    paths: Sequence[str],
    language: str,
    model: str = DEFAULT_MODEL,
    options: Optional[dict] = None,
    names: Optional[Sequence[str]] = None,
    with_meta: bool = False,
    verbose: bool = True,
):
    """Transcribe local audio files in one Modal batch.

    Returns a list of transcript dicts (our JSON shape) in the order of `paths`.
    With `with_meta=True` returns (transcripts, metas). `options` overrides
    DEFAULT_OPTIONS (faster-whisper `transcribe()` kwargs). Raises RuntimeError
    if any track fails after Modal's retry.
    """
    err = _modal_ready()
    if err:
        raise RuntimeError(err)
    paths = [str(p) for p in paths]
    names = list(names) if names is not None else [Path(p).parent.name + "/" + Path(p).name for p in paths]
    assert len(names) == len(paths)
    opts = merged_options(options)

    t0 = time.time()
    payloads = [audio_payload(p) for p in paths]
    if verbose:
        mb = sum(len(b) for b in payloads) / 1e6
        print(f"transcribing {len(paths)} file(s) on Modal ({GPU}, {model}, profile {os.environ['MODAL_PROFILE']}); "
              f"{mb:.0f} MB of 16 kHz FLAC prepared in {time.time() - t0:.0f}s", flush=True)

    transcripts: List[Optional[dict]] = [None] * len(paths)
    metas: List[Optional[dict]] = [None] * len(paths)
    failures: List[Tuple[str, Exception]] = []
    t1 = time.time()
    with modal.enable_output():
        with app.run():
            tr = Transcriber(model_name=model)
            results = tr.transcribe.map(names, payloads, kwargs={"language": language, "options": opts},
                                        order_outputs=True, return_exceptions=True)
            for i, (name, result) in enumerate(zip(names, results)):
                if isinstance(result, Exception):
                    failures.append((name, result))
                    print(f"  {name}: FAILED {result!r}", file=sys.stderr, flush=True)
                    continue
                transcripts[i] = result["transcript"]
                metas[i] = result["meta"]
                if verbose:
                    m = result["meta"]
                    print(f"  {name}: {m['n_words']} words, {m['audio_s']:.0f}s audio in {m['gpu_s']:.1f}s GPU "
                          f"[{time.time() - t1:.0f}s wall]", flush=True)
    if verbose:
        gpu = sum(m["gpu_s"] for m in metas if m)
        aud = sum(m["audio_s"] for m in metas if m)
        print(f"done: {len(paths) - len(failures)}/{len(paths)} files, {aud / 60:.1f} min audio, "
              f"{gpu:.0f}s GPU decode ({aud / gpu:.0f}x realtime) in {time.time() - t1:.0f}s wall", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} file(s) failed on Modal: " + ", ".join(n for n, _ in failures))
    if with_meta:
        return transcripts, metas
    return transcripts


# ---- CLI --------------------------------------------------------------------

def stem_path(prefix: str, idx: int) -> Optional[Path]:
    """sep/<prefix>_sep/htdemucs/NN/vocals.{m4a,wav}, else audio/<prefix>/NN.m4a, else None."""
    nn = str(idx).zfill(2)
    for cand in (Path("sep") / f"{prefix}_sep" / "htdemucs" / nn / "vocals.m4a",
                 Path("sep") / f"{prefix}_sep" / "htdemucs" / nn / "vocals.wav",
                 Path("audio") / prefix / f"{nn}.m4a"):
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None


def _parse_tracks(spec: Optional[str]) -> Optional[List[int]]:
    if not spec:
        return None
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="Whisper (faster-whisper) transcription on Modal")
    ap.add_argument("config", help="configs/<opera>.yaml")
    ap.add_argument("--out-dir", required=True, help="e.g. transcribed/<prefix>_whisperv2_transcribed")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"faster-whisper model name (default {DEFAULT_MODEL}; "
                                                           f"baked: {', '.join(BAKED_MODELS)})")
    ap.add_argument("--tracks", help="comma list / ranges of 1-based track numbers, e.g. 03,24 or 5-9 (default: all)")
    ap.add_argument("--force", action="store_true", help="re-transcribe tracks whose JSON already exists in --out-dir")
    ap.add_argument("--dry-run", action="store_true", help="list what would run and exit")
    ap.add_argument("--beam-size", type=int)
    ap.add_argument("--no-vad", action="store_true", help="disable the Silero VAD pre-filter")
    ap.add_argument("--condition-on-previous-text", action="store_true",
                    help="whisper default (off here: off is the loop cure)")
    ap.add_argument("--temperature", help="comma list, e.g. 0 or 0,0.2,0.4")
    ap.add_argument("--initial-prompt")
    ap.add_argument("--option", action="append", default=[],
                    help="extra faster-whisper transcribe() kwarg as key=json, e.g. --option no_speech_threshold=0.5")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    prefix = cfg.get("file_prefix")
    language = cfg.get("language")
    if not prefix or not language:
        print("config needs file_prefix and language", file=sys.stderr)
        return 1
    start_idx = int(cfg.get("start_idx", 1))
    end_idx = cfg.get("end_idx")
    if end_idx is None:
        stems = sorted(Path("sep", f"{prefix}_sep", "htdemucs").glob("[0-9][0-9]"))
        end_idx = (max(int(p.name) for p in stems) + 1) if stems else start_idx
    wanted = _parse_tracks(args.tracks) or list(range(start_idx, int(end_idx)))

    overrides: Dict = {}
    if args.beam_size:
        overrides["beam_size"] = args.beam_size
    if args.no_vad:
        overrides["vad_filter"] = False
    if args.condition_on_previous_text:
        overrides["condition_on_previous_text"] = True
    if args.temperature:
        overrides["temperature"] = [float(x) for x in args.temperature.split(",")]
    if args.initial_prompt:
        overrides["initial_prompt"] = args.initial_prompt
    for kv in args.option:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v)

    out_dir = Path(args.out_dir)
    todo: List[Tuple[int, Path]] = []
    missing: List[int] = []
    for idx in wanted:
        out = out_dir / f"{str(idx).zfill(2)}.json"
        if out.exists() and out.stat().st_size > 0 and not args.force:
            continue
        p = stem_path(prefix, idx)
        if p is None:
            missing.append(idx)
            continue
        todo.append((idx, p))
    if missing:
        print(f"no audio for track(s) {' '.join(str(i).zfill(2) for i in missing)} — skipped", file=sys.stderr)
    if not todo:
        print(f"nothing to do: every requested track already has a JSON in {out_dir}")
        return 0
    print(f"{len(todo)} track(s) -> {out_dir} with {args.model}: " + " ".join(str(i).zfill(2) for i, _ in todo))
    if args.dry_run:
        print("options:", json.dumps(merged_options(overrides)))
        return 0

    err = _modal_ready()
    if err:
        print(err, file=sys.stderr)
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(i).zfill(2) for i, _ in todo]
    paths = [str(p) for _, p in todo]
    t0 = time.time()
    try:
        transcripts, metas = transcribe_paths_modal(paths, language, model=args.model, options=overrides,
                                                    names=names, with_meta=True)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    for name, tr in zip(names, transcripts):
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(tr, f, ensure_ascii=False, indent=1)

    meta_path = out_dir / "_asr_meta.json"
    meta_all: Dict = {"model": args.model, "gpu": GPU, "compute_type": COMPUTE_TYPE,
                      "options": merged_options(overrides), "tracks": {}}
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("model") == args.model:
                meta_all["tracks"] = old.get("tracks", {})
        except json.JSONDecodeError:
            pass
    for name, m in zip(names, metas):
        meta_all["tracks"][name] = m
    meta_all["last_run"] = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "tracks": names,
                            "wall_s": round(time.time() - t0, 1),
                            "gpu_s": round(sum(m["gpu_s"] for m in metas), 1)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(names)} transcript(s) to {out_dir}/ (+ _asr_meta.json) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
