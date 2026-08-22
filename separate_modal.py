#!/usr/bin/env python3
"""Vocal separation on Modal GPUs (drop-in for the local demucs step).

    python separate_modal.py configs/carmen.yaml            # separate every track lacking a vocals stem
    python separate_modal.py configs/carmen.yaml --dry-run  # just list what would run
    python separate_modal.py configs/carmen.yaml --force    # redo all tracks

Reads audio/<prefix>/NN.m4a, runs `demucs -n htdemucs --two-stems=vocals` (same
model and defaults as separate.sh, so detect_instrumental.py's calibration
carries over) on an L4 per track, and writes sep/<prefix>_sep/htdemucs/NN/vocals.m4a.
Only the vocals stem comes back (nothing downstream uses no_vocals). Nothing is
stored on Modal: audio goes up as function arguments, stems come back as return
values.

Local demucs on this Mac is ~1x realtime (3-4 h per opera); an L4 does a 3-minute
track in ~10 s, so a whole opera is a few minutes and a few cents. The Modal
workspace is `chromatic` (MODAL_PROFILE is set to it unless KUNSTWERK_MODAL_PROFILE
says otherwise).

Exit codes: 0 ok; 1 failure; 3 Modal unavailable (not installed / no token) —
separate.sh falls back to local demucs on 3.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Must be set before `import modal` — the client reads the profile from the environment.
os.environ.setdefault("MODAL_PROFILE", os.getenv("KUNSTWERK_MODAL_PROFILE", "chromatic"))

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

import yaml

APP_NAME = "kunstwerk-demucs"
MODEL = "htdemucs"
GPU = os.getenv("KUNSTWERK_MODAL_GPU", "L4")

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1")
        # Pin to the versions the local venv uses; newer torchaudio removed the I/O demucs relies on.
        .pip_install("torch==2.5.1", "torchaudio==2.5.1", "demucs==4.0.1", "soundfile")
        # Bake the weights into the image so containers don't fetch them on cold start.
        .run_commands(f"python -c \"from demucs.pretrained import get_model; get_model('{MODEL}')\"")
    )
    app = modal.App(APP_NAME)

    @app.cls(image=image, gpu=GPU, timeout=30 * 60, retries=1, max_containers=8, scaledown_window=60)
    class Separator:
        @modal.enter()
        def load(self):
            import torch
            from demucs.pretrained import get_model

            self.model = get_model(MODEL)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

        @modal.method()
        def separate(self, name: str, audio: bytes) -> bytes:
            """m4a bytes in -> vocals m4a bytes out."""
            import subprocess
            import tempfile

            import torch
            from demucs.apply import apply_model
            from demucs.audio import AudioFile, save_audio

            with tempfile.TemporaryDirectory() as td:
                src = Path(td) / f"{name}.m4a"
                src.write_bytes(audio)
                wav = AudioFile(src).read(streams=0, samplerate=self.model.samplerate, channels=self.model.audio_channels)
                ref = wav.mean(0)
                wav = (wav - ref.mean()) / ref.std()
                with torch.no_grad():
                    # demucs CLI defaults: shifts=1, split=True, overlap=0.25
                    sources = apply_model(self.model, wav[None], device=self.device, shifts=1, split=True,
                                          overlap=0.25, progress=False)[0]
                sources = sources * ref.std() + ref.mean()
                vocals = sources[self.model.sources.index("vocals")]
                wav_path = Path(td) / "vocals.wav"
                save_audio(vocals, str(wav_path), samplerate=self.model.samplerate)
                m4a_path = Path(td) / "vocals.m4a"
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav_path), str(m4a_path)], check=True)
                return m4a_path.read_bytes()


def tracks_to_do(prefix: str, force: bool) -> list:
    audio_dir = Path("audio") / prefix
    sep_dir = Path("sep") / f"{prefix}_sep" / MODEL
    todo = []
    for f in sorted(audio_dir.glob("[0-9][0-9].m4a")):
        name = f.stem
        out = sep_dir / name / "vocals.m4a"
        out_wav = sep_dir / name / "vocals.wav"
        if not force and ((out.exists() and out.stat().st_size > 0) or (out_wav.exists() and out_wav.stat().st_size > 0)):
            continue
        todo.append((name, f, out))
    return todo


def main() -> int:
    ap = argparse.ArgumentParser(description="demucs vocal separation on Modal")
    ap.add_argument("config", help="configs/<opera>.yaml")
    ap.add_argument("--dry-run", action="store_true", help="list the tracks that would be separated and exit")
    ap.add_argument("--force", action="store_true", help="re-separate tracks that already have a vocals stem")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    prefix = cfg.get("file_prefix")
    if not prefix:
        print("config has no file_prefix", file=sys.stderr)
        return 1

    todo = tracks_to_do(prefix, args.force)
    if not todo:
        print(f"all tracks in audio/{prefix} already have vocals stems under sep/{prefix}_sep/{MODEL}/")
        return 0
    print(f"{len(todo)} track(s) to separate on Modal ({GPU}, profile {os.environ['MODAL_PROFILE']}): "
          + " ".join(n for n, _, _ in todo))
    if args.dry_run:
        return 0

    if modal is None:
        print("modal is not installed in this environment (pip install modal)", file=sys.stderr)
        return 3
    try:
        modal.config._check_config()  # type: ignore[attr-defined]  # raises if no token for this profile
    except Exception as e:  # noqa: BLE001
        print(f"Modal is not configured for profile {os.environ['MODAL_PROFILE']}: {e}", file=sys.stderr)
        return 3

    names = [n for n, _, _ in todo]
    payloads = [p.read_bytes() for _, p, _ in todo]
    outs = {n: o for n, _, o in todo}
    t0 = time.time()
    failures = []
    with modal.enable_output():
        with app.run():
            sep = Separator()
            for name, result in zip(names, sep.separate.map(names, payloads, return_exceptions=True)):
                if isinstance(result, Exception):
                    failures.append((name, result))
                    print(f"  {name}: FAILED {result!r}", file=sys.stderr)
                    continue
                out = outs[name]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(result)
                print(f"  {name}: {len(result) / 1e6:.1f} MB -> {out}  [{time.time() - t0:.0f}s]")
    if failures:
        print(f"{len(failures)} track(s) failed: {' '.join(n for n, _ in failures)}", file=sys.stderr)
        return 1
    print(f"separated {len(names)} track(s) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
