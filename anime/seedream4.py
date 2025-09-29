"""
Batch image editing pipeline for ByteDance Seedream v4 via fal.ai.

Dependencies: pip install fal-client
Env: FAL_KEY must be set
"""

from __future__ import annotations

import argparse
import mimetypes
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from dotenv import load_dotenv

import fal_client

from anime.image_batch_runner import (
    GenerationRequest,
    add_common_arguments,
    gather_generation_requests,
    run_batch,
    save_binary_file,
)

load_dotenv()


class SeedreamProvider:
    MODEL_ID_EDIT = "fal-ai/bytedance/seedream/v4/edit"
    MODEL_ID_TEXT = "fal-ai/bytedance/seedream/v4/text-to-image"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.upload_cache: Dict[Path, str] = {}
        self.static_image_urls: List[str] = []
        self.parsed_image_size: Optional[dict | str] = None

    @staticmethod
    def add_model_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--input-image",
            action="append",
            default=None,
            help="Local image path(s) to upload once and include with every request",
        )
        parser.add_argument(
            "--jsonl-image-field",
            type=str,
            default=None,
            help="JSONL field containing per-record image path(s) for editing",
        )
        parser.add_argument(
            "--jsonl-image-root",
            type=str,
            default=None,
            help="Optional base directory to resolve JSONL image paths",
        )
        parser.add_argument(
            "--image-size",
            type=str,
            default=None,
            help="Seedream image_size preset (e.g. square_hd) or custom WIDTHxHEIGHT",
        )
        parser.add_argument(
            "--num-images",
            type=int,
            default=1,
            help="Number of generations to request per prompt (1-6)",
        )
        parser.add_argument(
            "--max-images",
            type=int,
            default=None,
            help="Upper bound on images per generation (1-6)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Optional random seed for deterministic outputs",
        )
        parser.add_argument(
            "--sync-mode",
            action="store_true",
            help="Enable sync_mode to receive inline data URIs",
        )
        parser.add_argument(
            "--disable-safety-checker",
            action="store_true",
            help="Disable the model safety checker",
        )
        parser.add_argument(
            "--show-logs",
            action="store_true",
            help="Stream provider logs while waiting for outputs",
        )

    def setup(self) -> None:
        api_key = os.environ.get("FAL_KEY")
        if not api_key:
            raise RuntimeError("FAL_KEY not set; Seedream requests require fal.ai API key")

        if self.args.image_size:
            self.parsed_image_size = self._parse_image_size(self.args.image_size)

        static_paths = self.args.input_image or []
        for path_str in static_paths:
            path = Path(path_str).expanduser()
            if self.args.jsonl_image_root and not path.is_absolute():
                path = Path(self.args.jsonl_image_root).expanduser() / path
            if not path.is_file():
                raise RuntimeError(f"input image not found: {path}")
            url = self._upload_image(path)
            if url:
                self.static_image_urls.append(url)

    def generate(self, request: GenerationRequest) -> None:
        image_urls: List[str] = list(self.static_image_urls)

        if self.args.jsonl_image_field and request.record:
            record_paths = self._extract_record_image_paths(request.record)
            for rpath in record_paths:
                url = self._upload_image(rpath)
                if url:
                    image_urls.append(url)

        if request.ref_images:
            for ref_path in request.ref_images:
                url = self._upload_image(ref_path)
                if url:
                    image_urls.append(url)

        if len(image_urls) > 10:
            print("Note: more than 10 images provided; Seedream will use the last 10 inputs")
            image_urls = image_urls[-10:]
        prompt_text = self._build_prompt(request)

        use_edit = bool(image_urls)
        model_id = self.MODEL_ID_EDIT if use_edit else self.MODEL_ID_TEXT

        payload = {
            "prompt": prompt_text,
            "num_images": max(1, min(self.args.num_images or 1, 6)),
        }
        if use_edit:
            payload["image_urls"] = image_urls
        if self.args.max_images:
            payload["max_images"] = max(1, min(self.args.max_images, 6))
        if self.parsed_image_size:
            payload["image_size"] = self.parsed_image_size
        if self.args.seed is not None:
            payload["seed"] = self.args.seed
        if self.args.sync_mode:
            payload["sync_mode"] = True
        if self.args.disable_safety_checker:
            payload["enable_safety_checker"] = False

        subscribe_kwargs = {"with_logs": bool(self.args.show_logs)}
        if self.args.show_logs:
            subscribe_kwargs["on_queue_update"] = self._on_queue_update

        try:
            result = fal_client.subscribe(model_id, arguments=payload, **subscribe_kwargs)
        except Exception as exc:
            print(f"Seedream request failed: {exc}")
            return

        images = result.get("images") if isinstance(result, dict) else None
        if not images:
            print("Seedream returned no images for this prompt")
            return

        variant = 0
        for image_obj in images:
            url = image_obj.get("url") if isinstance(image_obj, dict) else None
            if not url:
                continue
            try:
                data, content_type = self._download_image(url)
            except RuntimeError as exc:
                print(exc)
                continue
            ext = self._infer_extension(url, content_type)
            file_path = request.out_dir / (
                f"{request.prefix}_{request.index:03d}_{variant:02d}{ext}"
            )
            save_binary_file(file_path, data)
            variant += 1

        seed_used = result.get("seed") if isinstance(result, dict) else None
        if seed_used is not None:
            print(f"Seedream seed: {seed_used}")

    def _upload_image(self, path: Path) -> Optional[str]:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            print(f"Warning: image not found, skipping upload: {resolved}")
            return None
        cached = self.upload_cache.get(resolved)
        if cached:
            return cached
        try:
            url = fal_client.upload_file(str(resolved))
        except Exception as exc:
            print(f"Failed to upload {resolved}: {exc}")
            return None
        if not isinstance(url, str):
            print(f"Unexpected upload response for {resolved}: {url}")
            return None
        self.upload_cache[resolved] = url
        print(f"Uploaded {resolved} -> {url}")
        return url

    def _extract_record_image_paths(self, record: dict) -> List[Path]:
        field = self.args.jsonl_image_field
        if not field:
            return []
        value = record.get(field)
        raw_paths: List[str] = []
        if isinstance(value, str):
            raw_paths = [value]
        elif isinstance(value, Iterable):
            raw_paths = [str(item) for item in value]
        paths: List[Path] = []
        for raw in raw_paths:
            path = Path(raw).expanduser()
            if self.args.jsonl_image_root and not path.is_absolute():
                path = Path(self.args.jsonl_image_root).expanduser() / path
            paths.append(path)
        return paths

    def _build_prompt(self, request: GenerationRequest) -> str:
        segments: List[str] = []
        if request.style_guide:
            segments.append(
                "Global film style: apply consistently across this frame.\n" + request.style_guide
            )
        segments.append(request.prompt)
        if request.ref_images:
            try:
                ref_names = ", ".join(p.stem for p in request.ref_images)
            except Exception:
                ref_names = "character references"
            segments.append(
                "Use the attached character reference images as strict identity/style guides for: "
                + ref_names
            )
        return "\n\n".join(seg for seg in segments if seg)

    @staticmethod
    def _on_queue_update(update: object) -> None:
        if isinstance(update, fal_client.InProgress):
            for log in getattr(update, "logs", []) or []:
                message = log.get("message") if isinstance(log, dict) else None
                if message:
                    print(message)

    @staticmethod
    def _download_image(url: str) -> tuple[bytes, str]:
        try:
            with urlopen(url) as resp:
                data = resp.read()
                content_type = resp.headers.get("Content-Type", "")
        except URLError as exc:
            raise RuntimeError(f"Failed to download Seedream image {url}: {exc}") from exc
        return data, content_type

    @staticmethod
    def _infer_extension(url: str, content_type: str) -> str:
        mime = content_type.split(";")[0].strip() if content_type else ""
        ext = mimetypes.guess_extension(mime) if mime else None
        if not ext:
            path_ext = Path(urlparse(url).path).suffix
            ext = path_ext if path_ext else ".png"
        return ext if ext.startswith(".") else f".{ext}"

    @staticmethod
    def _parse_image_size(raw: str) -> dict | str:
        value = raw.strip()
        if "x" in value.lower():
            try:
                width_str, height_str = value.lower().split("x", 1)
                width = int(width_str)
                height = int(height_str)
                return {"width": width, "height": height}
            except ValueError as exc:
                raise RuntimeError(
                    "Custom image_size must be WIDTHxHEIGHT with integer values"
                ) from exc
        return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images from prompts using Seedream v4 (image-to-image)"
    )
    add_common_arguments(parser)
    SeedreamProvider.add_model_arguments(parser)

    args = parser.parse_args()

    requests = gather_generation_requests(args)
    provider = SeedreamProvider(args)
    run_batch(provider, requests)


if __name__ == "__main__":
    main()
