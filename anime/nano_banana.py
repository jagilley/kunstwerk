"""
Batch image generation for Google's Gemini Image model using prompts.

Dependencies: pip install google-genai
Env: GEMINI_API_KEY must be set
"""

import argparse
import mimetypes
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from google import genai
from google.genai import types

from anime.image_batch_runner import (
    GenerationRequest,
    add_common_arguments,
    gather_generation_requests,
    run_batch,
    save_binary_file,
)

load_dotenv()


class GeminiImageProvider:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client: Optional[genai.Client] = None
        self.model = args.model

    @staticmethod
    def add_model_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            type=str,
            default="gemini-2.5-flash-image-preview",
            help="Gemini model name",
        )

    def setup(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)

    def generate(self, request: GenerationRequest) -> None:
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        parts: List[types.Part] = []
        if request.style_guide:
            parts.append(
                types.Part.from_text(
                    text=(
                        "Global film style: apply consistently across this frame.\n"
                        + request.style_guide
                    )
                )
            )
        parts.append(types.Part.from_text(text=request.prompt))

        if request.ref_images:
            try:
                ref_names = ", ".join(p.stem for p in request.ref_images)
            except Exception:
                ref_names = "character references"
            parts.append(
                types.Part.from_text(
                    text=(
                        "Use the attached character reference images as strict identity/style guides "
                        f"for: {ref_names}. Match faces, hair, outfit, and overall design."
                    )
                )
            )
            for ipath in request.ref_images:
                try:
                    with open(ipath, "rb") as f:
                        image_bytes = f.read()
                    mime = mimetypes.guess_type(str(ipath))[0] or "image/png"
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
                except Exception as e:
                    print(f"Warning: failed to read reference image {ipath}: {e}")

        contents = [types.Content(role="user", parts=parts)]
        cfg = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

        variant = 0
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=cfg,
        ):
            try:
                cand = chunk.candidates[0]
                chunk_parts = cand.content.parts if cand and cand.content else None
                if not chunk_parts:
                    continue
                part0 = chunk_parts[0]
                if getattr(part0, "inline_data", None) and getattr(part0.inline_data, "data", None):
                    inline_data = part0.inline_data
                    data_buffer: bytes = inline_data.data
                    ext = mimetypes.guess_extension(inline_data.mime_type) or ".png"
                    file_path = request.out_dir / (
                        f"{request.prefix}_{request.index:03d}_{variant:02d}{ext}"
                    )
                    save_binary_file(file_path, data_buffer)
                    variant += 1
                elif getattr(chunk, "text", None):
                    print(chunk.text)
            except Exception:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from prompts using Gemini")
    add_common_arguments(parser)
    GeminiImageProvider.add_model_arguments(parser)

    args = parser.parse_args()

    requests = gather_generation_requests(args)
    provider = GeminiImageProvider(args)
    run_batch(provider, requests)


if __name__ == "__main__":
    main()
