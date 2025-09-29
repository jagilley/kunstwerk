"""Playground CLI for fal.ai Seedream and Dreamina image models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

import fal_client

from anime.image_batch_runner import GenerationRequest, load_style_from_path, save_binary_file
from anime.seedream4 import SeedreamProvider

try:
    import gradio as gr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for CLI mode
    gr = None  # type: ignore[assignment]

load_dotenv()

DREAMINA_MODEL_ID = "fal-ai/bytedance/dreamina/v3.1/text-to-image"


class PlaygroundArgs(argparse.Namespace):
    """Satisfy the attribute contract expected by SeedreamProvider."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure attributes referenced inside SeedreamProvider exist
        defaults = {
            "input_image": kwargs.get("input_image"),
            "jsonl_image_field": None,
            "jsonl_image_root": None,
            "num_images": kwargs.get("num_images"),
            "max_images": kwargs.get("max_images"),
            "image_size": kwargs.get("image_size"),
            "seed": kwargs.get("seed"),
            "sync_mode": kwargs.get("sync_mode"),
            "disable_safety_checker": kwargs.get("disable_safety_checker"),
            "show_logs": kwargs.get("show_logs"),
        }
        for key, value in defaults.items():
            setattr(self, key, value)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Playground for Seedream v4 (edit/text) and Dreamina v3.1 text-to-image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio UI instead of the CLI",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--prompt", type=str, help="Prompt text to send to Seedream")
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        help="Path to a file containing the prompt",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="seedream_playground_outputs",
        help="Directory for saving generated images",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="seedream",
        help="Filename prefix for saved images",
    )
    parser.add_argument(
        "--input-image",
        action="append",
        default=None,
        help="Base image(s) to upload for edit mode (can repeat)",
    )
    parser.add_argument(
        "--ref-image",
        action="append",
        default=None,
        help="Additional reference image(s) to upload alongside the prompt",
    )
    parser.add_argument(
        "--style-guide",
        type=str,
        default=None,
        help="Optional inline text or path to style guide instructions",
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
        help="Number of generations per request (1-6)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images returned per generation (1-6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic results",
    )
    parser.add_argument(
        "--sync-mode",
        action="store_true",
        help="Enable sync_mode for inline data URIs",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        default=True,
        help="Disable the Seedream safety checker",
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Stream fal.ai logs while the job is running",
    )
    parser.add_argument(
        "--model",
        choices=["seedream", "dreamina"],
        default="seedream",
        help="Image model to target",
    )

    parsed = parser.parse_args(argv)
    if not parsed.ui and not (parsed.prompt or parsed.prompt_file):
        parser.error("one of the arguments --prompt --prompt-file is required unless --ui is set")
    return parsed


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    path = Path(args.prompt_file)
    return path.read_text(encoding="utf-8").strip()


def resolve_style(style_arg: Optional[str]) -> Optional[str]:
    if not style_arg:
        return None
    path = Path(style_arg)
    if path.exists():
        return load_style_from_path(path)
    return style_arg


def build_request(args: argparse.Namespace, prompt_text: str) -> GenerationRequest:
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_paths: List[Path] = []
    if args.ref_image:
        for raw in args.ref_image:
            ref_paths.append(Path(raw).expanduser())

    style_text = resolve_style(args.style_guide)

    return GenerationRequest(
        prompt=prompt_text,
        index=1,
        total=1,
        out_dir=out_dir,
        prefix=args.prefix,
        style_guide=style_text,
        ref_images=ref_paths,
        record=None,
    )


def build_text_prompt(request: GenerationRequest) -> str:
    segments: List[str] = []
    if request.style_guide:
        segments.append(
            "Global film style: apply consistently across this frame.\n" + request.style_guide
        )
    segments.append(request.prompt)
    return "\n\n".join(seg for seg in segments if seg)


def run_dreamina(args: argparse.Namespace, request: GenerationRequest) -> None:
    if args.input_image:
        print("Note: Dreamina is text-to-image; ignoring --input-image")
    if request.ref_images:
        print("Note: Dreamina cannot attach reference images; ignoring --ref-image")

    prompt_text = build_text_prompt(request)

    payload = {
        "prompt": prompt_text,
        "num_images": max(1, min(args.num_images or 1, 6)),
    }
    if args.max_images:
        payload["max_images"] = max(1, min(args.max_images, 6))
    if args.image_size:
        payload["image_size"] = SeedreamProvider._parse_image_size(args.image_size)
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.sync_mode:
        payload["sync_mode"] = True
    if args.disable_safety_checker:
        payload["enable_safety_checker"] = False

    subscribe_kwargs = {"with_logs": bool(args.show_logs)}
    if args.show_logs:
        subscribe_kwargs["on_queue_update"] = SeedreamProvider._on_queue_update

    try:
        result = fal_client.subscribe(DREAMINA_MODEL_ID, arguments=payload, **subscribe_kwargs)
    except Exception as exc:
        print(f"Dreamina request failed: {exc}")
        return

    images = result.get("images") if isinstance(result, dict) else None
    if not images:
        print("Dreamina returned no images for this prompt")
        return

    variant = 0
    for image_obj in images:
        url = image_obj.get("url") if isinstance(image_obj, dict) else None
        if not url:
            continue
        try:
            data, content_type = SeedreamProvider._download_image(url)
        except RuntimeError as exc:
            print(exc)
            continue
        ext = SeedreamProvider._infer_extension(url, content_type)
        file_path = request.out_dir / f"{request.prefix}_{request.index:03d}_{variant:02d}{ext}"
        save_binary_file(file_path, data)
        variant += 1

    seed_used = result.get("seed") if isinstance(result, dict) else None
    if seed_used is not None:
        print(f"Dreamina seed: {seed_used}")


def _ensure_gradio_available() -> None:
    if gr is None:  # pragma: no cover - executed only when dependency missing
        raise RuntimeError("Gradio is not installed. Run `pip install gradio` to launch the UI.")


def _collect_new_files(out_dir: Path, before: set[str]) -> List[Path]:
    try:
        current_files = [p for p in out_dir.iterdir() if p.is_file()]
    except FileNotFoundError:
        return []
    new_files = [p for p in current_files if p.name not in before]
    return sorted(new_files, key=lambda p: (p.stat().st_mtime, p.name))


def _coerce_optional_int(raw_value: Optional[str], field_name: str) -> Optional[int]:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - input validation
        raise ValueError(f"{field_name} must be an integer") from exc


def _normalize_path_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    paths: List[str] = []
    for item in values:
        if not item:
            continue
        paths.append(str(Path(item)))
    return paths or None


def _format_status_message(files: List[Path]) -> str:
    if not files:
        return "No new files were saved. Check the console logs for details."
    lines = ["Saved files:"]
    lines.extend(str(p) for p in files)
    return "\n".join(lines)


def gradio_generate(
    model_choice: str,
    prompt_text: str,
    style_guide_text: str,
    style_guide_file: Optional[str],
    input_images: Optional[List[str]],
    ref_images: Optional[List[str]],
    num_images: int,
    max_images_value: Optional[str],
    image_size: str,
    seed_value: Optional[str],
    sync_mode: bool,
    disable_safety_checker: bool,
    show_logs: bool,
    outdir: str,
    prefix: str,
):
    _ensure_gradio_available()

    if not prompt_text or not prompt_text.strip():
        raise gr.Error("Prompt is required.")

    prompt = prompt_text.strip()
    style_text = style_guide_text.strip() if style_guide_text else ""
    style_arg: Optional[str] = None
    if style_text:
        style_arg = style_text
    elif style_guide_file:
        style_arg = style_guide_file

    try:
        seed = _coerce_optional_int(seed_value, "Seed")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    try:
        max_images = _coerce_optional_int(max_images_value, "Max images")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    input_list = _normalize_path_list(input_images)
    ref_list = _normalize_path_list(ref_images)

    outdir_value = outdir.strip() if outdir else "seedream_playground_outputs"
    if not outdir_value:
        outdir_value = "seedream_playground_outputs"
    prefix_value = prefix.strip() if prefix else "seedream"
    if not prefix_value:
        prefix_value = "seedream"

    parsed_args = argparse.Namespace(
        prompt=prompt,
        prompt_file=None,
        outdir=outdir_value,
        prefix=prefix_value,
        input_image=input_list,
        ref_image=ref_list,
        style_guide=style_arg,
        image_size=image_size.strip() if image_size else None,
        num_images=int(num_images),
        max_images=max_images,
        seed=seed,
        sync_mode=bool(sync_mode),
        disable_safety_checker=bool(disable_safety_checker),
        show_logs=bool(show_logs),
        model=model_choice,
    )

    request = build_request(parsed_args, prompt)
    before = {p.name for p in request.out_dir.iterdir() if p.is_file()}

    try:
        if parsed_args.model == "dreamina":
            run_dreamina(parsed_args, request)
        else:
            provider_args = PlaygroundArgs(
                input_image=parsed_args.input_image,
                num_images=parsed_args.num_images,
                max_images=parsed_args.max_images,
                image_size=parsed_args.image_size,
                seed=parsed_args.seed,
                sync_mode=parsed_args.sync_mode,
                disable_safety_checker=parsed_args.disable_safety_checker,
                show_logs=parsed_args.show_logs,
            )
            provider = SeedreamProvider(provider_args)
            provider.setup()
            provider.generate(request)
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected issues
        raise gr.Error(f"Generation failed: {exc}") from exc

    new_files = _collect_new_files(request.out_dir, before)
    status = _format_status_message(new_files)
    return [str(p) for p in new_files], status


def launch_gradio_app() -> None:
    _ensure_gradio_available()

    if gr is None:  # pragma: no cover - guarded in _ensure_gradio_available
        return

    max_image_choices = ["None"] + [str(i) for i in range(1, 7)]

    with gr.Blocks(title="Seedream Playground") as demo:
        gr.Markdown("""
        ## Seedream Playground
        Provide a prompt, tweak the options, and generate images using Seedream v4 or Dreamina v3.1.
        Outputs are saved locally using the same logic as the CLI.
        """)

        with gr.Row():
            model = gr.Radio(
                choices=["seedream", "dreamina"],
                value="seedream",
                label="Model",
            )
            num_images = gr.Slider(
                minimum=1,
                maximum=6,
                value=1,
                step=1,
                label="Images per request",
            )
            max_images = gr.Dropdown(
                choices=max_image_choices,
                value="None",
                label="Max images returned",
            )
            image_size = gr.Textbox(
                label="Image size",
                placeholder="square_hd or WIDTHxHEIGHT",
            )

        prompt = gr.Textbox(
            label="Prompt",
            lines=4,
            placeholder="Describe the scene...",
        )

        with gr.Accordion("Optional style guide", open=False):
            style_text = gr.Textbox(
                label="Style guide text",
                lines=4,
                placeholder="Inline style instructions...",
            )
            style_file = gr.File(
                label="Upload style guide file",
                file_count="single",
                type="filepath",
            )

        with gr.Row():
            input_imgs = gr.File(
                label="Input images (for edit mode)",
                file_count="multiple",
                type="filepath",
            )
            ref_imgs = gr.File(
                label="Reference images",
                file_count="multiple",
                type="filepath",
            )

        with gr.Row():
            seed = gr.Textbox(label="Seed", placeholder="Optional integer seed")
            sync_mode = gr.Checkbox(label="Enable sync mode", value=False)
            disable_safety_checker = gr.Checkbox(
                label="Disable safety checker",
                value=True,
            )
            show_logs = gr.Checkbox(label="Show fal.ai logs in console", value=False)

        with gr.Row():
            outdir = gr.Textbox(
                label="Output directory",
                value="seedream_playground_outputs",
            )
            prefix = gr.Textbox(
                label="Filename prefix",
                value="seedream",
            )

        generate = gr.Button("Generate Images", variant="primary")

        gallery = gr.Gallery(label="Latest outputs", show_label=True, columns=3)
        status = gr.Textbox(label="Status", lines=6)

        generate.click(
            fn=gradio_generate,
            inputs=[
                model,
                prompt,
                style_text,
                style_file,
                input_imgs,
                ref_imgs,
                num_images,
                max_images,
                image_size,
                seed,
                sync_mode,
                disable_safety_checker,
                show_logs,
                outdir,
                prefix,
            ],
            outputs=[gallery, status],
        )

    demo.queue()
    demo.launch()


def main(argv: Optional[List[str]] = None) -> None:
    parsed = parse_args(argv)
    if getattr(parsed, "ui", False):
        launch_gradio_app()
        return
    prompt_text = resolve_prompt(parsed)
    if not prompt_text:
        raise SystemExit("Prompt is empty")

    request = build_request(parsed, prompt_text)
    if parsed.model == "dreamina":
        run_dreamina(parsed, request)
        return

    provider_args = PlaygroundArgs(
        input_image=parsed.input_image,
        num_images=parsed.num_images,
        max_images=parsed.max_images,
        image_size=parsed.image_size,
        seed=parsed.seed,
        sync_mode=parsed.sync_mode,
        disable_safety_checker=parsed.disable_safety_checker,
        show_logs=parsed.show_logs,
    )

    provider = SeedreamProvider(provider_args)
    provider.setup()
    provider.generate(request)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print("Interrupted")
