import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_OUTPUT_NAME = {
    "addition": "prompt_addition_qwen.txt",
    "replacement": "prompt_replacement_qwen.txt",
}


def load_model() -> Tuple[AutoProcessor, AutoModelForVision2Seq, torch.device]:
    print("Loading Qwen model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    if torch.cuda.is_available():
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        device = torch.device("cuda")
    else:
        model = AutoModelForVision2Seq.from_pretrained(MODEL_ID)
        device = torch.device("cpu")
        model.to(device)
    return processor, model, device


def extract_mask_metadata(label_path: Path) -> Dict[str, float]:
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    width = int(data.get("imageWidth"))
    height = int(data.get("imageHeight"))
    shapes = data.get("shapes", [])
    if not shapes:
        raise ValueError(f"`shapes` is empty in {label_path}")

    shape = shapes[0]
    label = shape.get("label", "")
    points = shape.get("points", [])
    if not points:
        raise ValueError(f"No points found in first shape of {label_path}")

    if shape.get("shape_type", "rectangle") == "rectangle" and len(points) == 2:
        (x0, y0), (x1, y1) = points
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

    x0_i, y0_i, x1_i, y1_i = map(int, map(round, (x0, y0, x1, y1)))
    area = max(x1 - x0, 0) * max(y1 - y0, 0)
    area_percent = (area / (width * height)) * 100 if width * height else 0

    return {
        "label": label,
        "width": width,
        "height": height,
        "x0": x0_i,
        "y0": y0_i,
        "x1": x1_i,
        "y1": y1_i,
        "area_percent": area_percent,
    }


def build_addition_instruction(metadata: Dict[str, float]) -> str:
    label = metadata.get("label", "the subject") or "the subject"
    width = metadata["width"]
    height = metadata["height"]
    x0 = metadata["x0"]
    y0 = metadata["y0"]
    x1 = metadata["x1"]
    y1 = metadata["y1"]
    area_percent = metadata["area_percent"]

    return (
        "You are an expert creative director writing prompts for a diffusion-based image editing system. "
        "Image A is the base scene to edit and Image B shows the subject that must be added. "
        f"Insert the subject (category: {label}) into Image A so it fits entirely inside the rectangle "
        f"with top-left ({x0}, {y0}) and bottom-right ({x1}, {y1}) on a {width}x{height} canvas (approx. "
        f"{area_percent:.1f}% of the frame). Describe the desired pose, scale, orientation, and location "
        "so the subject looks naturally grounded. Match Image A's lighting, shadows, color temperature, "
        "and perspective. Mention nearby context so the editor understands how the subject should "
        "interact with surrounding elements. Write three or four complete sentences in one paragraph. "
        "Avoid numbered lists or meta commentary; return only the final instruction."
    )


def build_replacement_instruction(metadata: Dict[str, float]) -> str:
    label = metadata.get("label", "the subject") or "the subject"
    width = metadata["width"]
    height = metadata["height"]
    x0 = metadata["x0"]
    y0 = metadata["y0"]
    x1 = metadata["x1"]
    y1 = metadata["y1"]
    area_percent = metadata["area_percent"]

    return (
        "You are an expert creative director writing prompts for a diffusion-based image editing system. "
        "Image A is the scene that currently contains an object to be replaced, and Image B shows the new subject. "
        f"Replace everything inside the rectangle with top-left ({x0}, {y0}) and bottom-right ({x1}, {y1}) on the "
        f"{width}x{height} canvas (about {area_percent:.1f}% of the frame) so the original content is completely removed and "
        f"the new subject (category: {label}) from Image B takes its place. Describe the desired pose, scale, orientation, "
        "and alignment so the subject looks naturally integrated, respecting Image A's camera perspective, horizon, "
        "and depth cues. Specify how to blend edges, lighting, shadows, reflections, and color temperature so the swap is seamless. "
        "Mention nearby environment elements the subject should interact with or occlude. Write three or four full sentences in one paragraph. "
        "Avoid bullets or meta commentary; return only the final instruction."
    )


def iter_sample_dirs(dataset_root: Path) -> Iterable[Path]:
    for path in sorted(dataset_root.iterdir()):
        if path.is_dir():
            yield path


def load_images(sample_dir: Path, base_name: str, subject_name: str) -> Tuple[Image.Image, Image.Image]:
    base_path = sample_dir / base_name
    subject_path = sample_dir / subject_name
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing add.png in {sample_dir}")
    if not subject_path.is_file():
        raise FileNotFoundError(f"Missing token_add.png in {sample_dir}")
    base_image = Image.open(base_path).convert("RGB")
    subject_image = Image.open(subject_path).convert("RGB")
    return base_image, subject_image


def generate_prompt(processor: AutoProcessor,
                    model: AutoModelForVision2Seq,
                    device: torch.device,
                    instruction: str,
                    base_image: Image.Image,
                    subject_image: Image.Image,
                    max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    chat_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(
        text=[chat_prompt],
        images=[[base_image, subject_image]],
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    cleaned = text.strip()
    marker = "\nassistant\n"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[1].strip()
    elif cleaned.lower().startswith("assistant:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned


def save_prompt(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate ACE++ addition prompts using Qwen2-VL."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path containing numbered folders with add.png/token_add.png/label.json."
    )
    parser.add_argument(
        "--task",
        choices=["addition", "replacement"],
        default="addition",
        help="Which edit task to describe (addition or replacement)."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional filename for the generated prompt in each sample folder."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate prompts even if the output file already exists."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
        help="Maximum tokens to generate per sample."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset_root `{dataset_root}` does not exist or is not a directory.")

    processor, model, device = load_model()
    output_name = args.output_name or DEFAULT_OUTPUT_NAME[args.task]

    for sample_dir in iter_sample_dirs(dataset_root):
        output_path = sample_dir / output_name
        if output_path.exists() and not args.overwrite:
            print(f"[Skip] {sample_dir.name}: {output_path.name} exists")
            continue

        label_path = sample_dir / "label.json"
        if not label_path.is_file():
            print(f"[Warn] {sample_dir.name}: missing label.json; skipping")
            continue

        try:
            metadata = extract_mask_metadata(label_path)
            if args.task == "addition":
                instruction = build_addition_instruction(metadata)
                base_image, subject_image = load_images(sample_dir, "add.png", "token_add.png")
            else:
                instruction = build_replacement_instruction(metadata)
                base_image, subject_image = load_images(sample_dir, "rep.png", "token_rep.png")
            prompt_text = generate_prompt(
                processor,
                model,
                device,
                instruction,
                base_image,
                subject_image,
                args.max_new_tokens,
            )
        except Exception as exc:
            print(f"[Error] {sample_dir.name}: {exc}")
            continue

        save_prompt(output_path, prompt_text)
        print(f"[OK] {sample_dir.name}: saved {output_path.name}")


if __name__ == "__main__":
    main()