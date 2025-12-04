#!/usr/bin/env python3

"""Preprocess DreamEditBench into ACE++ friendly layout."""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from dashscope import MultiModalConversation
from dashscope.api_entities.dashscope_response import Message

PROMPT_FALLBACKS = [
    "A high quality photo of the main subject in the scene.",
    "A studio shot highlighting the primary object.",
    "A natural photograph of the key object with soft lighting.",
]

def generate_prompt_vlm(subject_image: Path, source_image: Path) -> str:
    instruction = (
        "You are crafting an addition prompt for an image editing model. "
        "First image is the subject to insert, second image is the background scene. "
        "Describe a concise English instruction (<= 40 words) that tells the model "
        "to place the subject into the background, specifying position or relation "
        "so that the result feels natural."
    )

    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": instruction},
                {"type": "image", "image": str(subject_image)},
                {"type": "image", "image": str(source_image)},
            ],
        )
    ]

    response = MultiModalConversation.call(model="qwen-vl-plus", messages=messages)
    try:
        text = response.output.choices[0].message.content[0]["text"].strip()
    except Exception as exc:
        raise RuntimeError(f"Qwen-VL response parse error: {exc}")
    return text


def fallback_prompt() -> str:
    import random

    return random.choice(PROMPT_FALLBACKS)


def parse_bbox(bbox_str: str) -> List[int]:
    values = bbox_str.strip()[1:-1]
    coords = [int(float(x)) for x in values.split()]
    if len(coords) != 4:
        raise ValueError(f"Unexpected bbox: {bbox_str}")
    return coords


def save_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").save(dst)


def make_mask(width: int, height: int, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    mask[y1:y2, x1:x2] = 255
    return Image.fromarray(mask)


def preprocess_addition(csv_path: Path, dataset_root: Path, out_root: Path, use_vlm: bool) -> None:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(list(reader), desc="addition", unit="sample"):
            sample_id = row["id"].zfill(4)
            subject = row["subject_image_path"]
            source = row["source_image_path"]
            bbox = parse_bbox(row["add_bounding_box"])
            target_prompt = row.get("target_prompt", "")

            subject_path = dataset_root / subject
            source_path = dataset_root / source
            if not subject_path.exists() or not source_path.exists():
                print(f"[WARN] missing files for {sample_id}")
                continue

            with Image.open(source_path) as img:
                width, height = img.size
            mask = make_mask(width, height, *bbox)

            sample_dir = out_root / "addition" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_image(source_path, sample_dir / "input.png")
            save_image(subject_path, sample_dir / "subject.png")
            mask.save(sample_dir / "mask.png")

            base_prompt = target_prompt if target_prompt and target_prompt.upper() != "NA" else ""
            vlm_prompt = ""

            if use_vlm:
                try:
                    vlm_prompt = generate_prompt_vlm(subject_path, source_path)
                except Exception as exc:
                    print(f"[WARN] sample {sample_id}: VLM prompt failed ({exc}); falling back")

            prompt_text = vlm_prompt or base_prompt or fallback_prompt()
            (sample_dir / "prompt.txt").write_text(prompt_text)

            meta = {
                "sample_id": sample_id,
                "task": "addition",
                "source_image": str(source_path.resolve()),
                "subject_image": str(subject_path.resolve()),
                "bbox": bbox,
                "target_prompt": target_prompt,
                "prompt_used": prompt_text,
            }
            (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def preprocess(dataset_root: Path, out_root: Path, use_vlm: bool) -> None:
    dataset_root = dataset_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_root / "addition.csv"
    if csv_path.exists():
        preprocess_addition(csv_path, dataset_root, out_root, use_vlm)
    else:
        print("[WARN] addition.csv not found; nothing to preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DreamEditBench for ACE++ inference")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/DreamEditBench/DreamEditBench"),
        help="Path to DreamEditBench root (contains addition.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/ACEPP_inputs"),
        help="Processed output directory",
    )
    parser.add_argument("--use_vlm", action="store_true", help="Use Qwen-VL to generate prompts when missing")
    args = parser.parse_args()

    preprocess(args.dataset_root, args.output_dir, args.use_vlm)


if __name__ == "__main__":
    main()