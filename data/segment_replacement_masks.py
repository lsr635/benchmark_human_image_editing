#!/usr/bin/env python3
"""Generate segmentation masks for DreamEditBench replacement tasks using LangSAM."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure ImagenHub modules are importable
REPO_ROOT = Path(__file__).resolve().parents[1] / "ImagenHub" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imagen_hub.miscmodels import LangSAM, draw_image  # noqa: E402


def choose_mask(masks):
    """Pick the largest mask (by area) from LangSAM output."""
    if masks.numel() == 0:
        return None
    if masks.ndim == 2:
        return masks
    areas = masks.flatten(1).sum(dim=1)
    idx = int(areas.argmax())
    return masks[idx]


def save_mask(mask_tensor, save_path: Path) -> None:
    """Convert mask tensor to single-channel image and save."""
    mask_np = (mask_tensor.cpu().numpy() > 0).astype(np.uint8) * 255
    Image.fromarray(mask_np, mode="L").save(save_path)


def overlay_debug(image: Image.Image, masks, boxes, phrases, save_path: Path) -> None:
    """Save visualization with masks and boxes for manual inspection."""
    image_np = np.asarray(image)
    overlay = draw_image(image_np, masks, boxes, [str(p) for p in phrases])
    Image.fromarray(overlay).save(save_path)


def segment_replacement(replacement_root: Path, output_dir: Path, prompt_field: str, box_thresh: float, text_thresh: float, step: float, visualize: bool) -> None:
    csv_path = replacement_root.parent / "replacement.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"replacement.csv not found next to {replacement_root}")

    lang_sam = LangSAM()
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc="replacement", unit="sample"):
        sample_id = row["id"].zfill(4)
        source_rel = row["source_image_path"]
        prompt = row.get(prompt_field, "").strip()
        if not prompt:
            prompt = row.get("subject_names", "").strip()
        if not prompt:
            print(f"[WARN] sample {sample_id}: empty prompt, skipping")
            continue

        source_path = replacement_root.parent / source_rel
        if not source_path.exists():
            print(f"[WARN] sample {sample_id}: missing source image {source_rel}")
            continue

        image = Image.open(source_path).convert("RGB")
        masks, boxes, phrases, _ = lang_sam.predict_adaptive(image, prompt, box_threshold=box_thresh, text_threshold=text_thresh, step=step)
        mask_tensor = choose_mask(masks)
        if mask_tensor is None:
            print(f"[WARN] sample {sample_id}: segmentation returned empty mask")
            continue

        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_mask(mask_tensor, sample_dir / "mask.png")
        if visualize:
            overlay_debug(image, masks, boxes, phrases, sample_dir / "mask_debug.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment DreamEditBench replacement regions with LangSAM")
    parser.add_argument(
        "--replacement_root",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/DreamEditBench/DreamEditBench/replacement"),
        help="Path to DreamEditBench/replacement folder",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/DreamEditBench/DreamEditBench/replacement/masks"),
        help="Where to store generated masks",
    )
    parser.add_argument("--prompt_field", choices=["target_prompt", "source_prompt", "subject_names"], default="target_prompt", help="CSV field used as grounding prompt")
    parser.add_argument("--box_thresh", type=float, default=0.3, help="GroundingDINO box threshold")
    parser.add_argument("--text_thresh", type=float, default=0.25, help="GroundingDINO text threshold")
    parser.add_argument("--step", type=float, default=0.05, help="Threshold increment when masks are empty")
    parser.add_argument("--visualize", action="store_true", help="Save mask overlays for debugging")
    args = parser.parse_args()

    segment_replacement(
    replacement_root=args.replacement_root,
        output_dir=args.output_dir,
        prompt_field=args.prompt_field,
        box_thresh=args.box_thresh,
        text_thresh=args.text_thresh,
        step=args.step,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
