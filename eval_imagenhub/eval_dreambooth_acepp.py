#!/usr/bin/env python3
"""Evaluate ACE++ addition results against per-sample inputs/subjects."""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch

sys.path.insert(0, "/project/imgtextmod/imagenhub/liusr/ImagenHub/src")

from imagen_hub.metrics.dreambooth_metric import MetricCLIP_I, MetricDINO  # noqa: E402
from imagen_hub.utils import load_image  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def iter_generated_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def pick_image(sample_dir: Path, filename: str) -> Path:
    candidate = sample_dir / filename
    if candidate.exists():
        return candidate
    for ext in IMG_EXTS:
        alt = sample_dir / f"{Path(filename).stem}{ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"{filename} not found in {sample_dir}")


def batch_evaluate_addition(
    results_dir: Path,
    input_root: Path,
    csv_path: Path,
    model_name: str,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_metric = MetricCLIP_I(device=device)
    dino_metric = MetricDINO(device=device)

    rows: List[dict] = []

    for gen_path in iter_generated_images(results_dir):
        sample_id = gen_path.stem
        sample_dir = input_root / sample_id
        if not sample_dir.is_dir():
            print(f"[WARN] missing input directory for {sample_id}")
            continue

        try:
            input_path = pick_image(sample_dir, "input.png")
            subject_path = pick_image(sample_dir, "subject.png")
        except FileNotFoundError as exc:
            print(f"[WARN] {sample_id}: {exc}")
            continue

        gen_img = load_image(str(gen_path))
        input_img = load_image(str(input_path))
        subject_img = load_image(str(subject_path))

        clip_subject = clip_metric.evaluate(subject_img, gen_img)
        clip_input = clip_metric.evaluate(input_img, gen_img)
        dino_subject = dino_metric.evaluate(subject_img, gen_img)
        dino_input = dino_metric.evaluate(input_img, gen_img)

        rows.append(
            {
                "sample_id": sample_id,
                "model": model_name,
                "clip_gen_vs_subject": clip_subject,
                "clip_gen_vs_input": clip_input,
                "dino_gen_vs_subject": dino_subject,
                "dino_gen_vs_input": dino_input,
            }
        )

    if rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            avg_row = {col: df[col].mean() for col in numeric_cols}
        else:
            avg_row = {}
        avg_row.update({
            "sample_id": "average",
            "model": model_name,
        })
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
    else:
        print("No valid samples to evaluate.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ACE++ addition outputs")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/ACEPP_outputs/addition"),
        help="Directory containing generated result images (flat structure)",
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/ACEPP_inputs/addition"),
        help="Directory containing per-sample folders with input/subject images",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/benchmark/eval_results/acepp_addition_scores.csv"),
        help="Where to write the CSV summary",
    )
    parser.add_argument(
        "--model_name",
        default="ACE++",
        help="Model name label stored in the CSV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_evaluate_addition(args.results_dir, args.input_root, args.csv_path, args.model_name)
