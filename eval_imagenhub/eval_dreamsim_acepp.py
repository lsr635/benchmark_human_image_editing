#!/usr/bin/env python3
"""DreamSim evaluation tailored for ACE++ addition outputs."""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch

import sys
sys.path.insert(0, "/project/imgtextmod/imagenhub/liusr/ImagenHub/src")

from imagen_hub.metrics.dreamsim_metric import MetricDreamSim  # noqa: E402
from imagen_hub.utils import load_image  # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def iter_result_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def locate_image(sample_dir: Path, base_name: str) -> Path:
    candidate = sample_dir / base_name
    if candidate.exists():
        return candidate
    stem = Path(base_name).stem
    for ext in IMG_EXTS:
        alt = sample_dir / f"{stem}{ext}"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"{base_name} not found in {sample_dir}")


def evaluate_dreamsim(
    results_dir: Path,
    input_root: Path,
    csv_path: Path,
    model_name: str,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dreamsim_metric = MetricDreamSim(device=device)

    rows: List[dict] = []

    for result_path in iter_result_images(results_dir):
        sample_id = result_path.stem
        sample_dir = input_root / sample_id
        if not sample_dir.is_dir():
            print(f"[WARN] missing input directory for {sample_id}")
            continue

        try:
            input_path = locate_image(sample_dir, "input.png")
            subject_path = locate_image(sample_dir, "subject.png")
        except FileNotFoundError as exc:
            print(f"[WARN] {sample_id}: {exc}")
            continue

        result_img = load_image(str(result_path))
        input_img = load_image(str(input_path))
        subject_img = load_image(str(subject_path))

        score_subject = dreamsim_metric.evaluate(subject_img, result_img)
        score_input = dreamsim_metric.evaluate(input_img, result_img)

        rows.append(
            {
                "sample_id": sample_id,
                "model": model_name,
                "dreamsim_gen_vs_subject": score_subject,
                "dreamsim_gen_vs_input": score_input,
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
        print(f"DreamSim metrics saved to {csv_path}")
    else:
        print("No valid DreamSim pairs found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DreamSim on ACE++ addition outputs")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/ACEPP_outputs/addition"),
        help="Directory containing generated addition results (flat structure)",
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/ACEPP_inputs/addition"),
        help="Directory with per-sample folders holding input/subject images",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("/project/imgtextmod/imagenhub/liusr/benchmark/eval_results/acepp_addition_dreamsim.csv"),
        help="Output CSV file",
    )
    parser.add_argument(
        "--model_name",
        default="ACE++",
        help="Label for the model column in the CSV",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_dreamsim(args.results_dir, args.input_root, args.csv_path, args.model_name)
