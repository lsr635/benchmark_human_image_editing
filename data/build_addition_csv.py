import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_label(label_path: Path) -> Dict:
    with label_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_shape(label_data: Dict) -> Tuple[str, List[List[float]]]:
    shapes = label_data.get("shapes", [])
    if not shapes:
        return "subject", [[0.0, 0.0], [0.0, 0.0]]
    shape = shapes[0]
    label = shape.get("label", "subject") or "subject"
    points = shape.get("points", [[0.0, 0.0], [0.0, 0.0]])
    return label, points


def to_bbox(points: List[List[float]]) -> List[int]:
    if len(points) < 2:
        return [0, 0, 0, 0]
    (x1, y1), (x2, y2) = points[0], points[1]
    return [round(x1), round(y1), round(x2), round(y2)]


def build_row(case_dir: Path,
              rel_subject_root: Path,
              rel_source_root: Path,
              default_prompt: Optional[str] = None) -> Optional[Dict[str, str]]:
    label_path = case_dir / "label.json"
    if not label_path.is_file():
        print(f"[Skip] {case_dir.name}: missing label.json")
        return None

    label_data = load_label(label_path)
    subject_name, points = resolve_shape(label_data)
    bbox = to_bbox(points)

    if any(coord == 0 for coord in bbox) and points != [[0.0, 0.0], [0.0, 0.0]]:
        print(f"[Warn] {case_dir.name}: bounding box may be incomplete")

    prompt_template = default_prompt or "photo of a {label}"
    target_prompt = prompt_template.format(label=subject_name)

    out_subject = rel_subject_root / f"{case_dir.name}.png"
    out_source = rel_source_root / f"{case_dir.name}.png"

    return {
        "id": case_dir.name,
        "subject_names": subject_name,
        "subject_image_path": str(out_subject).replace('\\', '/'),
        "source_image_path": str(out_source).replace('\\', '/'),
        "source_prompt": "NA",
        "target_prompt": target_prompt,
        "add_bounding_box": f"[{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}]",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate addition.csv from ACE++ style dataset_add folders."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to dataset_add directory containing numbered folders")
    parser.add_argument("output_csv", type=Path, help="Destination CSV path")
    parser.add_argument("--subject-root", type=Path, default=Path("addition/subject_image"),
                        help="Relative subject image root used in CSV paths")
    parser.add_argument("--source-root", type=Path, default=Path("addition/source_image"),
                        help="Relative source image root used in CSV paths")
    parser.add_argument("--default-prompt", type=str, default=None,
                        help="Template for target prompts; use {label} placeholder for subject name")
    args = parser.parse_args()

    rows = []
    for case_dir in sorted((p for p in args.dataset_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        row = build_row(case_dir,
                        args.subject_root,
                        args.source_root,
                        args.default_prompt)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid samples found; CSV not created.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "id",
            "subject_names",
            "subject_image_path",
            "source_image_path",
            "source_prompt",
            "target_prompt",
            "add_bounding_box",
        ])
        for row in rows:
            writer.writerow([
                row["id"],
                row["subject_names"],
                row["subject_image_path"],
                row["source_image_path"],
                row["source_prompt"],
                row["target_prompt"],
                row["add_bounding_box"],
            ])
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
