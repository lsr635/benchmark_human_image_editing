import argparse
import json
from pathlib import Path
from typing import Dict, List

TARGET_SIZE = 512


def load_label(label_path: Path) -> Dict:
    with label_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scale_point(value: float, scale: float) -> int:
    return int(round(value * scale))


def clamp(val: int, minimum: int = 0, maximum: int = TARGET_SIZE) -> int:
    return max(minimum, min(maximum, val))


def build_bbox(points: List[List[float]], width: int, height: int) -> List[int]:
    if len(points) < 2:
        raise ValueError("Need at least two points to define a bounding box.")

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    scale_x = TARGET_SIZE / width if width else 1.0
    scale_y = TARGET_SIZE / height if height else 1.0

    left = clamp(scale_point(x_min, scale_x))
    top = clamp(scale_point(y_min, scale_y))
    right = clamp(scale_point(x_max, scale_x))
    bottom = clamp(scale_point(y_max, scale_y))

    if right <= left:
        right = clamp(left + 1)
    if bottom <= top:
        bottom = clamp(top + 1)
    return [left, top, right, bottom]


def process_case(case_dir: Path) -> Dict[str, List[int]]:
    label_path = case_dir / "label.json"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label.json in {case_dir}")

    label_data = load_label(label_path)
    shapes = label_data.get("shapes", [])
    if not shapes:
        raise ValueError(f"No shapes found in {label_path}")

    bbox = build_bbox(
        shapes[0].get("points", []),
        int(label_data.get("imageWidth", TARGET_SIZE)),
        int(label_data.get("imageHeight", TARGET_SIZE)),
    )
    return {case_dir.name: bbox}


def build_bbox_map(dataset_root: Path) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for case_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        case_bbox = process_case(case_dir)
        key, value = next(iter(case_bbox.items()))
        mapping[f"{key}.png"] = value
    if not mapping:
        raise RuntimeError(f"No bounding boxes generated from {dataset_root}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ACE++ labels into DreamEdit bbox.json")
    parser.add_argument("dataset_root", type=Path, help="Path to dataset_* directory containing numbered folders")
    parser.add_argument("output_json", type=Path, help="Destination bbox.json path")
    args = parser.parse_args()

    bbox_map = build_bbox_map(args.dataset_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(bbox_map, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {len(bbox_map)} entries to {args.output_json}")


if __name__ == "__main__":
    main()
