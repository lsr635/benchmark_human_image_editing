import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _round_point(point):
    if len(point) != 2:
        raise ValueError(f"Point must have exactly 2 coordinates, got {point}")
    x, y = point
    return int(round(x)), int(round(y))


def _draw_shape(draw, shape):
    shape_type = shape.get("shape_type")
    points = shape.get("points", [])
    if not points:
        return

    if shape_type == "rectangle":
        if len(points) != 2:
            raise ValueError(f"Rectangle expects 2 points, got {len(points)}")
        top_left = _round_point(points[0])
        bottom_right = _round_point(points[1])
        draw.rectangle([top_left, bottom_right], fill=255)
        return

    if shape_type == "polygon":
        polygon = [_round_point(point) for point in points]
        draw.polygon(polygon, fill=255)
        return

    raise ValueError(f"Unsupported shape_type `{shape_type}` in label file")


def build_mask(label_path, output_path=None):
    with open(label_path, "r", encoding="utf-8") as file:
        label_data = json.load(file)

    width = int(label_data["imageWidth"])
    height = int(label_data["imageHeight"])

    mask = Image.new("L", (width, height), color=0)
    drawer = ImageDraw.Draw(mask)

    for shape in label_data.get("shapes", []):
        _draw_shape(drawer, shape)

    if output_path is None:
        label_path = Path(label_path)
        output_path = label_path.with_name(f"{label_path.stem}_mask.png")

    mask.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a binary mask from a labelme-style JSON annotation file."
    )
    parser.add_argument(
        "label_json",
        type=Path,
        help="Path to the label.json file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the output mask image"
    )
    args = parser.parse_args()

    if not args.label_json.is_file():
        raise FileNotFoundError(f"Label JSON not found: {args.label_json}")

    output_path = build_mask(args.label_json, args.output)
    print(f"Mask saved to {output_path}")


if __name__ == "__main__":
    main()
