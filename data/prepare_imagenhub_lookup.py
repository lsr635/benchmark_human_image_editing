import argparse
import json
from pathlib import Path


def build_lookup(dataset_root: Path, output_path: Path) -> None:
    lookup = {}
    for case_dir in sorted(dataset_root.iterdir()):
        if not case_dir.is_dir():
            continue
        label_path = case_dir / "label.json"
        if not label_path.is_file():
            continue
        with label_path.open("r", encoding="utf-8") as file:
            label_data = json.load(file)
        shapes = label_data.get("shapes", [])
        if not shapes:
            continue
        subject = shapes[0].get("label", "")
        lookup[f"{case_dir.name}.png"] = {"subject": subject}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(lookup, file, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate subject labels from dataset label.json files into ImagenHub-style lookup JSON."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to dataset folders")
    parser.add_argument("output_path", type=Path, help="Destination JSON file")
    args = parser.parse_args()

    build_lookup(args.dataset_root, args.output_path)


if __name__ == "__main__":
    main()
