import argparse
import shutil
from pathlib import Path


def copy_tokens(dataset_root: Path, output_root: Path) -> None:
    add_dir = output_root / "add"
    rep_dir = output_root / "rep"
    add_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    for sample_dir in sorted(dataset_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        case_name = sample_dir.name

        add_source = sample_dir / "add.png"
        rep_source = sample_dir / "rep.png"

        if add_source.is_file():
            shutil.copy2(add_source, add_dir / f"{case_name}.png")
        if rep_source.is_file():
            shutil.copy2(rep_source, rep_dir / f"{case_name}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy token_add.png and token_rep.png into consolidated folders"
    )
    parser.add_argument("dataset_root", type=Path, help="Root directory containing numbered folders")
    parser.add_argument("output_root", type=Path, help="Destination directory for token_add/token_rep folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_tokens(args.dataset_root, args.output_root)


if __name__ == "__main__":
    main()
