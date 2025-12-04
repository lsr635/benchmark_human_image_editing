import argparse
import shutil
from pathlib import Path

REQUIRED_FILES = {
    "add": ["add.png", "label_mask.png", "label.json", "token_add.png"],
    "rep": ["rep.png", "label_mask.png", "label.json", "token_rep.png"],
}


def copy_subset(src_dir: Path, dst_dir: Path, file_list: list[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file_name in file_list:
        src_path = src_dir / file_name
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing `{file_name}` in {src_dir}")
        shutil.copy2(src_path, dst_dir / file_name)


def process_dataset(root: Path, output_root: Path) -> None:
    dataset_dirs = sorted(
        entry for entry in root.iterdir() if entry.is_dir()
    )

    add_root = output_root / "dataset_add"
    rep_root = output_root / "dataset_rep"
    add_root.mkdir(parents=True, exist_ok=True)
    rep_root.mkdir(parents=True, exist_ok=True)

    for case_dir in dataset_dirs:
        case_name = case_dir.name
        copy_subset(case_dir, add_root / case_name, REQUIRED_FILES["add"])
        copy_subset(case_dir, rep_root / case_name, REQUIRED_FILES["rep"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create dataset_add and dataset_rep subsets from a dataset folder."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset directory containing numbered folders."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional destination root. Defaults to the parent of dataset_root."
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset_root `{dataset_root}` does not exist or is not a directory")

    output_root = args.output_root if args.output_root is not None else dataset_root.parent
    output_root = Path(output_root)

    process_dataset(dataset_root, output_root)


if __name__ == "__main__":
    main()
