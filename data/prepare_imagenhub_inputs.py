import argparse
import shutil
from pathlib import Path


def prepare_inputs(dataset_root: Path, output_root: Path) -> None:
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset directory `{dataset_root}` not found")

    input_dir = output_root / "input"
    token_dir = output_root / "token"
    input_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)

    for case_dir in sorted(dataset_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name
        rep_path = case_dir / "rep.png"
        token_path = case_dir / "token_rep.png"

        if not rep_path.is_file():
            raise FileNotFoundError(f"Missing rep.png in {case_dir}")
        if not token_path.is_file():
            raise FileNotFoundError(f"Missing token_rep.png in {case_dir}")

        shutil.copy2(rep_path, input_dir / f"{case_name}.png")
        shutil.copy2(token_path, token_dir / f"{case_name}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ImagenHub-style input/token folders from a custom dataset."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset containing numbered subfolders."
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Destination root directory for ImagenHub-style folders."
    )
    args = parser.parse_args()

    prepare_inputs(args.dataset_root, args.output_root)


if __name__ == "__main__":
    main()
