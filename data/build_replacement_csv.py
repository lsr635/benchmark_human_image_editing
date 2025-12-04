import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


def load_label(label_path: Path) -> Dict:
    with label_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_subject(label_data: Dict) -> str:
    shapes = label_data.get("shapes", [])
    if not shapes:
        return "subject"
    return shapes[0].get("label", "subject") or "subject"


def build_row(case_dir: Path,
              output_root: Path,
              rel_subject_root: Path,
              rel_source_root: Path,
              default_prompt: Optional[str] = None) -> Optional[Dict[str, str]]:
    label_path = case_dir / "label.json"
    if not label_path.is_file():
        print(f"[Skip] {case_dir.name}: missing label.json")
        return None

    label_data = load_label(label_path)
    subject = resolve_subject(label_data)

    subject_img = case_dir / "token_rep.png"
    source_img = case_dir / "rep.png"

    if not subject_img.is_file() or not source_img.is_file():
        print(f"[Skip] {case_dir.name}: missing rep.png or token_rep.png")
        return None

    prompt_text = f"photo of a {subject}"

    out_subject = rel_subject_root / f"{case_dir.name}.png"
    out_source = rel_source_root / f"{case_dir.name}.png"

    return {
        "id": case_dir.name,
        "subject_names": subject,
        "subject_image_path": str(out_subject).replace('\\', '/'),
        "source_image_path": str(out_source).replace('\\', '/'),
        "source_prompt": prompt_text,
        "target_prompt": prompt_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate replacement.csv from ACE++ style dataset_rep folders."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to dataset_rep directory containing numbered folders")
    parser.add_argument("output_csv", type=Path, help="Destination CSV path")
    parser.add_argument("--subject-root", type=Path, default=Path("replacement/subject_image"),
                        help="Relative subject image root used in CSV paths")
    parser.add_argument("--source-root", type=Path, default=Path("replacement/source_image"),
                        help="Relative source image root used in CSV paths")
    parser.add_argument("--default-prompt", type=str, default=None,
                        help="Fallback prompt text when prompt_rep.txt missing")
    args = parser.parse_args()

    rows = []
    for case_dir in sorted(p for p in args.dataset_root.iterdir() if p.is_dir()):
        row = build_row(case_dir,
            args.dataset_root,
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
        writer.writerow(["id", "subject_names", "subject_image_path", "source_image_path", "source_prompt", "target_prompt"])
        for row in rows:
            writer.writerow([
                row["id"],
                row["subject_names"],
                row["subject_image_path"],
                row["source_image_path"],
                row["source_prompt"],
                row["target_prompt"],
            ])
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
