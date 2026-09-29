from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYPOINT_DIR = ROOT / "artifacts" / "keypoint_dataset"
DEFAULT_ANNOTATIONS_CSV = DEFAULT_KEYPOINT_DIR / "annotations.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "spine_artifact_dataset"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: str) -> int:
    return 0 if value == "" else int(float(value))


def clean_name(row: dict[str, str], suffix: str) -> str:
    label = "artifact" if as_int(row.get("spine_artifact", "")) == 1 else "clean"
    return f"img{int(row['image_number']):03d}_pilot{int(row['pilot_id']):04d}_{label}_{suffix}.png"


def base_row(
    source: dict[str, str],
    image_path: Path,
    source_image_path: str,
    augmentation: str,
    synthetic: int,
) -> dict[str, Any]:
    return {
        "split": source["split"],
        "image_path": image_path.as_posix(),
        "source_image_path": source_image_path,
        "augmentation": augmentation,
        "synthetic": synthetic,
        "pilot_id": source["pilot_id"],
        "image_number": source["image_number"],
        "study_folder": source["study_folder"],
        "filename": source["filename"],
        "artifact_bad": source.get("spine_artifact", ""),
        "spine_artifact": source.get("spine_artifact", ""),
        "spine_positioning": source.get("spine_positioning", ""),
        "spine_axis": source.get("spine_axis", ""),
    }


def save_horizontal_mirror(source: Path, dest: Path) -> None:
    image = Image.open(source).convert("L")
    mirrored = ImageOps.mirror(image)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mirrored.save(dest)


def build_dataset(keypoint_dir: Path, annotations_csv: Path, out_dir: Path) -> list[dict[str, Any]]:
    annotations = [row for row in read_csv(annotations_csv) if row["anatomical_region"] == "spine"]
    rows: list[dict[str, Any]] = []

    for row in annotations:
        source_rel = Path(row["image_path"])
        source = keypoint_dir / source_rel
        original_name = clean_name(row, "original")
        original_rel = Path("images") / row["split"] / original_name
        original_dest = out_dir / original_rel
        original_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, original_dest)
        rows.append(base_row(row, original_rel, source_rel.as_posix(), "original", 0))

        if row["split"] == "train" and as_int(row.get("spine_artifact", "")) == 1:
            mirror_name = clean_name(row, "hflip")
            mirror_rel = Path("images") / "train" / mirror_name
            save_horizontal_mirror(source, out_dir / mirror_rel)
            rows.append(base_row(row, mirror_rel, source_rel.as_posix(), "horizontal_flip", 1))

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoint-dir", type=Path, default=DEFAULT_KEYPOINT_DIR)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    rows = build_dataset(args.keypoint_dir, args.annotations_csv, args.out_dir)
    fieldnames = [
        "split",
        "image_path",
        "source_image_path",
        "augmentation",
        "synthetic",
        "pilot_id",
        "image_number",
        "study_folder",
        "filename",
        "artifact_bad",
        "spine_artifact",
        "spine_positioning",
        "spine_axis",
    ]
    write_csv(args.out_dir / "annotations.csv", rows, fieldnames)
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {args.out_dir / 'annotations.csv'}")


if __name__ == "__main__":
    main()
