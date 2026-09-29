from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYPOINT_DIR = ROOT / "artifacts" / "keypoint_dataset"
DEFAULT_ANNOTATIONS_CSV = DEFAULT_KEYPOINT_DIR / "annotations.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "spine_positioning_dataset"


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
    return f"img{int(row['image_number']):03d}_pilot{int(row['pilot_id']):04d}_{row['study_folder']}_{suffix}.png"


def both_iliac_crests_visible(row: dict[str, str]) -> bool:
    return (
        as_int(row.get("spine_left_iliac_crest_visible", "")) == 1
        and as_int(row.get("spine_right_iliac_crest_visible", "")) == 1
    )


def is_clean_positive_source(row: dict[str, str]) -> bool:
    return (
        as_int(row.get("spine_positioning", "")) == 0
        and as_int(row.get("spine_axis", "")) == 0
        and as_int(row.get("spine_artifact", "")) == 0
        and both_iliac_crests_visible(row)
    )


def crop_removes_both_iliac_crests(row: dict[str, str], crop_fraction: float, margin_norm: float) -> bool:
    left_y = row.get("spine_left_iliac_crest_y_norm", "")
    right_y = row.get("spine_right_iliac_crest_y_norm", "")
    if left_y == "" or right_y == "":
        return False

    crop_bottom_y = 1.0 - crop_fraction
    top_iliac_y = min(float(left_y), float(right_y))
    return crop_bottom_y <= top_iliac_y - margin_norm


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
        "positioning_bad": source.get("spine_positioning", ""),
        "iliac_visible": int(both_iliac_crests_visible(source)),
        "spine_positioning": source.get("spine_positioning", ""),
        "spine_axis": source.get("spine_axis", ""),
        "spine_artifact": source.get("spine_artifact", ""),
    }


def save_bottom_crop(source: Path, dest: Path, crop_fraction: float) -> None:
    image = Image.open(source).convert("L")
    width, height = image.size
    crop_height = max(1, round(height * (1.0 - crop_fraction)))
    cropped = image.crop((0, 0, width, crop_height)).resize((width, height), Image.Resampling.BILINEAR)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dest)


def build_dataset(
    keypoint_dir: Path,
    annotations_csv: Path,
    out_dir: Path,
    crop_fractions: list[float],
    crop_margin_norm: float,
) -> list[dict[str, Any]]:
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

        if row["split"] != "train" or not is_clean_positive_source(row):
            continue

        for crop_fraction in crop_fractions:
            if not crop_removes_both_iliac_crests(row, crop_fraction, crop_margin_norm):
                continue

            crop_name = clean_name(row, f"bottom_crop_{int(crop_fraction * 100):02d}")
            crop_rel = Path("images") / "train" / crop_name
            save_bottom_crop(source, out_dir / crop_rel, crop_fraction)
            item = base_row(row, crop_rel, source_rel.as_posix(), f"bottom_crop_{crop_fraction:.2f}", 1)
            item["positioning_bad"] = 1
            item["iliac_visible"] = 0
            item["spine_positioning"] = 1
            rows.append(item)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoint-dir", type=Path, default=DEFAULT_KEYPOINT_DIR)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--crop-fractions", type=float, nargs="+", default=[0.18, 0.25, 0.32, 0.40])
    parser.add_argument("--crop-margin-norm", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    rows = build_dataset(
        args.keypoint_dir,
        args.annotations_csv,
        args.out_dir,
        args.crop_fractions,
        args.crop_margin_norm,
    )
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
        "positioning_bad",
        "iliac_visible",
        "spine_positioning",
        "spine_axis",
        "spine_artifact",
    ]
    write_csv(args.out_dir / "annotations.csv", rows, fieldnames)
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {args.out_dir / 'annotations.csv'}")


if __name__ == "__main__":
    main()
