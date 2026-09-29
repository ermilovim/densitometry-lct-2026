from __future__ import annotations

import argparse
import csv
import random
import zipfile
from pathlib import Path
from typing import Any

from prepare_artifacts import DEFAULT_TRAIN_ZIP, image_from_dicom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANDMARK_CSV = ROOT / "artifacts" / "landmark_all.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "keypoint_dataset"

SPINE_KEYS = [
    "spine_th12_center",
    "spine_l1_center",
    "spine_l2_center",
    "spine_l3_center",
    "spine_l4_center",
    "spine_l5_center",
    "spine_left_iliac_crest",
    "spine_right_iliac_crest",
]

HIP_KEYS = [
    "hip_greater_trochanter_top_edge",
    "hip_greater_trochanter_lateral_edge",
    "hip_lesser_trochanter_upper",
    "hip_lesser_trochanter_tip",
    "hip_lesser_trochanter_lower",
    "hip_ischium_edge",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def point(row: dict[str, str], key: str) -> tuple[float, float] | None:
    x = row.get(f"{key}_x", "")
    y = row.get(f"{key}_y", "")
    if x == "" or y == "":
        return None
    try:
        return float(x), float(y)
    except ValueError:
        return None


def hip_label(row: dict[str, str], suffix: str) -> str:
    if row["anatomical_region"] == "right_hip":
        return row.get(f"right_hip_{suffix}", "")
    if row["anatomical_region"] == "left_hip":
        return row.get(f"left_hip_{suffix}", "")
    return ""


def split_by_study(rows: list[dict[str, str]], val_fraction: float, seed: int) -> dict[str, str]:
    studies = sorted({row["study_folder"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(studies)
    n_val = max(1, round(len(studies) * val_fraction))
    val_studies = set(studies[:n_val])
    return {study: "val" if study in val_studies else "train" for study in studies}


def point_columns(keys: list[str]) -> list[str]:
    cols: list[str] = []
    for key in keys:
        cols.extend([f"{key}_x", f"{key}_y", f"{key}_x_norm", f"{key}_y_norm", f"{key}_visible"])
    return cols


def add_points(out: dict[str, Any], row: dict[str, str], keys: list[str], width: int, height: int) -> None:
    for key in keys:
        pt = point(row, key)
        if pt is None:
            out[f"{key}_x"] = ""
            out[f"{key}_y"] = ""
            out[f"{key}_x_norm"] = ""
            out[f"{key}_y_norm"] = ""
            out[f"{key}_visible"] = 0
            continue
        x, y = pt
        out[f"{key}_x"] = f"{x:.3f}"
        out[f"{key}_y"] = f"{y:.3f}"
        out[f"{key}_x_norm"] = f"{x / width:.6f}" if width else ""
        out[f"{key}_y_norm"] = f"{y / height:.6f}" if height else ""
        out[f"{key}_visible"] = 1


def export_dataset(
    landmark_csv: Path,
    train_zip: Path,
    out_dir: Path,
    val_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows = [row for row in read_csv(landmark_csv) if row.get("annotation_status") == "done"]
    split_by_uid = split_by_study(rows, val_fraction=val_fraction, seed=seed)
    annotations: list[dict[str, Any]] = []

    with zipfile.ZipFile(train_zip) as zf:
        for row in rows:
            split = split_by_uid[row["study_folder"]]
            region = row["anatomical_region"]
            image_name = f"{int(row['pilot_id']):04d}_{row['image_number']}_{region}.png"
            image_rel = Path("images") / split / image_name
            image_path = out_dir / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)

            raw = zf.read(row["member"])
            image, _meta = image_from_dicom(raw)
            image.save(image_path)

            width, height = image.size
            item: dict[str, Any] = {
                "split": split,
                "image_path": image_rel.as_posix(),
                "pilot_id": row["pilot_id"],
                "image_number": row["image_number"],
                "study_folder": row["study_folder"],
                "anatomical_region": region,
                "member": row["member"],
                "filename": row["filename"],
                "width": width,
                "height": height,
                "quality_class": row.get("quality_class", ""),
                "violation_type": row.get("violation_type", ""),
                "spine_positioning": row.get("spine_positioning", ""),
                "spine_axis": row.get("spine_axis", ""),
                "spine_artifact": row.get("spine_artifact", ""),
                "hip_positioning_rotation": hip_label(row, "positioning_rotation"),
                "hip_roi": hip_label(row, "roi"),
                "annotation_notes": row.get("annotation_notes", ""),
            }
            add_points(item, row, SPINE_KEYS + HIP_KEYS, width, height)
            annotations.append(item)
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-csv", type=Path, default=DEFAULT_LANDMARK_CSV)
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = export_dataset(
        landmark_csv=args.landmark_csv,
        train_zip=args.train_zip,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    base_cols = [
        "split",
        "image_path",
        "pilot_id",
        "image_number",
        "study_folder",
        "anatomical_region",
        "member",
        "filename",
        "width",
        "height",
        "quality_class",
        "violation_type",
        "spine_positioning",
        "spine_axis",
        "spine_artifact",
        "hip_positioning_rotation",
        "hip_roi",
        "annotation_notes",
    ]
    fieldnames = base_cols + point_columns(SPINE_KEYS + HIP_KEYS)
    write_csv(args.out_dir / "annotations.csv", annotations, fieldnames=fieldnames)

    counts: dict[tuple[str, str], int] = {}
    for row in annotations:
        key = (str(row["split"]), str(row["anatomical_region"]))
        counts[key] = counts.get(key, 0) + 1
    print(f"Rows: {len(annotations)}")
    for key, count in sorted(counts.items()):
        print(f"{key[0]} {key[1]}: {count}")
    print(f"Wrote: {args.out_dir / 'annotations.csv'}")


if __name__ == "__main__":
    main()
