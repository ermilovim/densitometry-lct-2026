from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT / "artifacts" / "inference_skeleton.csv"
DEFAULT_MODEL_JSON = ROOT / "artifacts" / "landmark_mean_model.json"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "inference_landmark_predictions.csv"

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

ALL_KEYS = SPINE_KEYS + HIP_KEYS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def point_columns() -> list[str]:
    cols: list[str] = []
    for key in ALL_KEYS:
        cols.extend([f"{key}_x", f"{key}_y", f"{key}_visible"])
    return cols


def predict_row(row: dict[str, str], model: dict[str, Any]) -> dict[str, Any]:
    region = row.get("anatomical_region", "")
    width = float(row["columns"]) if row.get("columns") else 0.0
    height = float(row["rows"]) if row.get("rows") else 0.0
    region_model = model.get("regions", {}).get(region, {})
    points = region_model.get("points", {})

    out: dict[str, Any] = {
        "input_source": row.get("input_source", ""),
        "study_key": row.get("study_key", ""),
        "member": row.get("member", ""),
        "filename": row.get("filename", ""),
        "anatomical_region": region,
        "is_canonical_for_region": row.get("is_canonical_for_region", ""),
        "landmark_model": model.get("model_type", ""),
        "landmark_status": "",
    }

    for key in ALL_KEYS:
        point = points.get(key, {})
        if point.get("predict_visible") and width > 0 and height > 0:
            out[f"{key}_x"] = f"{point['x_norm'] * width:.3f}"
            out[f"{key}_y"] = f"{point['y_norm'] * height:.3f}"
            out[f"{key}_visible"] = 1
        else:
            out[f"{key}_x"] = ""
            out[f"{key}_y"] = ""
            out[f"{key}_visible"] = 0

    if not points:
        out["landmark_status"] = "unsupported_region"
    else:
        out["landmark_status"] = "predicted"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    out_rows = [predict_row(row, model) for row in rows]
    fieldnames = [
        "input_source",
        "study_key",
        "member",
        "filename",
        "anatomical_region",
        "is_canonical_for_region",
        "landmark_model",
        "landmark_status",
        *point_columns(),
    ]
    write_csv(args.out_csv, out_rows, fieldnames)
    status_counts: dict[str, int] = {}
    for row in out_rows:
        status_counts[row["landmark_status"]] = status_counts.get(row["landmark_status"], 0) + 1
    print(f"Rows: {len(out_rows)}")
    for key, value in sorted(status_counts.items()):
        print(f"{key}: {value}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
