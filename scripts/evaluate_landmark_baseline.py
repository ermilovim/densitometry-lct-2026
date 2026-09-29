from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS_CSV = ROOT / "artifacts" / "keypoint_dataset" / "annotations.csv"
DEFAULT_EVAL_CSV = ROOT / "artifacts" / "landmark_baseline_eval.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "landmark_baseline_report.csv"
DEFAULT_MODEL_JSON = ROOT / "artifacts" / "landmark_mean_model.json"

PIXEL_SPACING_X_MM = 0.60
PIXEL_SPACING_Y_MM = 1.05

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

REGION_KEYS = {
    "spine": SPINE_KEYS,
    "left_hip": HIP_KEYS,
    "right_hip": HIP_KEYS,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def train_mean_model(rows: list[dict[str, str]], min_visibility: float) -> dict[str, Any]:
    model: dict[str, Any] = {
        "model_type": "mean_normalized_landmarks",
        "min_visibility": min_visibility,
        "pixel_spacing_x_mm": PIXEL_SPACING_X_MM,
        "pixel_spacing_y_mm": PIXEL_SPACING_Y_MM,
        "regions": {},
    }
    train_rows = [row for row in rows if row["split"] == "train"]
    for region, keys in REGION_KEYS.items():
        region_rows = [row for row in train_rows if row["anatomical_region"] == region]
        region_model: dict[str, Any] = {"n_train": len(region_rows), "points": {}}
        for key in keys:
            xs = [as_float(row.get(f"{key}_x_norm", "")) for row in region_rows if row.get(f"{key}_visible") == "1"]
            ys = [as_float(row.get(f"{key}_y_norm", "")) for row in region_rows if row.get(f"{key}_visible") == "1"]
            xs = [value for value in xs if value is not None]
            ys = [value for value in ys if value is not None]
            visibility = len(xs) / len(region_rows) if region_rows else 0.0
            if xs and ys and visibility >= min_visibility:
                region_model["points"][key] = {
                    "x_norm": sum(xs) / len(xs),
                    "y_norm": sum(ys) / len(ys),
                    "visibility": visibility,
                    "predict_visible": True,
                }
            else:
                region_model["points"][key] = {
                    "x_norm": None,
                    "y_norm": None,
                    "visibility": visibility,
                    "predict_visible": False,
                }
        model["regions"][region] = region_model
    return model


def predict_point(row: dict[str, str], model: dict[str, Any], key: str) -> tuple[float, float] | None:
    region = row["anatomical_region"]
    point = model["regions"].get(region, {}).get("points", {}).get(key)
    if not point or not point["predict_visible"]:
        return None
    width = float(row["width"])
    height = float(row["height"])
    return point["x_norm"] * width, point["y_norm"] * height


def point_error_mm(pred: tuple[float, float], true_x: float, true_y: float) -> tuple[float, float, float]:
    dx_mm = (pred[0] - true_x) * PIXEL_SPACING_X_MM
    dy_mm = (pred[1] - true_y) * PIXEL_SPACING_Y_MM
    return dx_mm, dy_mm, math.hypot(dx_mm, dy_mm)


def evaluate_model(rows: list[dict[str, str]], model: dict[str, Any]) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        region = row["anatomical_region"]
        keys = REGION_KEYS.get(region, [])
        for key in keys:
            true_x = as_float(row.get(f"{key}_x", ""))
            true_y = as_float(row.get(f"{key}_y", ""))
            visible = row.get(f"{key}_visible", "") == "1"
            pred = predict_point(row, model, key)
            if not visible:
                continue
            if pred is None or true_x is None or true_y is None:
                eval_rows.append(
                    {
                        "split": row["split"],
                        "anatomical_region": region,
                        "point": key,
                        "pilot_id": row["pilot_id"],
                        "image_number": row["image_number"],
                        "is_predicted": 0,
                        "dx_mm": "",
                        "dy_mm": "",
                        "error_mm": "",
                    }
                )
                continue
            dx_mm, dy_mm, error_mm = point_error_mm(pred, true_x, true_y)
            eval_rows.append(
                {
                    "split": row["split"],
                    "anatomical_region": region,
                    "point": key,
                    "pilot_id": row["pilot_id"],
                    "image_number": row["image_number"],
                    "is_predicted": 1,
                    "dx_mm": f"{dx_mm:.3f}",
                    "dy_mm": f"{dy_mm:.3f}",
                    "error_mm": f"{error_mm:.3f}",
                }
            )
    return eval_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, round((len(vals) - 1) * p))
    return vals[idx]


def build_report(eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        grouped[(row["split"], row["anatomical_region"], row["point"])].append(row)

    report: list[dict[str, Any]] = []
    for (split, region, point), rows in sorted(grouped.items()):
        predicted = [row for row in rows if row["is_predicted"] == 1]
        errors = [float(row["error_mm"]) for row in predicted]
        report.append(
            {
                "split": split,
                "anatomical_region": region,
                "point": point,
                "n_visible": len(rows),
                "n_predicted": len(predicted),
                "mean_error_mm": f"{mean(errors):.3f}" if errors else "",
                "median_error_mm": f"{percentile(errors, 0.5):.3f}" if errors else "",
                "p90_error_mm": f"{percentile(errors, 0.9):.3f}" if errors else "",
            }
        )

    for split in sorted({row["split"] for row in eval_rows}):
        rows = [row for row in eval_rows if row["split"] == split and row["is_predicted"] == 1]
        errors = [float(row["error_mm"]) for row in rows]
        report.append(
            {
                "split": split,
                "anatomical_region": "ALL",
                "point": "ALL",
                "n_visible": sum(1 for row in eval_rows if row["split"] == split),
                "n_predicted": len(rows),
                "mean_error_mm": f"{mean(errors):.3f}" if errors else "",
                "median_error_mm": f"{percentile(errors, 0.5):.3f}" if errors else "",
                "p90_error_mm": f"{percentile(errors, 0.9):.3f}" if errors else "",
            }
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.annotations_csv)
    model = train_mean_model(rows, min_visibility=args.min_visibility)
    eval_rows = evaluate_model(rows, model)
    report = build_report(eval_rows)
    write_csv(args.eval_csv, eval_rows)
    write_csv(
        args.report_csv,
        report,
        fieldnames=[
            "split",
            "anatomical_region",
            "point",
            "n_visible",
            "n_predicted",
            "mean_error_mm",
            "median_error_mm",
            "p90_error_mm",
        ],
    )
    args.model_json.parent.mkdir(parents=True, exist_ok=True)
    args.model_json.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rows: {len(rows)}")
    print(f"Eval rows: {len(eval_rows)}")
    for row in report:
        if row["point"] == "ALL":
            print(
                f"{row['split']} ALL: n={row['n_predicted']}/{row['n_visible']} "
                f"mean={row['mean_error_mm']}mm median={row['median_error_mm']}mm p90={row['p90_error_mm']}mm"
            )
    print(f"Wrote: {args.eval_csv}")
    print(f"Wrote: {args.report_csv}")
    print(f"Wrote: {args.model_json}")


if __name__ == "__main__":
    main()
