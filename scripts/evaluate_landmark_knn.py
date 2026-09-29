from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "artifacts" / "keypoint_dataset"
DEFAULT_ANNOTATIONS_CSV = DEFAULT_DATASET_DIR / "annotations.csv"
DEFAULT_EVAL_CSV = ROOT / "artifacts" / "landmark_knn_eval.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "landmark_knn_report.csv"
DEFAULT_MODEL_NPZ = ROOT / "artifacts" / "landmark_knn_model.npz"
DEFAULT_MODEL_JSON = ROOT / "artifacts" / "landmark_knn_model.json"

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


def image_vector(path: Path, size: int) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    arr = arr / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    if std > 1e-6:
        arr = (arr - mean) / std
    else:
        arr = arr - mean
    return arr.reshape(-1)


def load_vectors(rows: list[dict[str, str]], dataset_dir: Path, size: int) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for row in rows:
        vectors[row["pilot_id"]] = image_vector(dataset_dir / row["image_path"], size)
    return vectors


def nearest_rows(
    query: dict[str, str],
    candidates: list[dict[str, str]],
    vectors: dict[str, np.ndarray],
    exclude_self: bool,
    k: int,
) -> list[tuple[dict[str, str], float]]:
    qv = vectors[query["pilot_id"]]
    ranked: list[tuple[dict[str, str], float]] = []
    for candidate in candidates:
        if exclude_self and candidate["pilot_id"] == query["pilot_id"]:
            continue
        diff = qv - vectors[candidate["pilot_id"]]
        dist = float(np.sqrt(np.mean(diff * diff)))
        ranked.append((candidate, dist))
    ranked.sort(key=lambda item: item[1])
    return ranked[:k]


def point_from_neighbors(
    neighbors: list[tuple[dict[str, str], float]],
    target: dict[str, str],
    key: str,
) -> tuple[float, float] | None:
    values: list[tuple[float, float, float]] = []
    for neighbor, dist in neighbors:
        if neighbor.get(f"{key}_visible") != "1":
            continue
        x_norm = as_float(neighbor.get(f"{key}_x_norm", ""))
        y_norm = as_float(neighbor.get(f"{key}_y_norm", ""))
        if x_norm is None or y_norm is None:
            continue
        weight = 1.0 / max(dist, 1e-6)
        values.append((x_norm, y_norm, weight))
    if not values:
        return None
    total_weight = sum(weight for _x, _y, weight in values)
    x_norm = sum(x * weight for x, _y, weight in values) / total_weight
    y_norm = sum(y * weight for _x, y, weight in values) / total_weight
    return x_norm * float(target["width"]), y_norm * float(target["height"])


def point_error_mm(pred: tuple[float, float], true_x: float, true_y: float) -> tuple[float, float, float]:
    dx_mm = (pred[0] - true_x) * PIXEL_SPACING_X_MM
    dy_mm = (pred[1] - true_y) * PIXEL_SPACING_Y_MM
    return dx_mm, dy_mm, math.hypot(dx_mm, dy_mm)


def evaluate(rows: list[dict[str, str]], vectors: dict[str, np.ndarray], k: int) -> list[dict[str, Any]]:
    train_by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        all_by_region[row["anatomical_region"]].append(row)
        if row["split"] == "train":
            train_by_region[row["anatomical_region"]].append(row)

    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        region = row["anatomical_region"]
        if region not in REGION_KEYS:
            continue
        candidates = train_by_region[region] if row["split"] == "val" else all_by_region[region]
        neighbors = nearest_rows(row, candidates, vectors, exclude_self=row["split"] == "train", k=k)
        first_neighbor = neighbors[0][0] if neighbors else None
        first_distance = neighbors[0][1] if neighbors else None
        for key in REGION_KEYS[region]:
            true_x = as_float(row.get(f"{key}_x", ""))
            true_y = as_float(row.get(f"{key}_y", ""))
            if row.get(f"{key}_visible") != "1":
                continue
            pred = point_from_neighbors(neighbors, row, key)
            base = {
                "split": row["split"],
                "anatomical_region": region,
                "point": key,
                "pilot_id": row["pilot_id"],
                "image_number": row["image_number"],
                "neighbor_pilot_id": "" if first_neighbor is None else first_neighbor["pilot_id"],
                "neighbor_image_number": "" if first_neighbor is None else first_neighbor["image_number"],
                "neighbor_distance": "" if first_distance is None else f"{first_distance:.6f}",
            }
            if pred is None or true_x is None or true_y is None:
                eval_rows.append({**base, "is_predicted": 0, "dx_mm": "", "dy_mm": "", "error_mm": ""})
                continue
            dx_mm, dy_mm, error_mm = point_error_mm(pred, true_x, true_y)
            eval_rows.append(
                {
                    **base,
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


def save_model(
    rows: list[dict[str, str]],
    vectors: dict[str, np.ndarray],
    model_npz: Path,
    model_json: Path,
    size: int,
    k: int,
    train_all: bool,
) -> None:
    train_rows = rows if train_all else [row for row in rows if row["split"] == "train"]
    pilot_ids = [row["pilot_id"] for row in train_rows]
    matrix = np.stack([vectors[pilot_id] for pilot_id in pilot_ids], axis=0)
    model_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_npz, image_size=np.array([size], dtype=np.int32), pilot_ids=np.array(pilot_ids), vectors=matrix)
    metadata = {
        "model_type": "nearest_neighbor_landmarks",
        "image_size": size,
        "k": k,
        "train_all": train_all,
        "pixel_spacing_x_mm": PIXEL_SPACING_X_MM,
        "pixel_spacing_y_mm": PIXEL_SPACING_Y_MM,
        "rows": train_rows,
    }
    model_json.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--model-npz", type=Path, default=DEFAULT_MODEL_NPZ)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--train-all", action="store_true", help="Save train+val rows into the final kNN landmark model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.annotations_csv)
    vectors = load_vectors(rows, args.dataset_dir, args.size)
    eval_rows = evaluate(rows, vectors, k=args.k)
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
    save_model(rows, vectors, args.model_npz, args.model_json, args.size, args.k, args.train_all)
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
    print(f"Wrote: {args.model_npz}")
    print(f"Wrote: {args.model_json}")
    if args.train_all:
        print("Saved kNN model with previous train+val rows.")


if __name__ == "__main__":
    main()
