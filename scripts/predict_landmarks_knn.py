from __future__ import annotations

import argparse
import csv
import time
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from prepare_artifacts import image_from_dicom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT / "artifacts" / "inference_skeleton.csv"
DEFAULT_DICOM_INPUT = ROOT / "examples" / "example_input.zip"
DEFAULT_MODEL_NPZ = ROOT / "artifacts" / "landmark_knn_model.npz"
DEFAULT_MODEL_JSON = ROOT / "artifacts" / "landmark_knn_model.json"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "inference_landmark_predictions_knn.csv"

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


def image_vector(image: Image.Image, size: int) -> np.ndarray:
    arr = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    arr = arr / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    if std > 1e-6:
        arr = (arr - mean) / std
    else:
        arr = arr - mean
    return arr.reshape(-1)


def read_member_raw(dicom_input: Path, member: str) -> bytes:
    if dicom_input.is_file() and dicom_input.suffix.lower() == ".zip":
        with zipfile.ZipFile(dicom_input) as zf:
            return zf.read(member)
    path = dicom_input / member
    return path.read_bytes()


def load_model(model_npz: Path, model_json: Path) -> tuple[int, int, np.ndarray, list[dict[str, str]]]:
    arrays = np.load(model_npz)
    metadata = json.loads(model_json.read_text(encoding="utf-8"))
    return int(arrays["image_size"][0]), int(metadata.get("k", 5)), arrays["vectors"].astype(np.float32), metadata["rows"]


def nearest_indices(vector: np.ndarray, matrix: np.ndarray, rows: list[dict[str, str]], region: str, k: int) -> list[tuple[int, float]]:
    ranked: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        if row["anatomical_region"] != region:
            continue
        diff = vector - matrix[idx]
        dist = float(np.sqrt(np.mean(diff * diff)))
        ranked.append((idx, dist))
    ranked.sort(key=lambda item: item[1])
    return ranked[:k]


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def point_from_neighbors(
    neighbors: list[tuple[int, float]],
    model_rows: list[dict[str, str]],
    target_width: float,
    target_height: float,
    key: str,
) -> tuple[float, float] | None:
    values: list[tuple[float, float, float]] = []
    for idx, dist in neighbors:
        row = model_rows[idx]
        if row.get(f"{key}_visible") != "1":
            continue
        x_norm = as_float(row.get(f"{key}_x_norm", ""))
        y_norm = as_float(row.get(f"{key}_y_norm", ""))
        if x_norm is None or y_norm is None:
            continue
        weight = 1.0 / max(dist, 1e-6)
        values.append((x_norm, y_norm, weight))
    if not values:
        return None
    total_weight = sum(weight for _x, _y, weight in values)
    x_norm = sum(x * weight for x, _y, weight in values) / total_weight
    y_norm = sum(y * weight for _x, y, weight in values) / total_weight
    return x_norm * target_width, y_norm * target_height


def predict_row(
    row: dict[str, str],
    dicom_input: Path,
    image_size: int,
    k: int,
    matrix: np.ndarray,
    model_rows: list[dict[str, str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "input_source": row.get("input_source", ""),
        "study_key": row.get("study_key", ""),
        "member": row.get("member", ""),
        "filename": row.get("filename", ""),
        "anatomical_region": row.get("anatomical_region", ""),
        "is_canonical_for_region": row.get("is_canonical_for_region", ""),
        "landmark_model": "nearest_neighbor_landmarks",
        "landmark_status": "",
        "neighbor_pilot_ids": "",
    }
    for key in ALL_KEYS:
        out[f"{key}_x"] = ""
        out[f"{key}_y"] = ""
        out[f"{key}_visible"] = 0

    region = row.get("anatomical_region", "")
    if region not in {"spine", "left_hip", "right_hip"}:
        out["landmark_status"] = "unsupported_region"
        return out

    try:
        image, _meta = image_from_dicom(read_member_raw(dicom_input, row["member"]))
    except Exception as exc:
        out["landmark_status"] = f"render_failed: {exc}"
        return out

    vector = image_vector(image, image_size)
    neighbors = nearest_indices(vector, matrix, model_rows, region, k)
    out["neighbor_pilot_ids"] = ";".join(model_rows[idx]["pilot_id"] for idx, _dist in neighbors)
    keys = SPINE_KEYS if region == "spine" else HIP_KEYS
    for key in keys:
        point = point_from_neighbors(neighbors, model_rows, image.width, image.height, key)
        if point is None:
            continue
        out[f"{key}_x"] = f"{point[0]:.3f}"
        out[f"{key}_y"] = f"{point[1]:.3f}"
        out[f"{key}_visible"] = 1
    out["landmark_status"] = "predicted"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--dicom-input", type=Path, default=DEFAULT_DICOM_INPUT)
    parser.add_argument("--model-npz", type=Path, default=DEFAULT_MODEL_NPZ)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_size, k, matrix, model_rows = load_model(args.model_npz, args.model_json)
    rows = read_csv(args.input_csv)
    out_rows = []
    for row in rows:
        started = time.perf_counter()
        result = predict_row(row, args.dicom_input, image_size, k, matrix, model_rows)
        result["processing_seconds"] = f"{time.perf_counter() - started:.6f}"
        out_rows.append(result)
    fieldnames = [
        "input_source",
        "study_key",
        "member",
        "filename",
        "anatomical_region",
        "is_canonical_for_region",
        "landmark_model",
        "landmark_status",
        "neighbor_pilot_ids",
        "processing_seconds",
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
