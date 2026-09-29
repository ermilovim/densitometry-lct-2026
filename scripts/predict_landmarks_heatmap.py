from __future__ import annotations

import argparse
import csv
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from prepare_artifacts import image_from_dicom
from train_landmark_heatmap import HIP_KEYS, SPINE_KEYS, TinyUNet, predict_points


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT / "artifacts" / "inference_skeleton.csv"
DEFAULT_DICOM_INPUT = ROOT / "examples" / "example_input.zip"
DEFAULT_HIP_MODEL = ROOT / "artifacts" / "landmark_heatmap" / "hip" / "best.pt"
DEFAULT_SPINE_MODEL = ROOT / "artifacts" / "landmark_heatmap" / "spine" / "best.pt"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "inference_landmark_predictions_heatmap.csv"

ALL_KEYS = SPINE_KEYS + HIP_KEYS
LESSER_KEYS = [
    "hip_lesser_trochanter_upper",
    "hip_lesser_trochanter_tip",
    "hip_lesser_trochanter_lower",
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


def point_columns() -> list[str]:
    cols: list[str] = []
    for key in ALL_KEYS:
        cols.extend([f"{key}_x", f"{key}_y", f"{key}_visible"])
    return cols


def read_member_raw(dicom_input: Path, member: str) -> bytes:
    if dicom_input.is_file() and dicom_input.suffix.lower() == ".zip":
        with zipfile.ZipFile(dicom_input) as zf:
            return zf.read(member)
    return (dicom_input / member).read_bytes()


def image_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    resized = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - float(arr.mean())) / max(float(arr.std()), 1e-6)
    return torch.from_numpy(arr[None, None, :, :].astype(np.float32)).to(device)


def distinct_topk_points(heatmap: torch.Tensor, k: int = 12, min_distance: float = 4.0) -> list[tuple[float, float, float]]:
    height, width = heatmap.shape
    values, indices = torch.topk(heatmap.reshape(-1), k=min(k * 8, heatmap.numel()))
    points: list[tuple[float, float, float]] = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        y = float(index // width)
        x = float(index % width)
        if any((x - px) ** 2 + (y - py) ** 2 < min_distance**2 for _pv, px, py in points):
            continue
        points.append((float(value), x, y))
        if len(points) >= k:
            break
    return points


def constrained_lesser_points(probs: torch.Tensor, keys: list[str]) -> dict[str, tuple[float, float]]:
    key_to_channel = {key: i for i, key in enumerate(keys)}
    if not all(key in key_to_channel for key in LESSER_KEYS):
        return {}

    candidates = {
        key: distinct_topk_points(probs[key_to_channel[key]], k=12, min_distance=4.0)
        for key in LESSER_KEYS
    }
    best_score = -float("inf")
    best_points: dict[str, tuple[float, float]] = {}

    for upper in candidates["hip_lesser_trochanter_upper"]:
        for tip in candidates["hip_lesser_trochanter_tip"]:
            for lower in candidates["hip_lesser_trochanter_lower"]:
                _upper_value, upper_x, upper_y = upper
                _tip_value, tip_x, tip_y = tip
                _lower_value, lower_x, lower_y = lower
                log_score = (
                    np.log(max(upper[0], 1e-6))
                    + np.log(max(tip[0], 1e-6))
                    + np.log(max(lower[0], 1e-6))
                )
                upper_tip_dy = tip_y - upper_y
                tip_lower_dy = lower_y - tip_y
                total_dy = lower_y - upper_y
                order_penalty = max(0.0, 3.0 - upper_tip_dy) + max(0.0, 3.0 - tip_lower_dy)
                base_len = ((upper_x - lower_x) ** 2 + (upper_y - lower_y) ** 2) ** 0.5
                collapse_penalty = max(0.0, 8.0 - base_len)
                upper_tip_dist = ((upper_x - tip_x) ** 2 + (upper_y - tip_y) ** 2) ** 0.5
                tip_lower_dist = ((tip_x - lower_x) ** 2 + (tip_y - lower_y) ** 2) ** 0.5
                gap_penalty = (
                    max(0.0, 10.0 - upper_tip_dy)
                    + max(0.0, 10.0 - tip_lower_dy)
                    + max(0.0, upper_tip_dy - 32.0)
                    + max(0.0, tip_lower_dy - 36.0)
                    + max(0.0, total_dy - 58.0)
                )
                distance_penalty = (
                    max(0.0, upper_tip_dist - 36.0)
                    + max(0.0, tip_lower_dist - 40.0)
                    + max(0.0, base_len - 62.0)
                )
                balance_penalty = abs(upper_tip_dy - tip_lower_dy)
                score = (
                    log_score
                    - 1.5 * order_penalty
                    - 0.5 * collapse_penalty
                    - 0.8 * gap_penalty
                    - 1.2 * distance_penalty
                    - 0.08 * balance_penalty
                )

                if score > best_score:
                    best_score = score
                    best_points = {
                        "hip_lesser_trochanter_upper": (upper_x, upper_y),
                        "hip_lesser_trochanter_tip": (tip_x, tip_y),
                        "hip_lesser_trochanter_lower": (lower_x, lower_y),
                    }
    return best_points


def load_model(path: Path, device: torch.device) -> dict[str, Any] | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    keys = list(checkpoint["keys"])
    model = TinyUNet(out_channels=len(keys))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return {
        "path": path,
        "model": model,
        "keys": keys,
        "image_size": int(checkpoint["image_size"]),
        "region": checkpoint["region"],
    }


@torch.no_grad()
def predict_image(
    image: Image.Image,
    bundle: dict[str, Any],
    device: torch.device,
    anatomical_region: str = "",
) -> dict[str, tuple[float, float]]:
    tensor = image_tensor(image, bundle["image_size"], device)
    logits = bundle["model"](tensor)
    probs = torch.sigmoid(logits)[0].detach().cpu()
    points = predict_points(logits)[0].detach().cpu().numpy()
    denom = float(bundle["image_size"] - 1)
    out: dict[str, tuple[float, float]] = {}
    for key, (x, y) in zip(bundle["keys"], points, strict=True):
        out[key] = (float(x) / denom * image.width, float(y) / denom * image.height)
    if bundle["region"] == "hip":
        for key, (x, y) in constrained_lesser_points(probs, bundle["keys"]).items():
            out[key] = (x / denom * image.width, y / denom * image.height)
    return out


def empty_output(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "input_source": row.get("input_source", ""),
        "study_key": row.get("study_key", ""),
        "member": row.get("member", ""),
        "filename": row.get("filename", ""),
        "anatomical_region": row.get("anatomical_region", ""),
        "is_canonical_for_region": row.get("is_canonical_for_region", ""),
        "landmark_model": "heatmap_landmarks",
        "landmark_status": "",
    }
    for key in ALL_KEYS:
        out[f"{key}_x"] = ""
        out[f"{key}_y"] = ""
        out[f"{key}_visible"] = 0
    return out


def predict_row(
    row: dict[str, str],
    dicom_input: Path,
    hip_bundle: dict[str, Any] | None,
    spine_bundle: dict[str, Any] | None,
    device: torch.device,
    use_spine_model: bool,
) -> dict[str, Any]:
    out = empty_output(row)
    region = row.get("anatomical_region", "")
    if region in {"left_hip", "right_hip"}:
        bundle = hip_bundle
    elif region == "spine" and use_spine_model:
        bundle = spine_bundle
    elif region == "spine":
        out["landmark_status"] = "skipped_spine_model_disabled"
        return out
    else:
        out["landmark_status"] = "unsupported_region"
        return out
    if bundle is None:
        out["landmark_status"] = "model_missing"
        return out

    try:
        image, _meta = image_from_dicom(read_member_raw(dicom_input, row["member"]))
    except Exception as exc:
        out["landmark_status"] = f"render_failed: {exc}"
        return out

    points = predict_image(image, bundle, device, region)
    for key, (x, y) in points.items():
        out[f"{key}_x"] = f"{x:.3f}"
        out[f"{key}_y"] = f"{y:.3f}"
        out[f"{key}_visible"] = 1
    out["landmark_model"] = f"heatmap_{bundle['region']}"
    out["landmark_status"] = "predicted"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--dicom-input", type=Path, default=DEFAULT_DICOM_INPUT)
    parser.add_argument("--hip-model", type=Path, default=DEFAULT_HIP_MODEL)
    parser.add_argument("--spine-model", type=Path, default=DEFAULT_SPINE_MODEL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-spine-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    hip_bundle = load_model(args.hip_model, device)
    spine_bundle = load_model(args.spine_model, device) if args.use_spine_model else None
    rows = read_csv(args.input_csv)
    out_rows = []
    for row in rows:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = predict_row(
            row=row,
            dicom_input=args.dicom_input,
            hip_bundle=hip_bundle,
            spine_bundle=spine_bundle,
            device=device,
            use_spine_model=args.use_spine_model,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
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
