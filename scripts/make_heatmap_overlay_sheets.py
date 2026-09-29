from __future__ import annotations

import argparse
import csv
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from evaluate_landmark_knn import HIP_KEYS, SPINE_KEYS
from predict_landmarks_heatmap import load_model, predict_image
from prepare_artifacts import image_from_dicom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "artifacts" / "keypoint_dataset"
DEFAULT_ANNOTATIONS_CSV = DEFAULT_DATASET_DIR / "annotations.csv"
DEFAULT_HEATMAP_DIR = ROOT / "artifacts" / "landmark_heatmap"
DEFAULT_INFERENCE_CSV = ROOT / "artifacts" / "inference_skeleton.csv"
DEFAULT_PREDICTIONS_CSV = ROOT / "artifacts" / "inference_landmark_predictions_heatmap.csv"
DEFAULT_DICOM_INPUT = ROOT / "examples" / "example_input.zip"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "landmark_heatmap_overlay_sheets"
DEFAULT_HIP_MODEL = DEFAULT_HEATMAP_DIR / "hip" / "best.pt"
DEFAULT_SPINE_MODEL = DEFAULT_HEATMAP_DIR / "spine" / "best.pt"

GT_COLOR = (0, 190, 80)
PRED_COLOR = (230, 40, 190)
LINE_COLOR = (255, 170, 230)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def keys_for_region(region: str) -> list[str]:
    if region == "spine":
        return SPINE_KEYS
    if region in {"left_hip", "right_hip"}:
        return HIP_KEYS
    return []


def write_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    x, y = xy
    lines = [text[i : i + 46] for i in range(0, len(text), 46)]
    h = 13 * len(lines) + 4
    draw.rectangle((x, y, x + 300, y + h), fill=(255, 255, 255))
    for i, line in enumerate(lines):
        draw.text((x + 2, y + 2 + i * 13), line, fill=(0, 0, 0))


def draw_circle(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], scale: float) -> None:
    r = max(3, round(4 * scale))
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(2, round(2 * scale)))


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], scale: float) -> None:
    r = max(4, round(5 * scale))
    width = max(2, round(2 * scale))
    draw.line((x - r, y - r, x + r, y + r), fill=color, width=width)
    draw.line((x - r, y + r, x + r, y - r), fill=color, width=width)


def draw_points(
    image: Image.Image,
    gt_points: dict[str, tuple[float, float]],
    pred_points: dict[str, tuple[float, float]],
    title: str,
    thumb_width: int,
) -> Image.Image:
    scale = thumb_width / image.width
    canvas = image.convert("RGB").resize((thumb_width, max(1, round(image.height * scale))))
    draw = ImageDraw.Draw(canvas)
    for key, pt in gt_points.items():
        x, y = pt[0] * scale, pt[1] * scale
        draw_circle(draw, x, y, GT_COLOR, scale)
        draw.text((x + 5, y + 5), key.replace("spine_", "").replace("hip_", "")[:14], fill=GT_COLOR)
    for key, pt in pred_points.items():
        x, y = pt[0] * scale, pt[1] * scale
        if key in gt_points:
            gx, gy = gt_points[key][0] * scale, gt_points[key][1] * scale
            draw.line((gx, gy, x, y), fill=LINE_COLOR, width=max(1, round(scale)))
        draw_cross(draw, x, y, PRED_COLOR, scale)
    write_text_box(draw, (4, 4), title)
    return canvas


def make_sheet(cells: list[Image.Image], out_path: Path, cols: int, gap: int = 12) -> None:
    if not cells:
        return
    cols = min(cols, len(cells))
    max_w = max(cell.width for cell in cells)
    max_h = max(cell.height for cell in cells)
    rows = math.ceil(len(cells) / cols)
    sheet = Image.new("RGB", (cols * max_w + (cols + 1) * gap, rows * max_h + (rows + 1) * gap), "white")
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        sheet.paste(cell, (gap + c * (max_w + gap), gap + r * (max_h + gap)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def annotation_by_pilot(annotations_csv: Path) -> dict[str, dict[str, str]]:
    return {row["pilot_id"]: row for row in read_csv(annotations_csv)}


def val_rows_for_region(heatmap_dir: Path, region_group: str) -> list[dict[str, str]]:
    path = heatmap_dir / region_group / "val_predictions.csv"
    return read_csv(path) if path.exists() else []


def gt_points_from_annotation(row: dict[str, str]) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for key in keys_for_region(row["anatomical_region"]):
        if row.get(f"{key}_visible") != "1":
            continue
        x = as_float(row.get(f"{key}_x", ""))
        y = as_float(row.get(f"{key}_y", ""))
        if x is not None and y is not None:
            points[key] = (x, y)
    return points


def mean_point_error_mm(
    gt_points: dict[str, tuple[float, float]],
    pred_points: dict[str, tuple[float, float]],
) -> float:
    errors = []
    for key, gt in gt_points.items():
        pred = pred_points.get(key)
        if pred is None:
            continue
        dx_mm = (pred[0] - gt[0]) * 0.60
        dy_mm = (pred[1] - gt[1]) * 1.05
        errors.append((dx_mm * dx_mm + dy_mm * dy_mm) ** 0.5)
    return sum(errors) / len(errors) if errors else 0.0


def build_val_sheets(args: argparse.Namespace) -> None:
    annotations = [row for row in read_csv(args.annotations_csv) if row.get("split") == "val"]
    device = torch.device(args.device)
    bundles = {
        "hip": load_model(args.hip_model, device),
        "spine": load_model(args.spine_model, device),
    }

    cells_by_region: dict[str, list[Image.Image]] = defaultdict(list)
    for ann in annotations:
        region = ann["anatomical_region"]
        region_group = "spine" if region == "spine" else "hip"
        if region_group not in args.regions:
            continue
        bundle = bundles.get(region_group)
        if bundle is None:
            continue
        image = Image.open(args.dataset_dir / ann["image_path"])
        gt_points = gt_points_from_annotation(ann)
        pred_points = predict_image(image, bundle, device, region)
        mean_error = mean_point_error_mm(gt_points, pred_points)
        title = (
            f"heatmap val pilot={ann['pilot_id']} img={ann['image_number']} "
            f"{region} mean={mean_error:.1f}mm"
        )
        cells_by_region[region].append(draw_points(image, gt_points, pred_points, title, args.thumb_width))

    for region, cells in cells_by_region.items():
        make_sheet(cells, args.out_dir / f"val_heatmap_{region}_gt_green_pred_magenta.jpg", args.cols)


def pred_point(row: dict[str, str], key: str) -> tuple[float, float] | None:
    if row.get(f"{key}_visible") != "1":
        return None
    x = as_float(row.get(f"{key}_x", ""))
    y = as_float(row.get(f"{key}_y", ""))
    if x is None or y is None:
        return None
    return x, y


def read_member_raw(dicom_input: Path, member: str) -> bytes:
    if dicom_input.is_file() and dicom_input.suffix.lower() == ".zip":
        with zipfile.ZipFile(dicom_input) as zf:
            return zf.read(member)
    return (dicom_input / member).read_bytes()


def build_inference_sheet(args: argparse.Namespace) -> None:
    prediction_rows = [row for row in read_csv(args.predictions_csv) if row.get("landmark_status") == "predicted"]
    cells: list[Image.Image] = []
    for row in prediction_rows:
        keys = keys_for_region(row["anatomical_region"])
        pred_points = {key: pt for key in keys if (pt := pred_point(row, key)) is not None}
        image, _meta = image_from_dicom(read_member_raw(args.dicom_input, row["member"]))
        title = f"heatmap pred {row['filename']} {row['anatomical_region']}"
        cells.append(draw_points(image, {}, pred_points, title, args.thumb_width))
    make_sheet(cells, args.out_dir / "inference_heatmap_pred_magenta.jpg", args.cols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "inference"], default="val")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--heatmap-dir", type=Path, default=DEFAULT_HEATMAP_DIR)
    parser.add_argument("--inference-csv", type=Path, default=DEFAULT_INFERENCE_CSV)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--dicom-input", type=Path, default=DEFAULT_DICOM_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hip-model", type=Path, default=DEFAULT_HIP_MODEL)
    parser.add_argument("--spine-model", type=Path, default=DEFAULT_SPINE_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--regions", nargs="+", choices=["hip", "spine"], default=["hip", "spine"])
    parser.add_argument("--thumb-width", type=int, default=260)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "val":
        build_val_sheets(args)
    else:
        build_inference_sheet(args)
    print(f"Wrote sheets to: {args.out_dir}")


if __name__ == "__main__":
    main()
