from __future__ import annotations

import argparse
import csv
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from evaluate_landmark_knn import (
    HIP_KEYS,
    SPINE_KEYS,
    load_vectors,
    nearest_rows,
    point_from_neighbors,
    read_csv,
)
from prepare_artifacts import image_from_dicom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "artifacts" / "keypoint_dataset"
DEFAULT_ANNOTATIONS_CSV = DEFAULT_DATASET_DIR / "annotations.csv"
DEFAULT_INFERENCE_CSV = ROOT / "artifacts" / "inference_skeleton.csv"
DEFAULT_PREDICTIONS_CSV = ROOT / "artifacts" / "inference_landmark_predictions_knn.csv"
DEFAULT_DICOM_INPUT = ROOT / "examples" / "example_input.zip"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "landmark_overlay_sheets"

GT_COLOR = (0, 190, 80)
PRED_COLOR = (230, 40, 190)
LINE_COLOR = (255, 170, 230)


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


def gt_point(row: dict[str, str], key: str) -> tuple[float, float] | None:
    if row.get(f"{key}_visible") != "1":
        return None
    x = as_float(row.get(f"{key}_x", ""))
    y = as_float(row.get(f"{key}_y", ""))
    if x is None or y is None:
        return None
    return x, y


def pred_point(row: dict[str, str], key: str) -> tuple[float, float] | None:
    if row.get(f"{key}_visible") != "1":
        return None
    x = as_float(row.get(f"{key}_x", ""))
    y = as_float(row.get(f"{key}_y", ""))
    if x is None or y is None:
        return None
    return x, y


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
        x = gap + c * (max_w + gap)
        y = gap + r * (max_h + gap)
        sheet.paste(cell, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def build_val_sheets(args: argparse.Namespace) -> None:
    rows = read_csv(args.annotations_csv)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    vectors = load_vectors(rows, args.dataset_dir, args.size)
    train_by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        train_by_region[row["anatomical_region"]].append(row)

    cells_by_region: dict[str, list[Image.Image]] = defaultdict(list)
    for row in val_rows:
        region = row["anatomical_region"]
        keys = keys_for_region(region)
        neighbors = nearest_rows(row, train_by_region[region], vectors, exclude_self=False, k=args.k)
        gt_points = {key: pt for key in keys if (pt := gt_point(row, key)) is not None}
        pred_points = {
            key: pt
            for key in keys
            if (pt := point_from_neighbors(neighbors, row, key)) is not None
        }
        image = Image.open(args.dataset_dir / row["image_path"])
        title = f"val pilot={row['pilot_id']} img={row['image_number']} {region}"
        cells_by_region[region].append(draw_points(image, gt_points, pred_points, title, args.thumb_width))

    for region, cells in cells_by_region.items():
        make_sheet(cells, args.out_dir / f"val_{region}_gt_green_pred_magenta.jpg", args.cols)


def read_member_raw(dicom_input: Path, member: str) -> bytes:
    if dicom_input.is_file() and dicom_input.suffix.lower() == ".zip":
        with zipfile.ZipFile(dicom_input) as zf:
            return zf.read(member)
    return (dicom_input / member).read_bytes()


def build_inference_sheet(args: argparse.Namespace) -> None:
    inference_rows = {row["member"]: row for row in read_csv(args.inference_csv)}
    prediction_rows = [row for row in read_csv(args.predictions_csv) if row.get("landmark_status") == "predicted"]
    cells: list[Image.Image] = []
    for row in prediction_rows:
        source = inference_rows.get(row["member"], {})
        keys = keys_for_region(row["anatomical_region"])
        pred_points = {key: pt for key in keys if (pt := pred_point(row, key)) is not None}
        image, _meta = image_from_dicom(read_member_raw(args.dicom_input, row["member"]))
        title = f"pred {row['filename']} {row['anatomical_region']} canonical={row['is_canonical_for_region']}"
        cells.append(draw_points(image, {}, pred_points, title, args.thumb_width))
    make_sheet(cells, args.out_dir / "inference_pred_magenta.jpg", args.cols)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["val", "inference"], default="val")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--inference-csv", type=Path, default=DEFAULT_INFERENCE_CSV)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--dicom-input", type=Path, default=DEFAULT_DICOM_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--k", type=int, default=5)
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
