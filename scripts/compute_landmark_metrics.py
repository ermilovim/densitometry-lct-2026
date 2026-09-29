from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_CSV = ROOT / "artifacts" / "landmark_pilot.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "landmark_metrics.csv"

PIXEL_SPACING_X_MM = 0.60
PIXEL_SPACING_Y_MM = 1.05

SPINE_CENTER_KEYS = [
    "spine_th12_center",
    "spine_l1_center",
    "spine_l2_center",
    "spine_l3_center",
    "spine_l4_center",
    "spine_l5_center",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def point_px(row: dict[str, str], key: str) -> tuple[float, float] | None:
    x = row.get(f"{key}_x", "")
    y = row.get(f"{key}_y", "")
    if x == "" or y == "":
        return None
    try:
        return float(x), float(y)
    except ValueError:
        return None


def to_mm(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return x * PIXEL_SPACING_X_MM, y * PIXEL_SPACING_Y_MM


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def point_line_distance(
    point: tuple[float, float],
    line_a: tuple[float, float],
    line_b: tuple[float, float],
) -> float | None:
    base = distance(line_a, line_b)
    if base == 0:
        return None
    cross = (line_b[0] - line_a[0]) * (point[1] - line_a[1]) - (line_b[1] - line_a[1]) * (point[0] - line_a[0])
    return abs(cross) / base


def hip_lesser_trochanter_metrics(row: dict[str, str]) -> dict[str, Any]:
    upper = point_px(row, "hip_lesser_trochanter_upper")
    tip = point_px(row, "hip_lesser_trochanter_tip")
    lower = point_px(row, "hip_lesser_trochanter_lower")
    if upper is None or tip is None or lower is None:
        return {
            "hip_lesser_prominence_mm": "",
            "hip_lesser_baseline_mm": "",
            "hip_lesser_prominence_ratio": "",
        }
    upper_mm = to_mm(upper)
    tip_mm = to_mm(tip)
    lower_mm = to_mm(lower)
    baseline_mm = distance(upper_mm, lower_mm)
    prominence_mm = point_line_distance(tip_mm, upper_mm, lower_mm)
    if prominence_mm is None or baseline_mm == 0:
        ratio = ""
    else:
        ratio = prominence_mm / baseline_mm
    return {
        "hip_lesser_prominence_mm": "" if prominence_mm is None else f"{prominence_mm:.3f}",
        "hip_lesser_baseline_mm": f"{baseline_mm:.3f}",
        "hip_lesser_prominence_ratio": "" if ratio == "" else f"{ratio:.4f}",
    }


def hip_roi_metrics(row: dict[str, str]) -> dict[str, Any]:
    greater_top_edge = point_px(row, "hip_greater_trochanter_top_edge")
    greater_lateral_edge = point_px(row, "hip_greater_trochanter_lateral_edge")
    ischium_edge = point_px(row, "hip_ischium_edge")
    try:
        width = float(row.get("columns", ""))
        height = float(row.get("rows", ""))
    except ValueError:
        width = 0
        height = 0

    top_margin_mm = ""
    if greater_top_edge is not None:
        _x, y = greater_top_edge
        top_margin_mm = max(y, 0) * PIXEL_SPACING_Y_MM

    lateral_margin_mm = ""
    lateral_margin_side = ""
    if greater_lateral_edge is not None and width > 0:
        x, _y = greater_lateral_edge
        region = row.get("anatomical_region", "")
        if region == "right_hip":
            lateral_margin_px = max(x, 0)
            lateral_margin_side = "left"
        elif region == "left_hip":
            lateral_margin_px = max(width - x, 0)
            lateral_margin_side = "right"
        else:
            lateral_margin_px = min(max(x, 0), max(width - x, 0))
            lateral_margin_side = "nearest"
        lateral_margin_mm = lateral_margin_px * PIXEL_SPACING_X_MM

    bottom_margin_mm = ""
    if ischium_edge is not None and height > 0:
        _x, y = ischium_edge
        bottom_margin_px = max(height - y, 0)
        bottom_margin_mm = bottom_margin_px * PIXEL_SPACING_Y_MM

    return {
        "hip_top_margin_mm": "" if top_margin_mm == "" else f"{top_margin_mm:.3f}",
        "hip_lateral_margin_mm": "" if lateral_margin_mm == "" else f"{lateral_margin_mm:.3f}",
        "hip_lateral_margin_side": lateral_margin_side,
        "hip_bottom_margin_mm": "" if bottom_margin_mm == "" else f"{bottom_margin_mm:.3f}",
    }


def spine_axis_metrics(row: dict[str, str]) -> dict[str, Any]:
    centers = []
    for key in SPINE_CENTER_KEYS:
        point = point_px(row, key)
        if point is not None:
            centers.append((key, point))

    l5_visible = int(point_px(row, "spine_l5_center") is not None)
    left_iliac_visible = int(point_px(row, "spine_left_iliac_crest") is not None)
    right_iliac_visible = int(point_px(row, "spine_right_iliac_crest") is not None)
    base = {
        "spine_visible_centers": len(centers),
        "spine_l5_visible": l5_visible,
        "spine_left_iliac_visible": left_iliac_visible,
        "spine_right_iliac_visible": right_iliac_visible,
        "spine_both_iliac_visible": int(left_iliac_visible and right_iliac_visible),
        "spine_axis_angle_deg": "",
        "spine_axis_max_deviation_mm": "",
        "spine_axis_mean_deviation_mm": "",
        "spine_axis_sum_deviation_mm": "",
    }
    if len(centers) < 2:
        return base

    centers.sort(key=lambda item: item[1][1])
    center_points_mm = [(key, to_mm(point)) for key, point in centers]
    top = center_points_mm[0][1]
    bottom = center_points_mm[-1][1]
    dx = bottom[0] - top[0]
    dy = bottom[1] - top[1]
    angle = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 90.0
    deviations = [
        point_line_distance(point, top, bottom)
        for _key, point in center_points_mm
    ]
    values = [value for value in deviations if value is not None]
    if values:
        base.update(
            {
                "spine_axis_angle_deg": f"{angle:.3f}",
                "spine_axis_max_deviation_mm": f"{max(values):.3f}",
                "spine_axis_mean_deviation_mm": f"{sum(values) / len(values):.3f}",
                "spine_axis_sum_deviation_mm": f"{sum(values):.3f}",
            }
        )
    return base


def build_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item: dict[str, Any] = {
            "pilot_id": row.get("pilot_id", ""),
            "annotation_status": row.get("annotation_status", ""),
            "study_folder": row.get("study_folder", ""),
            "anatomical_region": row.get("anatomical_region", ""),
            "quality_class": row.get("quality_class", ""),
            "violation_type": row.get("violation_type", ""),
            "filename": row.get("filename", ""),
        }
        item.update(hip_lesser_trochanter_metrics(row))
        item.update(hip_roi_metrics(row))
        item.update(spine_axis_metrics(row))
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-csv", type=Path, default=DEFAULT_PILOT_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.pilot_csv)
    metrics = build_metrics(rows)
    write_csv(args.out_csv, metrics)
    hip_ready = sum(1 for row in metrics if row["hip_lesser_prominence_mm"] != "")
    spine_ready = sum(1 for row in metrics if row["spine_axis_angle_deg"] != "")
    print(f"Rows: {len(metrics)}")
    print(f"Hip lesser trochanter metrics ready: {hip_ready}")
    print(f"Spine axis metrics ready: {spine_ready}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
