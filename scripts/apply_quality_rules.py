from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFERENCE_CSV = ROOT / "artifacts" / "inference_skeleton.csv"
DEFAULT_LANDMARKS_CSV = ROOT / "artifacts" / "inference_landmark_predictions_knn.csv"
DEFAULT_SPINE_CLASSIFIERS_CSV = ROOT / "artifacts" / "inference_spine_classifier_predictions.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "quality_predictions_baseline.csv"

PIXEL_SPACING_X_MM = 0.60
PIXEL_SPACING_Y_MM = 1.05

SPINE_AXIS_ANGLE_THRESHOLD_DEG = 1.979
HIP_LESSER_PROMINENCE_LOW_THRESHOLD_MM = 0.200
HIP_LESSER_PROMINENCE_HIGH_THRESHOLD_MM = 6.346
HIP_ROI_TOP_THRESHOLD_MM = 30.000
HIP_ROI_LATERAL_THRESHOLD_MM = 20.000
HIP_ROI_BOTTOM_THRESHOLD_MM = 30.000
HIP_ROI_LESSER_LOWER_BOTTOM_THRESHOLD_MM = 30.000

SPINE_CENTER_KEYS = [
    "spine_th12_center",
    "spine_l1_center",
    "spine_l2_center",
    "spine_l3_center",
    "spine_l4_center",
    "spine_l5_center",
]
HIP_GREATER_TOP_KEY = "hip_greater_trochanter_top_edge"
HIP_GREATER_LATERAL_KEY = "hip_greater_trochanter_lateral_edge"
HIP_LESSER_UPPER_KEY = "hip_lesser_trochanter_upper"
HIP_LESSER_LOWER_KEY = "hip_lesser_trochanter_lower"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def is_bad_flag(value: str) -> bool:
    return value in {"1", "1.0", "True", "true"}


def point(row: dict[str, str], key: str) -> tuple[float, float] | None:
    if row.get(f"{key}_visible") != "1":
        return None
    x = as_float(row.get(f"{key}_x", ""))
    y = as_float(row.get(f"{key}_y", ""))
    if x is None or y is None:
        return None
    return x, y


def to_mm(pt: tuple[float, float]) -> tuple[float, float]:
    return pt[0] * PIXEL_SPACING_X_MM, pt[1] * PIXEL_SPACING_Y_MM


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def point_line_distance(point_mm: tuple[float, float], a_mm: tuple[float, float], b_mm: tuple[float, float]) -> float | None:
    base = distance(a_mm, b_mm)
    if base == 0:
        return None
    cross = (b_mm[0] - a_mm[0]) * (point_mm[1] - a_mm[1]) - (b_mm[1] - a_mm[1]) * (point_mm[0] - a_mm[0])
    return abs(cross) / base


def spine_metrics(row: dict[str, str]) -> dict[str, Any]:
    centers = [(key, point(row, key)) for key in SPINE_CENTER_KEYS]
    centers = [(key, pt) for key, pt in centers if pt is not None]
    out: dict[str, Any] = {
        "spine_l5_visible": int(point(row, "spine_l5_center") is not None),
        "spine_left_iliac_visible": int(point(row, "spine_left_iliac_crest") is not None),
        "spine_right_iliac_visible": int(point(row, "spine_right_iliac_crest") is not None),
        "spine_axis_angle_deg": "",
    }
    if len(centers) >= 2:
        centers.sort(key=lambda item: item[1][1])
        top = to_mm(centers[0][1])
        bottom = to_mm(centers[-1][1])
        dx = bottom[0] - top[0]
        dy = bottom[1] - top[1]
        out["spine_axis_angle_deg"] = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 90.0
    return out


def hip_metrics(row: dict[str, str], anatomical_region: str, columns: float, rows: float) -> dict[str, Any]:
    upper = point(row, "hip_lesser_trochanter_upper")
    tip = point(row, "hip_lesser_trochanter_tip")
    lower = point(row, "hip_lesser_trochanter_lower")
    greater_top = point(row, "hip_greater_trochanter_top_edge")
    greater_lateral = point(row, "hip_greater_trochanter_lateral_edge")
    ischium = point(row, "hip_ischium_edge")

    out: dict[str, Any] = {
        "hip_lesser_prominence_mm": "",
        "hip_top_margin_mm": "",
        "hip_lateral_margin_mm": "",
        "hip_bottom_margin_mm": "",
        "hip_lesser_lower_bottom_margin_mm": "",
    }
    if upper is not None and tip is not None and lower is not None:
        prominence = point_line_distance(to_mm(tip), to_mm(upper), to_mm(lower))
        out["hip_lesser_prominence_mm"] = "" if prominence is None else prominence
    if greater_top is not None:
        out["hip_top_margin_mm"] = max(greater_top[1], 0) * PIXEL_SPACING_Y_MM
    if greater_lateral is not None:
        if anatomical_region == "right_hip":
            lateral_px = max(greater_lateral[0], 0)
        elif anatomical_region == "left_hip":
            lateral_px = max(columns - greater_lateral[0], 0)
        else:
            lateral_px = min(max(greater_lateral[0], 0), max(columns - greater_lateral[0], 0))
        out["hip_lateral_margin_mm"] = lateral_px * PIXEL_SPACING_X_MM
    if ischium is not None:
        out["hip_bottom_margin_mm"] = max(rows - ischium[1], 0) * PIXEL_SPACING_Y_MM
    if lower is not None:
        out["hip_lesser_lower_bottom_margin_mm"] = max(rows - lower[1], 0) * PIXEL_SPACING_Y_MM
    return out


def hip_positioning_order_bad(row: dict[str, str], anatomical_region: str) -> bool:
    greater_top = point(row, HIP_GREATER_TOP_KEY)
    greater_lateral = point(row, HIP_GREATER_LATERAL_KEY)
    lesser_upper = point(row, HIP_LESSER_UPPER_KEY)
    lesser_lower = point(row, HIP_LESSER_LOWER_KEY)
    if not all([greater_top, greater_lateral, lesser_upper, lesser_lower]):
        return False

    vertical_ok = greater_top[1] < greater_lateral[1] < lesser_upper[1]
    return not vertical_ok


def fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def evaluate_canonical(
    inference_row: dict[str, str],
    landmark_row: dict[str, str],
    model_name: str,
    spine_classifier_row: dict[str, str] | None = None,
) -> dict[str, Any]:
    region = inference_row["anatomical_region"]
    violations: list[str] = []
    metrics: dict[str, Any] = {}
    status = f"quality_predicted_{model_name}"

    if region == "spine":
        metrics = spine_metrics(landmark_row)
        axis_value = metrics["spine_axis_angle_deg"]
        axis_bad = axis_value != "" and float(axis_value) >= SPINE_AXIS_ANGLE_THRESHOLD_DEG

        if spine_classifier_row is not None and spine_classifier_row.get("spine_classifier_status") == "predicted":
            metrics.update(
                {
                    "spine_positioning_prob_bad": spine_classifier_row.get("spine_positioning_prob_bad", ""),
                    "spine_positioning_pred_bad": spine_classifier_row.get("spine_positioning_pred_bad", ""),
                    "spine_artifact_prob_bad": spine_classifier_row.get("spine_artifact_prob_bad", ""),
                    "spine_artifact_pred_bad": spine_classifier_row.get("spine_artifact_pred_bad", ""),
                }
            )
            if is_bad_flag(spine_classifier_row.get("spine_positioning_pred_bad", "")):
                violations.append("spine_positioning")
            if is_bad_flag(spine_classifier_row.get("spine_artifact_pred_bad", "")):
                violations.append("spine_artifact")
        else:
            positioning_bad = not (
                metrics["spine_left_iliac_visible"]
                and metrics["spine_right_iliac_visible"]
            )
            if positioning_bad:
                violations.append("spine_positioning")

        if axis_bad:
            violations.append("spine_axis")
    elif region in {"left_hip", "right_hip"}:
        columns = float(inference_row["columns"])
        rows = float(inference_row["rows"])
        metrics = hip_metrics(landmark_row, region, columns, rows)
        metrics["hip_positioning_order_bad"] = int(hip_positioning_order_bad(landmark_row, region))
        prominence = metrics["hip_lesser_prominence_mm"]
        top = metrics["hip_top_margin_mm"]
        lateral = metrics["hip_lateral_margin_mm"]
        bottom = metrics["hip_bottom_margin_mm"]
        lesser_lower_bottom = metrics["hip_lesser_lower_bottom_margin_mm"]
        structure_missing = top == "" or lateral == "" or bottom == ""
        positioning_order_bad = bool(metrics["hip_positioning_order_bad"])
        rotation_bad = (
            prominence == ""
            or float(prominence) <= HIP_LESSER_PROMINENCE_LOW_THRESHOLD_MM
            or float(prominence) >= HIP_LESSER_PROMINENCE_HIGH_THRESHOLD_MM
        )
        roi_bad = (
            top == ""
            or lateral == ""
            or bottom == ""
            or float(top) < HIP_ROI_TOP_THRESHOLD_MM
            or float(lateral) < HIP_ROI_LATERAL_THRESHOLD_MM
            or float(bottom) < HIP_ROI_BOTTOM_THRESHOLD_MM
            or (lesser_lower_bottom != "" and float(lesser_lower_bottom) < HIP_ROI_LESSER_LOWER_BOTTOM_THRESHOLD_MM)
        )
        if structure_missing or positioning_order_bad or rotation_bad:
            violations.append("hip_positioning_rotation")
        if roi_bad:
            violations.append("hip_roi")
    else:
        violations.append("unknown_anatomical_region")
        status = "unsupported_region"

    return {
        "quality_class": int(bool(violations)),
        "violation_type": ";".join(violations),
        "processing_status": status,
        **metrics,
    }


def build_quality_rows(
    inference_rows: list[dict[str, str]],
    landmark_rows: list[dict[str, str]],
    model_name: str,
    spine_classifier_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    landmarks_by_member = {row["member"]: row for row in landmark_rows}
    spine_classifiers_by_member = {row["member"]: row for row in spine_classifier_rows or []}
    use_spine_classifiers = bool(spine_classifier_rows)
    out_rows: list[dict[str, Any]] = []

    for row in inference_rows:
        region = row.get("anatomical_region", "")
        landmark = landmarks_by_member.get(row["member"])
        spine_classifier = spine_classifiers_by_member.get(row["member"])

        if region not in {"spine", "left_hip", "right_hip"}:
            quality = evaluate_canonical(row, landmark or {}, model_name, spine_classifier)
        elif landmark is None or landmark.get("landmark_status") != "predicted":
            quality = {
                "quality_class": "",
                "violation_type": "landmark_prediction_missing",
                "processing_status": "quality_not_evaluated_landmarks_missing",
            }
        elif (
            use_spine_classifiers
            and region == "spine"
            and (spine_classifier is None or spine_classifier.get("spine_classifier_status") != "predicted")
        ):
            quality = {
                "quality_class": "",
                "violation_type": "spine_classifier_prediction_missing",
                "processing_status": "quality_not_evaluated_spine_classifier_missing",
            }
        else:
            quality = evaluate_canonical(row, landmark, model_name, spine_classifier)

        out_rows.append(
            {
                "input_source": row.get("input_source", ""),
                "path_to_study": row.get("path_to_study", ""),
                "study_key": row.get("study_key", ""),
                "dicom_study_uid": row.get("dicom_study_uid", ""),
                "series_uid": row.get("series_uid", ""),
                "image_uid": row.get("image_uid", ""),
                "member": row.get("member", ""),
                "filename": row.get("filename", ""),
                "anatomical_region": region,
                "is_canonical_for_region": row.get("is_canonical_for_region", ""),
                "is_duplicate_or_derived": row.get("is_duplicate_or_derived", ""),
                "canonical_member": row.get("canonical_member", ""),
                "quality_class": quality.get("quality_class", ""),
                "violation_type": quality.get("violation_type", ""),
                "processing_status": quality.get("processing_status", "quality_not_evaluated"),
                "spine_axis_angle_deg": fmt(quality.get("spine_axis_angle_deg", "")),
                "spine_positioning_prob_bad": fmt(quality.get("spine_positioning_prob_bad", "")),
                "spine_positioning_pred_bad": fmt(quality.get("spine_positioning_pred_bad", "")),
                "spine_artifact_prob_bad": fmt(quality.get("spine_artifact_prob_bad", "")),
                "spine_artifact_pred_bad": fmt(quality.get("spine_artifact_pred_bad", "")),
                "hip_lesser_prominence_mm": fmt(quality.get("hip_lesser_prominence_mm", "")),
                "hip_positioning_order_bad": fmt(quality.get("hip_positioning_order_bad", "")),
                "hip_top_margin_mm": fmt(quality.get("hip_top_margin_mm", "")),
                "hip_lateral_margin_mm": fmt(quality.get("hip_lateral_margin_mm", "")),
                "hip_bottom_margin_mm": fmt(quality.get("hip_bottom_margin_mm", "")),
                "hip_lesser_lower_bottom_margin_mm": fmt(quality.get("hip_lesser_lower_bottom_margin_mm", "")),
            }
        )
    return out_rows

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-csv", type=Path, default=DEFAULT_INFERENCE_CSV)
    parser.add_argument("--landmarks-csv", type=Path, default=DEFAULT_LANDMARKS_CSV)
    parser.add_argument("--spine-classifiers-csv", type=Path, default=DEFAULT_SPINE_CLASSIFIERS_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--model-name", default="knn_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spine_classifier_rows = read_csv(args.spine_classifiers_csv) if args.spine_classifiers_csv.exists() else []
    rows = build_quality_rows(
        read_csv(args.inference_csv),
        read_csv(args.landmarks_csv),
        args.model_name,
        spine_classifier_rows=spine_classifier_rows,
    )
    fieldnames = [
        "input_source",
        "path_to_study",
        "study_key",
        "dicom_study_uid",
        "series_uid",
        "image_uid",
        "member",
        "filename",
        "anatomical_region",
        "is_canonical_for_region",
        "is_duplicate_or_derived",
        "canonical_member",
        "quality_class",
        "violation_type",
        "processing_status",
        "spine_axis_angle_deg",
        "spine_positioning_prob_bad",
        "spine_positioning_pred_bad",
        "spine_artifact_prob_bad",
        "spine_artifact_pred_bad",
        "hip_lesser_prominence_mm",
        "hip_positioning_order_bad",
        "hip_top_margin_mm",
        "hip_lateral_margin_mm",
        "hip_bottom_margin_mm",
        "hip_lesser_lower_bottom_margin_mm",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
