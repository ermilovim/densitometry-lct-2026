from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANDMARKS_CSV = ROOT / "artifacts" / "landmark_all.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "hip_roi_all_rule_check.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "hip_roi_all_rule_report.csv"

PIXEL_SPACING_X_MM = 0.60
PIXEL_SPACING_Y_MM = 1.05


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


def roi_label(row: dict[str, str]) -> str:
    if row["anatomical_region"] == "left_hip":
        return row.get("left_hip_roi", "")
    if row["anatomical_region"] == "right_hip":
        return row.get("right_hip_roi", "")
    return ""


def hip_roi_margins(row: dict[str, str]) -> dict[str, float | None]:
    columns = as_float(row.get("columns", ""))
    rows = as_float(row.get("rows", ""))
    greater_top_y = as_float(row.get("hip_greater_trochanter_top_edge_y", ""))
    greater_lateral_x = as_float(row.get("hip_greater_trochanter_lateral_edge_x", ""))
    ischium_y = as_float(row.get("hip_ischium_edge_y", ""))

    top = None if greater_top_y is None else greater_top_y * PIXEL_SPACING_Y_MM
    if greater_lateral_x is None or columns is None:
        lateral = None
    elif row["anatomical_region"] == "right_hip":
        lateral = greater_lateral_x * PIXEL_SPACING_X_MM
    else:
        lateral = (columns - greater_lateral_x) * PIXEL_SPACING_X_MM
    bottom = None if rows is None or ischium_y is None else (rows - ischium_y) * PIXEL_SPACING_Y_MM
    return {"top": top, "lateral": lateral, "bottom": bottom}


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def predict_bad(margins: dict[str, float | None], top_thr: float, lateral_thr: float, bottom_thr: float) -> tuple[int, str]:
    reasons: list[str] = []
    if margins["top"] is None:
        reasons.append("top_missing")
    elif margins["top"] < top_thr:
        reasons.append("top")
    if margins["lateral"] is None:
        reasons.append("lateral_missing")
    elif margins["lateral"] < lateral_thr:
        reasons.append("lateral")
    if margins["bottom"] is None:
        reasons.append("bottom_missing")
    elif margins["bottom"] < bottom_thr:
        reasons.append("bottom")
    return int(bool(reasons)), ";".join(reasons)


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["true_bad"] == 1 and row["pred_bad"] == 1)
    fp = sum(1 for row in rows if row["true_bad"] == 0 and row["pred_bad"] == 1)
    fn = sum(1 for row in rows if row["true_bad"] == 1 and row["pred_bad"] == 0)
    tn = sum(1 for row in rows if row["true_bad"] == 0 and row["pred_bad"] == 0)
    precision = tp / (tp + fp) if tp + fp else ""
    recall = tp / (tp + fn) if tp + fn else ""
    specificity = tn / (tn + fp) if tn + fp else ""
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else ""
    balanced_accuracy = (
        (float(recall) + float(specificity)) / 2
        if recall != "" and specificity != ""
        else ""
    )
    return {
        "n": len(rows),
        "positives": sum(1 for row in rows if row["true_bad"] == 1),
        "negatives": sum(1 for row in rows if row["true_bad"] == 0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def build_rows(input_rows: list[dict[str, str]], top_thr: float, lateral_thr: float, bottom_thr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in input_rows:
        if row.get("annotation_status") != "done":
            continue
        if row.get("anatomical_region") not in {"left_hip", "right_hip"}:
            continue
        label = roi_label(row)
        if label not in {"0", "1"}:
            continue

        margins = hip_roi_margins(row)
        pred, reasons = predict_bad(margins, top_thr, lateral_thr, bottom_thr)
        true_bad = int(label)
        out.append(
            {
                "split": row.get("split", ""),
                "pilot_id": row.get("pilot_id", ""),
                "image_number": row.get("image_number", ""),
                "study_folder": row.get("study_folder", ""),
                "anatomical_region": row.get("anatomical_region", ""),
                "filename": row.get("filename", ""),
                "true_bad": true_bad,
                "pred_bad": pred,
                "error_type": "" if pred == true_bad else ("FP" if pred == 1 else "FN"),
                "triggered_by": reasons,
                "top_margin_mm": fmt(margins["top"]),
                "lateral_margin_mm": fmt(margins["lateral"]),
                "bottom_margin_mm": fmt(margins["bottom"]),
                "threshold_top_mm": f"{top_thr:.3f}",
                "threshold_lateral_mm": f"{lateral_thr:.3f}",
                "threshold_bottom_mm": f"{bottom_thr:.3f}",
                "violation_type": row.get("violation_type", ""),
                "annotation_notes": row.get("annotation_notes", ""),
                "manual_notes": row.get("manual_notes", ""),
            }
        )
    return out


def build_report(rows: list[dict[str, Any]], top_thr: float, lateral_thr: float, bottom_thr: float) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for split in ["ALL", "train", "val"]:
        subset = rows if split == "ALL" else [row for row in rows if row["split"] == split]
        if not subset:
            continue
        item = {
            "split": split,
            "threshold_top_mm": f"{top_thr:.3f}",
            "threshold_lateral_mm": f"{lateral_thr:.3f}",
            "threshold_bottom_mm": f"{bottom_thr:.3f}",
            **confusion(subset),
        }
        for key in ["precision", "recall", "specificity", "f1", "balanced_accuracy"]:
            if item[key] != "":
                item[key] = f"{item[key]:.3f}"
        report.append(item)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-csv", type=Path, default=DEFAULT_LANDMARKS_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--top-threshold-mm", type=float, default=30.0)
    parser.add_argument("--lateral-threshold-mm", type=float, default=20.0)
    parser.add_argument("--bottom-threshold-mm", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        read_csv(args.landmarks_csv),
        args.top_threshold_mm,
        args.lateral_threshold_mm,
        args.bottom_threshold_mm,
    )
    report = build_report(rows, args.top_threshold_mm, args.lateral_threshold_mm, args.bottom_threshold_mm)
    write_csv(
        args.out_csv,
        rows,
        [
            "split",
            "pilot_id",
            "image_number",
            "study_folder",
            "anatomical_region",
            "filename",
            "true_bad",
            "pred_bad",
            "error_type",
            "triggered_by",
            "top_margin_mm",
            "lateral_margin_mm",
            "bottom_margin_mm",
            "threshold_top_mm",
            "threshold_lateral_mm",
            "threshold_bottom_mm",
            "violation_type",
            "annotation_notes",
            "manual_notes",
        ],
    )
    write_csv(
        args.report_csv,
        report,
        [
            "split",
            "threshold_top_mm",
            "threshold_lateral_mm",
            "threshold_bottom_mm",
            "n",
            "positives",
            "negatives",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "specificity",
            "f1",
            "balanced_accuracy",
        ],
    )
    print(f"Rows: {len(rows)}")
    for row in report:
        print(
            f"{row['split']}: n={row['n']} pos={row['positives']} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']} "
            f"BA={row['balanced_accuracy']}"
        )
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.report_csv}")


if __name__ == "__main__":
    main()
