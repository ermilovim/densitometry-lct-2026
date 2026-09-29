from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_CSV = ROOT / "artifacts" / "landmark_pilot.csv"
DEFAULT_METRICS_CSV = ROOT / "artifacts" / "landmark_metrics.csv"
DEFAULT_SUMMARY_CSV = ROOT / "artifacts" / "landmark_analysis_summary.csv"
DEFAULT_ROWS_CSV = ROOT / "artifacts" / "landmark_analysis_rows.csv"


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


def summarize(values: list[float | None]) -> dict[str, Any]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0, "missing": len(values), "min": "", "q25": "", "median": "", "q75": "", "max": ""}

    def q(p: float) -> float:
        return vals[min(len(vals) - 1, round((len(vals) - 1) * p))]

    return {
        "n": len(vals),
        "missing": len(values) - len(vals),
        "min": f"{vals[0]:.3f}",
        "q25": f"{q(0.25):.3f}",
        "median": f"{q(0.50):.3f}",
        "q75": f"{q(0.75):.3f}",
        "max": f"{vals[-1]:.3f}",
    }


def add_summary(summary: list[dict[str, Any]], section: str, label: str, metric: str, values: list[float | None]) -> None:
    row = {"section": section, "label": label, "metric": metric}
    row.update(summarize(values))
    summary.append(row)


def build_rows(pilot_rows: list[dict[str, str]], metric_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metrics_by_id = {row["pilot_id"]: row for row in metric_rows}
    out = []
    for row in pilot_rows:
        metrics = metrics_by_id.get(row["pilot_id"], {})
        if row["pilot_group"] == "hip":
            rotation_label = row["right_hip_positioning_rotation"] if row["anatomical_region"] == "right_hip" else row["left_hip_positioning_rotation"]
            roi_label = row["right_hip_roi"] if row["anatomical_region"] == "right_hip" else row["left_hip_roi"]
        else:
            rotation_label = ""
            roi_label = ""
        out.append(
            {
                "pilot_id": row["pilot_id"],
                "image_number": row["image_number"],
                "pilot_group": row["pilot_group"],
                "anatomical_region": row["anatomical_region"],
                "quality_class": row["quality_class"],
                "violation_type": row["violation_type"],
                "annotation_status": row["annotation_status"],
                "spine_axis_label": row["spine_axis"],
                "hip_rotation_label": rotation_label,
                "hip_roi_label": roi_label,
                "spine_axis_angle_deg": metrics.get("spine_axis_angle_deg", ""),
                "hip_lesser_prominence_mm": metrics.get("hip_lesser_prominence_mm", ""),
                "hip_lesser_prominence_ratio": metrics.get("hip_lesser_prominence_ratio", ""),
                "hip_top_margin_mm": metrics.get("hip_top_margin_mm", ""),
                "hip_lateral_margin_mm": metrics.get("hip_lateral_margin_mm", ""),
                "hip_bottom_margin_mm": metrics.get("hip_bottom_margin_mm", ""),
                "annotation_notes": row.get("annotation_notes", ""),
            }
        )
    return out


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    status_counts = Counter(row["annotation_status"] for row in rows)
    for status, count in sorted(status_counts.items()):
        summary.append({"section": "status", "label": status, "metric": "count", "n": count})

    for label in ["0", "1"]:
        vals = [as_float(row["spine_axis_angle_deg"]) for row in rows if row["pilot_group"] == "spine" and row["spine_axis_label"] == label]
        add_summary(summary, "spine_axis", label, "spine_axis_angle_deg", vals)

    for label in ["0", "1"]:
        hip_rows = [row for row in rows if row["pilot_group"] == "hip" and row["hip_rotation_label"] == label]
        add_summary(summary, "hip_rotation", label, "hip_lesser_prominence_mm", [as_float(row["hip_lesser_prominence_mm"]) for row in hip_rows])
        add_summary(summary, "hip_rotation", label, "hip_lesser_prominence_ratio", [as_float(row["hip_lesser_prominence_ratio"]) for row in hip_rows])

    for label in ["0", "1"]:
        hip_rows = [row for row in rows if row["pilot_group"] == "hip" and row["hip_roi_label"] == label]
        for metric in ["hip_top_margin_mm", "hip_lateral_margin_mm", "hip_bottom_margin_mm"]:
            add_summary(summary, "hip_roi", label, metric, [as_float(row[metric]) for row in hip_rows])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-csv", type=Path, default=DEFAULT_PILOT_CSV)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--rows-csv", type=Path, default=DEFAULT_ROWS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(read_csv(args.pilot_csv), read_csv(args.metrics_csv))
    summary = build_summary(rows)
    write_csv(args.rows_csv, rows)
    write_csv(
        args.summary_csv,
        summary,
        fieldnames=["section", "label", "metric", "n", "missing", "min", "q25", "median", "q75", "max"],
    )
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {args.rows_csv}")
    print(f"Wrote: {args.summary_csv}")


if __name__ == "__main__":
    main()
