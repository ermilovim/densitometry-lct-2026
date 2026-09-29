from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_CSV = ROOT / "artifacts" / "clean_image_index.csv"
DEFAULT_EXISTING_CSV = ROOT / "artifacts" / "landmark_pilot.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "landmark_all.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "landmark_all_report.csv"

BASE_COLUMNS = [
    "pilot_id",
    "pilot_group",
    "selected_reason",
    "annotation_status",
    "annotation_notes",
    "image_number",
    "split",
    "study_folder",
    "anatomical_region",
    "member",
    "filename",
    "columns",
    "rows",
    "quality_class",
    "violation_type",
    "manual_notes",
    "label_row_no",
]

LABEL_COLUMNS = [
    "spine_positioning",
    "spine_axis",
    "spine_artifact",
    "right_hip_positioning_rotation",
    "right_hip_roi",
    "left_hip_positioning_rotation",
    "left_hip_roi",
]

SPINE_LANDMARKS = [
    "spine_th12_center",
    "spine_l1_center",
    "spine_l2_center",
    "spine_l3_center",
    "spine_l4_center",
    "spine_l5_center",
    "spine_left_iliac_crest",
    "spine_right_iliac_crest",
]

HIP_LANDMARKS = [
    "hip_greater_trochanter_top_edge",
    "hip_greater_trochanter_lateral_edge",
    "hip_lesser_trochanter_upper",
    "hip_lesser_trochanter_tip",
    "hip_lesser_trochanter_lower",
    "hip_ischium_edge",
]

LANDMARK_COLUMNS = [f"{name}_{axis}" for name in [*SPINE_LANDMARKS, *HIP_LANDMARKS] for axis in ("x", "y")]
FIELDNAMES = BASE_COLUMNS + LABEL_COLUMNS + LANDMARK_COLUMNS


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pilot_group(region: str) -> str:
    return "spine" if region == "spine" else "hip"


def selected_reason(row: dict[str, str]) -> str:
    return "bad_label" if row.get("quality_class") == "1" else "good_control"


def build_rows(clean_rows: list[dict[str, str]], existing_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    existing_by_member = {row.get("member", ""): row for row in existing_rows}
    eligible = [
        row
        for row in clean_rows
        if row.get("split") == "train"
        and row.get("label_status") == "labeled"
        and row.get("anatomical_region") in {"spine", "right_hip", "left_hip"}
    ]
    eligible.sort(key=lambda row: (row.get("anatomical_region", ""), int(float(row.get("label_row_no") or 999999)), int(float(row.get("image_number") or 999999))))

    output: list[dict[str, Any]] = []
    for row in eligible:
        old = existing_by_member.get(row["member"], {})
        out = {field: "" for field in FIELDNAMES}
        out.update(
            {
                "pilot_id": len(output) + 1,
                "pilot_group": pilot_group(row["anatomical_region"]),
                "selected_reason": selected_reason(row),
                "annotation_status": old.get("annotation_status", "new") or "new",
                "annotation_notes": old.get("annotation_notes", ""),
            }
        )
        for field in BASE_COLUMNS:
            if field in row:
                out[field] = row[field]
        for field in LABEL_COLUMNS:
            out[field] = row.get(field, "")
        for field in LANDMARK_COLUMNS:
            out[field] = old.get(field, "")
        output.append(out)
    return output


def build_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters = {
        "total": Counter({"rows": len(rows)}),
        "by_group": Counter(row["pilot_group"] for row in rows),
        "by_region": Counter(row["anatomical_region"] for row in rows),
        "by_quality": Counter((row["pilot_group"], row["quality_class"]) for row in rows),
        "by_status": Counter(row["annotation_status"] for row in rows),
        "by_reason": Counter(row["selected_reason"] for row in rows),
    }
    report = []
    for section, counter in counters.items():
        for metric, value in sorted(counter.items(), key=lambda item: str(item[0])):
            if isinstance(metric, tuple):
                metric = " / ".join(metric)
            report.append({"section": section, "metric": metric, "value": value})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-csv", type=Path, default=DEFAULT_CLEAN_CSV)
    parser.add_argument("--existing-csv", type=Path, default=DEFAULT_EXISTING_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(read_csv(args.clean_csv), read_csv(args.existing_csv))
    report = build_report(rows)
    write_csv(args.out_csv, rows, FIELDNAMES)
    write_csv(args.report_csv, report, ["section", "metric", "value"])
    print(f"Rows: {len(rows)}")
    print("By group:", dict(Counter(row["pilot_group"] for row in rows)))
    print("By status:", dict(Counter(row["annotation_status"] for row in rows)))
    print("By quality:", dict(Counter((row["pilot_group"], row["quality_class"]) for row in rows)))
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.report_csv}")


if __name__ == "__main__":
    main()
