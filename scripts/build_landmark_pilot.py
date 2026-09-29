from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_CSV = ROOT / "artifacts" / "clean_image_index.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "landmark_pilot.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "landmark_pilot_report.csv"

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
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_sort_key(row: dict[str, str]) -> tuple[int, int]:
    label_no = int(float(row.get("label_row_no") or 999999))
    image_no = int(float(row.get("image_number") or 999999))
    return label_no, image_no


def take_good_controls(rows: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    if n <= 0:
        return []
    return sorted(rows, key=row_sort_key)[:n]


def pilot_group(row: dict[str, str]) -> str:
    return "spine" if row["anatomical_region"] == "spine" else "hip"


def build_pilot_rows(clean_rows: list[dict[str, str]], spine_good: int, hip_good: int) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in clean_rows
        if row.get("split") == "train" and row.get("label_status") == "labeled"
    ]
    spine_bad = [row for row in eligible if row["anatomical_region"] == "spine" and row.get("quality_class") == "1"]
    spine_good_rows = [row for row in eligible if row["anatomical_region"] == "spine" and row.get("quality_class") == "0"]
    hip_bad = [
        row
        for row in eligible
        if row["anatomical_region"] in {"right_hip", "left_hip"} and row.get("quality_class") == "1"
    ]
    hip_good_rows = [
        row
        for row in eligible
        if row["anatomical_region"] in {"right_hip", "left_hip"} and row.get("quality_class") == "0"
    ]

    selected: list[tuple[str, dict[str, str]]] = []
    selected.extend(("bad_label", row) for row in sorted(spine_bad, key=row_sort_key))
    selected.extend(("good_control", row) for row in take_good_controls(spine_good_rows, spine_good))
    selected.extend(("bad_label", row) for row in sorted(hip_bad, key=row_sort_key))
    selected.extend(("good_control", row) for row in take_good_controls(hip_good_rows, hip_good))

    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for reason, row in selected:
        member = row["member"]
        if member in seen:
            continue
        seen.add(member)
        out = {field: "" for field in FIELDNAMES}
        out.update(
            {
                "pilot_id": len(output) + 1,
                "pilot_group": pilot_group(row),
                "selected_reason": reason,
                "annotation_status": "new",
                "annotation_notes": "",
            }
        )
        for field in BASE_COLUMNS:
            if field in row:
                out[field] = row[field]
        for field in LABEL_COLUMNS:
            out[field] = row.get(field, "")
        output.append(out)
    return output


def merge_existing_annotations(
    pilot_rows: list[dict[str, Any]],
    existing_csv: Path,
) -> list[dict[str, Any]]:
    if not existing_csv.exists():
        return pilot_rows
    with existing_csv.open("r", encoding="utf-8", newline="") as fp:
        existing_rows = list(csv.DictReader(fp))
    existing_by_member = {row.get("member", ""): row for row in existing_rows}
    preserve_fields = {
        "annotation_status",
        "annotation_notes",
        *{f"{name}_{axis}" for name in [*SPINE_LANDMARKS, *HIP_LANDMARKS] for axis in ("x", "y")},
        # Legacy fields from the first hip-point schema. Kept only if already present.
        "hip_greater_trochanter_x",
        "hip_greater_trochanter_y",
        "hip_ischium_x",
        "hip_ischium_y",
        "hip_lesser_trochanter_x",
        "hip_lesser_trochanter_y",
        "hip_femoral_neck_center_x",
        "hip_femoral_neck_center_y",
        "hip_anatomy_top_x",
        "hip_anatomy_top_y",
        "hip_anatomy_bottom_x",
        "hip_anatomy_bottom_y",
        "hip_anatomy_lateral_x",
        "hip_anatomy_lateral_y",
    }
    for row in pilot_rows:
        old = existing_by_member.get(row.get("member", ""))
        if not old:
            continue
        for field in preserve_fields:
            if field in row and old.get(field, "") != "":
                row[field] = old[field]
    return pilot_rows


def build_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    counters = {
        "total": Counter({"rows": len(rows)}),
        "by_group": Counter(row["pilot_group"] for row in rows),
        "by_region": Counter(row["anatomical_region"] for row in rows),
        "by_reason": Counter(row["selected_reason"] for row in rows),
        "by_quality": Counter((row["pilot_group"], row["quality_class"]) for row in rows),
        "by_violation": Counter(row["violation_type"] or "<none>" for row in rows),
    }
    for section, counter in counters.items():
        for metric, value in sorted(counter.items(), key=lambda item: str(item[0])):
            if isinstance(metric, tuple):
                metric = " / ".join(metric)
            report.append({"section": section, "metric": metric, "value": value})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-csv", type=Path, default=DEFAULT_CLEAN_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--spine-good", type=int, default=10)
    parser.add_argument("--hip-good", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_rows = read_csv(args.clean_csv)
    pilot_rows = build_pilot_rows(clean_rows, spine_good=args.spine_good, hip_good=args.hip_good)
    pilot_rows = merge_existing_annotations(pilot_rows, args.out_csv)
    report = build_report(pilot_rows)
    write_csv(args.out_csv, pilot_rows, FIELDNAMES)
    write_csv(args.report_csv, report, ["section", "metric", "value"])

    print(f"Pilot rows: {len(pilot_rows)}")
    print("By group:", dict(Counter(row["pilot_group"] for row in pilot_rows)))
    print("By reason:", dict(Counter(row["selected_reason"] for row in pilot_rows)))
    print("By quality:", dict(Counter((row["pilot_group"], row["quality_class"]) for row in pilot_rows)))
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.report_csv}")


if __name__ == "__main__":
    main()
