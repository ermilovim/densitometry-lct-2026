from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ERRORS_CSV = ROOT / "artifacts" / "landmark_all_rule_errors.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "label_audit.csv"

PRIMARY_RULES = {
    ("spine_positioning", "bad_if_l5_or_iliac_missing"),
    ("hip_positioning_rotation", "lesser_prominence_mm_ge_threshold_missing_bad"),
    ("hip_roi", "composite_any_margin_le_threshold_missing_bad"),
    ("spine_axis", "angle_deg_ge_threshold"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def error_kind(row: dict[str, str]) -> str:
    if row.get("label") == "1" and row.get("prediction") == "0":
        return "FN"
    if row.get("label") == "0" and row.get("prediction") == "1":
        return "FP"
    return "mismatch"


def build_audit_rows(error_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    seen: set[tuple[str, str, str]] = set()
    for row in error_rows:
        key = (row.get("task", ""), row.get("rule", ""))
        if key not in PRIMARY_RULES:
            continue
        dedup_key = (row.get("task", ""), row.get("pilot_id", ""), row.get("image_number", ""))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(
            {
                "audit_status": "",
                "audit_decision": "",
                "corrected_label": "",
                "audit_notes": "",
                "task": row.get("task", ""),
                "rule": row.get("rule", ""),
                "error_kind": error_kind(row),
                "pilot_id": row.get("pilot_id", ""),
                "image_number": row.get("image_number", ""),
                "anatomical_region": row.get("anatomical_region", ""),
                "expert_label": row.get("label", ""),
                "rule_prediction": row.get("prediction", ""),
                "value": row.get("value", ""),
                "threshold": row.get("threshold", ""),
                "violation_type": row.get("violation_type", ""),
                "annotation_notes": row.get("annotation_notes", ""),
                "study_folder": row.get("study_folder", ""),
            }
        )
    return sorted(out, key=lambda r: (r["task"], r["error_kind"], int(r["pilot_id"] or 0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_audit_rows(read_csv(args.errors_csv))
    fieldnames = [
        "audit_status",
        "audit_decision",
        "corrected_label",
        "audit_notes",
        "task",
        "rule",
        "error_kind",
        "pilot_id",
        "image_number",
        "anatomical_region",
        "expert_label",
        "rule_prediction",
        "value",
        "threshold",
        "violation_type",
        "annotation_notes",
        "study_folder",
    ]
    write_csv(args.out_csv, rows, fieldnames=fieldnames)
    print(f"Rows: {len(rows)}")
    print(f"Wrote: {args.out_csv}")
    print("audit_decision values: keep_expert, fix_label, fix_rule, ignore_unclear")
    print("corrected_label values: 0 or 1, only when audit_decision=fix_label")


if __name__ == "__main__":
    main()
