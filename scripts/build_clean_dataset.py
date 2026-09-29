from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_REVIEW_CSV = DEFAULT_ARTIFACTS_DIR / "manual_image_review.csv"
DEFAULT_INDEX_CSV = DEFAULT_ARTIFACTS_DIR / "dicom_index.csv"
DEFAULT_LABELS_CSV = DEFAULT_ARTIFACTS_DIR / "labels.csv"
DEFAULT_CLEAN_CSV = DEFAULT_ARTIFACTS_DIR / "clean_image_index.csv"
DEFAULT_REPORT_CSV = DEFAULT_ARTIFACTS_DIR / "clean_dataset_report.csv"
DEFAULT_ISSUES_CSV = DEFAULT_ARTIFACTS_DIR / "clean_dataset_issues.csv"

REGIONS = {"spine", "right_hip", "left_hip"}

REGION_LABELS = {
    "spine": {
        "quality": "spine_quality_or",
        "criteria": [
            ("spine_positioning", "spine_positioning"),
            ("spine_axis", "spine_axis"),
            ("spine_artifact", "spine_artifact"),
        ],
    },
    "right_hip": {
        "quality": "right_hip_quality_or",
        "criteria": [
            ("right_hip_positioning_rotation", "hip_positioning_rotation"),
            ("right_hip_roi", "hip_roi"),
        ],
    },
    "left_hip": {
        "quality": "left_hip_quality_or",
        "criteria": [
            ("left_hip_positioning_rotation", "hip_positioning_rotation"),
            ("left_hip_roi", "hip_roi"),
        ],
    },
}

LABEL_COLUMNS = [
    "spine_positioning",
    "spine_axis",
    "spine_artifact",
    "right_hip_positioning_rotation",
    "right_hip_roi",
    "left_hip_positioning_rotation",
    "left_hip_roi",
    "spine_quality_or",
    "right_hip_quality_or",
    "left_hip_quality_or",
    "raw_spine_quality",
    "raw_right_hip_quality",
    "raw_left_hip_quality",
]

INDEX_COLUMNS = [
    "archive",
    "path_to_study",
    "series_folder",
    "file_size",
    "study_uid",
    "series_uid",
    "image_uid",
    "modality",
    "manufacturer",
    "software_versions",
    "study_description",
    "series_description",
    "protocol_name",
    "photometric",
    "bits_allocated",
    "bits_stored",
    "pixel_spacing",
    "imager_pixel_spacing",
    "pixel_aspect_ratio",
    "private_tag_count",
]


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


def is_nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def as_binary(value: Any) -> int | None:
    if not is_nonempty(value):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def label_status(split: str, label: dict[str, str] | None, region: str) -> str:
    if split == "test":
        return "unlabeled_test"
    if label is None:
        return "missing_study_label"
    spec = REGION_LABELS[region]
    values = [label.get(column, "") for column, _name in spec["criteria"]]
    values.append(label.get(spec["quality"], ""))
    if not any(is_nonempty(value) for value in values):
        return "missing_region_label"
    return "labeled"


def quality_and_violations(label: dict[str, str] | None, region: str) -> tuple[str, str]:
    if label is None:
        return "", ""
    spec = REGION_LABELS[region]
    quality = label.get(spec["quality"], "")
    violations = [
        name
        for column, name in spec["criteria"]
        if as_binary(label.get(column, "")) == 1
    ]
    return quality, ";".join(violations)


def issue_row(issue_type: str, row: dict[str, str], details: str = "") -> dict[str, Any]:
    return {
        "issue_type": issue_type,
        "split": row.get("split", ""),
        "study_folder": row.get("study_folder", ""),
        "manual_region": row.get("manual_region", ""),
        "member": row.get("member", ""),
        "filename": row.get("filename", ""),
        "details": details,
    }


def build_clean_rows(
    review_rows: list[dict[str, str]],
    index_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index_by_member = {row["member"]: row for row in index_rows}
    labels_by_study = {row["study_uid"]: row for row in label_rows}

    clean_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for image_number, row in enumerate(review_rows, start=1):
        region = row.get("manual_region", "")
        is_canonical = row.get("manual_is_canonical", "")
        if is_canonical != "1" or region not in REGIONS:
            continue

        index = index_by_member.get(row["member"])
        label = labels_by_study.get(row["study_folder"])
        if index is None:
            issues.append(issue_row("missing_dicom_index_row", row))
            index = {}
        if row.get("split") == "train" and label is None:
            issues.append(issue_row("missing_study_label", row))

        quality_class, violation_type = quality_and_violations(label, region)
        status = label_status(row.get("split", ""), label, region)
        if status == "missing_region_label":
            issues.append(issue_row("missing_region_label", row))

        clean_row: dict[str, Any] = {
            "image_number": image_number,
            "split": row.get("split", ""),
            "study_folder": row.get("study_folder", ""),
            "anatomical_region": region,
            "member": row.get("member", ""),
            "filename": row.get("filename", ""),
            "columns": row.get("columns", ""),
            "rows": row.get("rows", ""),
            "region_hint": row.get("region_hint", ""),
            "candidate_rank_in_region": row.get("candidate_rank_in_region", ""),
            "manual_notes": row.get("manual_notes", ""),
            "label_status": status,
            "label_row_no": "" if label is None else label.get("row_no", ""),
            "label_study_uid": "" if label is None else label.get("study_uid", ""),
            "quality_class": quality_class,
            "violation_type": violation_type,
        }

        for column in INDEX_COLUMNS:
            clean_row[f"dicom_{column}"] = index.get(column, "")
        for column in LABEL_COLUMNS:
            clean_row[column] = "" if label is None else label.get(column, "")
        clean_rows.append(clean_row)

    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_key[(row["split"], row["study_folder"], row["anatomical_region"])].append(row)
    for key, rows in by_key.items():
        if len(rows) <= 1:
            continue
        details = f"{key[0]} / {key[1]} / {key[2]} has {len(rows)} canonical rows"
        for row in rows:
            issues.append(
                {
                    "issue_type": "duplicate_canonical_region",
                    "split": row["split"],
                    "study_folder": row["study_folder"],
                    "manual_region": row["anatomical_region"],
                    "member": row["member"],
                    "filename": row["filename"],
                    "details": details,
                }
            )

    clean_keys = {
        (row["split"], row["study_folder"], row["anatomical_region"])
        for row in clean_rows
    }
    for label in label_rows:
        for region, spec in REGION_LABELS.items():
            quality = label.get(spec["quality"], "")
            criteria = [label.get(column, "") for column, _name in spec["criteria"]]
            if not any(is_nonempty(value) for value in [quality, *criteria]):
                continue
            key = ("train", label["study_uid"], region)
            if key in clean_keys:
                continue
            issues.append(
                {
                    "issue_type": "label_without_canonical_image",
                    "split": "train",
                    "study_folder": label["study_uid"],
                    "manual_region": region,
                    "member": "",
                    "filename": "",
                    "details": f"label exists for {region}, but no canonical image is selected",
                }
            )

    report = build_report(review_rows, clean_rows, issues)
    return clean_rows, report, issues


def add_report(report: list[dict[str, Any]], section: str, metric: str, value: Any) -> None:
    report.append({"section": section, "metric": metric, "value": value})


def build_report(
    review_rows: list[dict[str, str]],
    clean_rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []

    add_report(report, "rows", "manual_review_total", len(review_rows))
    add_report(report, "rows", "clean_total", len(clean_rows))
    for split, count in sorted(Counter(row["split"] for row in clean_rows).items()):
        add_report(report, "rows_by_split", split, count)
    for region, count in sorted(Counter(row["anatomical_region"] for row in clean_rows).items()):
        add_report(report, "rows_by_region", region, count)
    for key, count in sorted(Counter((row["split"], row["anatomical_region"]) for row in clean_rows).items()):
        add_report(report, "rows_by_split_region", " / ".join(key), count)

    excluded = [
        row
        for row in review_rows
        if row.get("manual_is_canonical") != "1" or row.get("manual_region") not in REGIONS
    ]
    for region, count in sorted(Counter(row.get("manual_region", "") for row in excluded).items()):
        add_report(report, "excluded_manual_region", region or "<empty>", count)

    for status, count in sorted(Counter(row["label_status"] for row in clean_rows).items()):
        add_report(report, "label_status", status, count)

    train_rows = [row for row in clean_rows if row["split"] == "train"]
    for region in sorted(REGIONS):
        region_rows = [row for row in train_rows if row["anatomical_region"] == region]
        quality_counts = Counter(row["quality_class"] if row["quality_class"] != "" else "<empty>" for row in region_rows)
        for quality, count in sorted(quality_counts.items()):
            add_report(report, f"{region}_quality_class", quality, count)

    criterion_specs = [
        ("spine", "spine_positioning"),
        ("spine", "spine_axis"),
        ("spine", "spine_artifact"),
        ("right_hip", "right_hip_positioning_rotation"),
        ("right_hip", "right_hip_roi"),
        ("left_hip", "left_hip_positioning_rotation"),
        ("left_hip", "left_hip_roi"),
    ]
    for region, column in criterion_specs:
        region_rows = [row for row in train_rows if row["anatomical_region"] == region]
        values = Counter(row[column] if row[column] != "" else "<empty>" for row in region_rows)
        for value, count in sorted(values.items()):
            add_report(report, f"{region}_criterion_{column}", value, count)

    for issue_type, count in sorted(Counter(row["issue_type"] for row in issues).items()):
        add_report(report, "issues", issue_type, count)
    add_report(report, "issues", "total", len(issues))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--clean-csv", type=Path, default=DEFAULT_CLEAN_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--issues-csv", type=Path, default=DEFAULT_ISSUES_CSV)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_rows = read_csv(args.review_csv)
    index_rows = read_csv(args.index_csv)
    label_rows = read_csv(args.labels_csv)

    clean_rows, report, issues = build_clean_rows(review_rows, index_rows, label_rows)

    write_csv(args.clean_csv, clean_rows)
    write_csv(args.report_csv, report, fieldnames=["section", "metric", "value"])
    write_csv(
        args.issues_csv,
        issues,
        fieldnames=["issue_type", "split", "study_folder", "manual_region", "member", "filename", "details"],
    )

    print(f"Clean rows: {len(clean_rows)}")
    print("By split:", dict(Counter(row["split"] for row in clean_rows)))
    print("By region:", dict(Counter(row["anatomical_region"] for row in clean_rows)))
    print("Label status:", dict(Counter(row["label_status"] for row in clean_rows)))
    print("Issues:", dict(Counter(row["issue_type"] for row in issues)))
    print(f"Wrote: {args.clean_csv}")
    print(f"Wrote: {args.report_csv}")
    print(f"Wrote: {args.issues_csv}")

    if args.strict and issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
