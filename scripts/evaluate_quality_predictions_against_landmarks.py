from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANDMARKS_CSV = ROOT / "artifacts" / "landmark_all.csv"
DEFAULT_PREDICTIONS_CSV = ROOT / "artifacts" / "quality_predictions_train_hybrid.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "quality_train_landmarked_predictions.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "quality_train_landmarked_report.csv"
DEFAULT_ERRORS_CSV = ROOT / "artifacts" / "quality_train_landmarked_errors.csv"


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


def as_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def true_value(row: dict[str, str], target: str) -> int | None:
    region = row["anatomical_region"]
    if target == "quality":
        return as_int(row.get("quality_class", ""))
    if region == "spine" and target in {"spine_positioning", "spine_axis", "spine_artifact"}:
        return as_int(row.get(target, ""))
    if region == "left_hip":
        mapping = {
            "hip_positioning_rotation": "left_hip_positioning_rotation",
            "hip_roi": "left_hip_roi",
        }
        return as_int(row.get(mapping.get(target, ""), ""))
    if region == "right_hip":
        mapping = {
            "hip_positioning_rotation": "right_hip_positioning_rotation",
            "hip_roi": "right_hip_roi",
        }
        return as_int(row.get(mapping.get(target, ""), ""))
    return None


def pred_value(pred: dict[str, str], target: str) -> int | None:
    if target == "quality":
        return as_int(pred.get("quality_class", ""))
    violations = {item for item in pred.get("violation_type", "").split(";") if item}
    return int(target in violations)


def metric_row(split: str, target: str, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    tp = sum(1 for y, p in pairs if y == 1 and p == 1)
    fp = sum(1 for y, p in pairs if y == 0 and p == 1)
    fn = sum(1 for y, p in pairs if y == 1 and p == 0)
    tn = sum(1 for y, p in pairs if y == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    ba = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "split": split,
        "target": target,
        "n": len(pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": "" if precision is None else f"{precision:.3f}",
        "recall": "" if recall is None else f"{recall:.3f}",
        "specificity": "" if specificity is None else f"{specificity:.3f}",
        "f1": "" if f1 is None else f"{f1:.3f}",
        "balanced_accuracy": "" if ba is None else f"{ba:.3f}",
    }


def build_rows(landmark_rows: list[dict[str, str]], prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    pred_by_member = {row["member"]: row for row in prediction_rows}
    out: list[dict[str, Any]] = []
    for row in landmark_rows:
        if row.get("annotation_status") != "done":
            continue
        pred = pred_by_member.get(row["member"])
        if pred is None:
            continue
        region = row["anatomical_region"]
        targets = ["quality"]
        if region == "spine":
            targets += ["spine_positioning", "spine_axis", "spine_artifact"]
        elif region in {"left_hip", "right_hip"}:
            targets += ["hip_positioning_rotation", "hip_roi"]
        item: dict[str, Any] = {
            "split": row.get("split", ""),
            "pilot_id": row.get("pilot_id", ""),
            "image_number": row.get("image_number", ""),
            "study_folder": row.get("study_folder", ""),
            "member": row.get("member", ""),
            "filename": row.get("filename", ""),
            "anatomical_region": region,
            "is_canonical_for_region": pred.get("is_canonical_for_region", ""),
            "is_duplicate_or_derived": pred.get("is_duplicate_or_derived", ""),
            "quality_true": true_value(row, "quality"),
            "quality_pred": pred_value(pred, "quality"),
            "violations_true": row.get("violation_type", ""),
            "violations_pred": pred.get("violation_type", ""),
            "processing_status": pred.get("processing_status", ""),
        }
        for target in ["spine_positioning", "spine_axis", "spine_artifact", "hip_positioning_rotation", "hip_roi"]:
            y = true_value(row, target)
            p = pred_value(pred, target) if y is not None else None
            item[f"{target}_true"] = "" if y is None else y
            item[f"{target}_pred"] = "" if p is None else p
        item["quality_match"] = "" if item["quality_true"] is None or item["quality_pred"] is None else int(item["quality_true"] == item["quality_pred"])
        out.append(item)
    return out


def build_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = ["quality", "spine_positioning", "spine_axis", "spine_artifact", "hip_positioning_rotation", "hip_roi"]
    report: list[dict[str, Any]] = []
    for split in ["ALL", "train", "val"]:
        split_rows = rows if split == "ALL" else [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        for target in targets:
            pairs: list[tuple[int, int]] = []
            true_key = f"{target}_true" if target != "quality" else "quality_true"
            pred_key = f"{target}_pred" if target != "quality" else "quality_pred"
            for row in split_rows:
                y = row.get(true_key, "")
                p = row.get(pred_key, "")
                if y == "" or p == "" or y is None or p is None:
                    continue
                pairs.append((int(y), int(p)))
            if pairs:
                report.append(metric_row(split, target, pairs))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-csv", type=Path, default=DEFAULT_LANDMARKS_CSV)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(read_csv(args.landmarks_csv), read_csv(args.predictions_csv))
    report = build_report(rows)
    errors = [row for row in rows if row.get("quality_match") != 1]
    write_csv(args.out_csv, rows)
    write_csv(args.report_csv, report)
    write_csv(args.errors_csv, errors)
    print(f"Rows: {len(rows)}")
    print(f"Quality errors: {len(errors)}")
    for row in report:
        if row["split"] in {"ALL", "val"} and row["target"] in {"quality", "hip_roi", "hip_positioning_rotation"}:
            print(f"{row['split']} {row['target']}: n={row['n']} TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']} BA={row['balanced_accuracy']} F1={row['f1']}")
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.report_csv}")
    print(f"Wrote: {args.errors_csv}")


if __name__ == "__main__":
    main()
