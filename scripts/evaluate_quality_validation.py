from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from apply_quality_rules import evaluate_canonical
from evaluate_landmark_knn import (
    HIP_KEYS,
    SPINE_KEYS,
    load_vectors,
    nearest_rows,
    point_from_neighbors,
)
from predict_landmarks_heatmap import load_model, predict_image
from predict_spine_classifiers import load_classifier, predict_bad_probability


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "artifacts" / "keypoint_dataset"
DEFAULT_ANNOTATIONS_CSV = DEFAULT_DATASET_DIR / "annotations.csv"
DEFAULT_HIP_MODEL = ROOT / "artifacts" / "landmark_heatmap" / "hip" / "best.pt"
DEFAULT_SPINE_POSITIONING_MODEL = ROOT / "artifacts" / "spine_classifiers" / "positioning" / "best.pt"
DEFAULT_SPINE_ARTIFACT_MODEL = ROOT / "artifacts" / "spine_classifiers" / "artifact" / "best.pt"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "quality_validation_predictions.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "quality_validation_report.csv"
DEFAULT_ERRORS_CSV = ROOT / "artifacts" / "quality_validation_errors.csv"

ALL_KEYS = SPINE_KEYS + HIP_KEYS


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


def empty_landmark_row(row: dict[str, str], model_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "member": row["member"],
        "filename": row["filename"],
        "anatomical_region": row["anatomical_region"],
        "landmark_model": model_name,
        "landmark_status": "predicted",
    }
    for key in ALL_KEYS:
        out[f"{key}_x"] = ""
        out[f"{key}_y"] = ""
        out[f"{key}_visible"] = "0"
    return out


def predict_hip_landmarks(
    row: dict[str, str],
    dataset_dir: Path,
    hip_bundle: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    out = empty_landmark_row(row, "heatmap_hip")
    image = Image.open(dataset_dir / row["image_path"])
    points = predict_image(image, hip_bundle, device, row.get("anatomical_region", ""))
    for key, (x, y) in points.items():
        out[f"{key}_x"] = f"{x:.3f}"
        out[f"{key}_y"] = f"{y:.3f}"
        out[f"{key}_visible"] = "1"
    return out


def predict_spine_landmarks(
    row: dict[str, str],
    train_by_region: dict[str, list[dict[str, str]]],
    vectors: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    out = empty_landmark_row(row, "knn_spine")
    neighbors = nearest_rows(row, train_by_region["spine"], vectors, exclude_self=False, k=k)
    for key in SPINE_KEYS:
        point = point_from_neighbors(neighbors, row, key)
        if point is None:
            continue
        out[f"{key}_x"] = f"{point[0]:.3f}"
        out[f"{key}_y"] = f"{point[1]:.3f}"
        out[f"{key}_visible"] = "1"
    return out


def label_violations(row: dict[str, str]) -> list[str]:
    region = row["anatomical_region"]
    if region == "spine":
        fields = ["spine_positioning", "spine_axis", "spine_artifact"]
    elif region in {"left_hip", "right_hip"}:
        fields = ["hip_positioning_rotation", "hip_roi"]
    else:
        fields = []
    return [field for field in fields if as_int(row.get(field, "")) == 1]


def metric_row(name: str, items: list[tuple[int, int]]) -> dict[str, Any]:
    tp = sum(1 for gt, pred in items if gt == 1 and pred == 1)
    fp = sum(1 for gt, pred in items if gt == 0 and pred == 1)
    fn = sum(1 for gt, pred in items if gt == 1 and pred == 0)
    tn = sum(1 for gt, pred in items if gt == 0 and pred == 0)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    balanced_accuracy = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "target": name,
        "n": len(items),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": "" if precision is None else f"{precision:.3f}",
        "recall": "" if recall is None else f"{recall:.3f}",
        "specificity": "" if specificity is None else f"{specificity:.3f}",
        "f1": "" if f1 is None else f"{f1:.3f}",
        "balanced_accuracy": "" if balanced_accuracy is None else f"{balanced_accuracy:.3f}",
    }


def spine_classifier_prediction_row(
    row: dict[str, str],
    dataset_dir: Path,
    positioning_bundle: dict[str, Any] | None,
    artifact_bundle: dict[str, Any] | None,
    device: torch.device,
) -> dict[str, str] | None:
    if positioning_bundle is None or artifact_bundle is None:
        return None
    image = Image.open(dataset_dir / row["image_path"])
    positioning_prob = predict_bad_probability(image, positioning_bundle, device)
    artifact_prob = predict_bad_probability(image, artifact_bundle, device)
    positioning_threshold = float(positioning_bundle["threshold"])
    artifact_threshold = float(artifact_bundle["threshold"])
    return {
        "member": row["member"],
        "spine_classifier_status": "predicted",
        "spine_positioning_prob_bad": f"{positioning_prob:.6f}",
        "spine_positioning_threshold": f"{positioning_threshold:.6f}",
        "spine_positioning_pred_bad": str(int(positioning_prob + 1e-6 >= positioning_threshold)),
        "spine_artifact_prob_bad": f"{artifact_prob:.6f}",
        "spine_artifact_threshold": f"{artifact_threshold:.6f}",
        "spine_artifact_pred_bad": str(int(artifact_prob + 1e-6 >= artifact_threshold)),
    }


def evaluate_rows(
    rows: list[dict[str, str]],
    dataset_dir: Path,
    hip_bundle: dict[str, Any],
    positioning_bundle: dict[str, Any] | None,
    artifact_bundle: dict[str, Any] | None,
    device: torch.device,
    knn_size: int,
    k: int,
) -> list[dict[str, Any]]:
    vectors = load_vectors(rows, dataset_dir, knn_size)
    train_by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            train_by_region[row["anatomical_region"]].append(row)

    out_rows: list[dict[str, Any]] = []
    for row in [item for item in rows if item["split"] == "val"]:
        region = row["anatomical_region"]
        if region in {"left_hip", "right_hip"}:
            landmark = predict_hip_landmarks(row, dataset_dir, hip_bundle, device)
        elif region == "spine":
            landmark = predict_spine_landmarks(row, train_by_region, vectors, k)
        else:
            continue

        inference_row = {
            "anatomical_region": region,
            "columns": row["width"],
            "rows": row["height"],
        }
        spine_classifier = None
        if region == "spine":
            spine_classifier = spine_classifier_prediction_row(
                row,
                dataset_dir,
                positioning_bundle,
                artifact_bundle,
                device,
            )
        pred = evaluate_canonical(inference_row, landmark, "validation_hybrid", spine_classifier)
        pred_violations = [item for item in str(pred["violation_type"]).split(";") if item]
        gt_quality = as_int(row.get("quality_class", ""))
        pred_quality = int(pred["quality_class"])
        gt_violations = label_violations(row)

        out_rows.append(
            {
                "split": row["split"],
                "pilot_id": row["pilot_id"],
                "image_number": row["image_number"],
                "study_folder": row["study_folder"],
                "filename": row["filename"],
                "anatomical_region": region,
                "quality_true": "" if gt_quality is None else gt_quality,
                "quality_pred": pred_quality,
                "quality_match": "" if gt_quality is None else int(gt_quality == pred_quality),
                "violations_true": ";".join(gt_violations),
                "violations_pred": ";".join(pred_violations),
                "landmark_model": landmark["landmark_model"],
                "spine_positioning_true": row.get("spine_positioning", ""),
                "spine_positioning_pred": int("spine_positioning" in pred_violations),
                "spine_axis_true": row.get("spine_axis", ""),
                "spine_axis_pred": int("spine_axis" in pred_violations),
                "spine_artifact_true": row.get("spine_artifact", ""),
                "spine_artifact_pred": int("spine_artifact" in pred_violations),
                "spine_positioning_prob_bad": pred.get("spine_positioning_prob_bad", ""),
                "spine_artifact_prob_bad": pred.get("spine_artifact_prob_bad", ""),
                "hip_positioning_rotation_true": row.get("hip_positioning_rotation", ""),
                "hip_positioning_rotation_pred": int("hip_positioning_rotation" in pred_violations),
                "hip_roi_true": row.get("hip_roi", ""),
                "hip_roi_pred": int("hip_roi" in pred_violations),
                "spine_axis_angle_deg": pred.get("spine_axis_angle_deg", ""),
                "hip_lesser_prominence_mm": pred.get("hip_lesser_prominence_mm", ""),
                "hip_top_margin_mm": pred.get("hip_top_margin_mm", ""),
                "hip_lateral_margin_mm": pred.get("hip_lateral_margin_mm", ""),
                "hip_bottom_margin_mm": pred.get("hip_bottom_margin_mm", ""),
            }
        )
    return out_rows


def build_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = ["quality"]
    for field in ["spine_positioning", "spine_axis", "spine_artifact", "hip_positioning_rotation", "hip_roi"]:
        targets.append(field)

    report: list[dict[str, Any]] = []
    for target in targets:
        items: list[tuple[int, int]] = []
        true_key = "quality_true" if target == "quality" else f"{target}_true"
        pred_key = "quality_pred" if target == "quality" else f"{target}_pred"
        for row in rows:
            gt = as_int(str(row.get(true_key, "")))
            pred = as_int(str(row.get(pred_key, "")))
            if gt is None or pred is None:
                continue
            if target.startswith("spine") and row["anatomical_region"] != "spine":
                continue
            if target.startswith("hip") and row["anatomical_region"] not in {"left_hip", "right_hip"}:
                continue
            items.append((gt, pred))
        report.append(metric_row(target, items))

    for region in ["spine", "left_hip", "right_hip"]:
        items = []
        for row in rows:
            if row["anatomical_region"] != region:
                continue
            gt = as_int(str(row.get("quality_true", "")))
            pred = as_int(str(row.get("quality_pred", "")))
            if gt is not None and pred is not None:
                items.append((gt, pred))
        report.append(metric_row(f"quality_{region}", items))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--hip-model", type=Path, default=DEFAULT_HIP_MODEL)
    parser.add_argument("--spine-positioning-model", type=Path, default=DEFAULT_SPINE_POSITIONING_MODEL)
    parser.add_argument("--spine-artifact-model", type=Path, default=DEFAULT_SPINE_ARTIFACT_MODEL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--knn-size", type=int, default=96)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    hip_bundle = load_model(args.hip_model, device)
    if hip_bundle is None:
        raise RuntimeError(f"Hip model not found: {args.hip_model}")
    positioning_bundle = load_classifier(args.spine_positioning_model, device)
    artifact_bundle = load_classifier(args.spine_artifact_model, device)
    if positioning_bundle is None:
        raise RuntimeError(f"Spine positioning model not found: {args.spine_positioning_model}")
    if artifact_bundle is None:
        raise RuntimeError(f"Spine artifact model not found: {args.spine_artifact_model}")
    rows = read_csv(args.annotations_csv)
    out_rows = evaluate_rows(
        rows,
        args.dataset_dir,
        hip_bundle,
        positioning_bundle,
        artifact_bundle,
        device,
        args.knn_size,
        args.k,
    )
    report = build_report(out_rows)
    errors = [row for row in out_rows if row["quality_match"] != 1]

    write_csv(args.out_csv, out_rows)
    write_csv(args.report_csv, report)
    write_csv(args.errors_csv, errors)

    print(f"Validation rows: {len(out_rows)}")
    print(f"Quality mismatches: {len(errors)}")
    for row in report:
        if row["target"] in {"quality", "quality_spine", "quality_left_hip", "quality_right_hip"}:
            print(
                f"{row['target']}: n={row['n']} "
                f"precision={row['precision']} recall={row['recall']} "
                f"f1={row['f1']} balanced_accuracy={row['balanced_accuracy']}"
            )
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.report_csv}")
    print(f"Wrote: {args.errors_csv}")


if __name__ == "__main__":
    main()
