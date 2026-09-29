from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_CSV = ROOT / "artifacts" / "landmark_pilot.csv"
DEFAULT_METRICS_CSV = ROOT / "artifacts" / "landmark_metrics.csv"
DEFAULT_EVAL_CSV = ROOT / "artifacts" / "landmark_rule_eval.csv"
DEFAULT_ERRORS_CSV = ROOT / "artifacts" / "landmark_rule_errors.csv"

ROI_TOP_THRESHOLD_MM = 30.0
ROI_BOTTOM_THRESHOLD_MM = 30.0
ROI_LATERAL_THRESHOLD_MM = 20.0


@dataclass(frozen=True)
class EvalResult:
    task: str
    rule: str
    threshold: str
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    balanced_accuracy: float


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


def as_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def has_point(row: dict[str, str], key: str) -> bool:
    return row.get(f"{key}_x", "") != "" and row.get(f"{key}_y", "") != ""


def div(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate_predictions(task: str, rule: str, threshold: str, pairs: list[tuple[int, int]]) -> EvalResult:
    tp = sum(1 for label, pred in pairs if label == 1 and pred == 1)
    fp = sum(1 for label, pred in pairs if label == 0 and pred == 1)
    tn = sum(1 for label, pred in pairs if label == 0 and pred == 0)
    fn = sum(1 for label, pred in pairs if label == 1 and pred == 0)
    n = len(pairs)
    accuracy = div(tp + tn, n)
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2
    return EvalResult(task, rule, threshold, n, tp, fp, tn, fn, accuracy, precision, recall, specificity, f1, balanced_accuracy)


def result_row(result: EvalResult) -> dict[str, Any]:
    return {
        "task": result.task,
        "rule": result.rule,
        "threshold": result.threshold,
        "n": result.n,
        "tp": result.tp,
        "fp": result.fp,
        "tn": result.tn,
        "fn": result.fn,
        "accuracy": f"{result.accuracy:.3f}",
        "precision": f"{result.precision:.3f}",
        "recall": f"{result.recall:.3f}",
        "specificity": f"{result.specificity:.3f}",
        "f1": f"{result.f1:.3f}",
        "balanced_accuracy": f"{result.balanced_accuracy:.3f}",
    }


def merged_rows(pilot_rows: list[dict[str, str]], metric_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics_by_id = {row["pilot_id"]: row for row in metric_rows}
    rows = []
    for row in pilot_rows:
        merged = dict(row)
        merged.update({f"metric_{key}": value for key, value in metrics_by_id.get(row["pilot_id"], {}).items()})
        rows.append(merged)
    return rows


def hip_label(row: dict[str, str], suffix: str) -> int | None:
    if row["anatomical_region"] == "right_hip":
        return as_int(row.get(f"right_hip_{suffix}", ""))
    if row["anatomical_region"] == "left_hip":
        return as_int(row.get(f"left_hip_{suffix}", ""))
    return None


def bool_metric(row: dict[str, str], metric: str) -> int:
    return int(as_int(row.get(f"metric_{metric}", "")) == 1)


def threshold_candidates(values: list[float]) -> list[float]:
    vals = sorted(set(values))
    if not vals:
        return []
    candidates = [vals[0] - 1e-6, vals[-1] + 1e-6]
    candidates.extend((a + b) / 2 for a, b in zip(vals, vals[1:]))
    return sorted(set(candidates))


def best_threshold_rule(
    rows: list[dict[str, str]],
    task: str,
    rule: str,
    label_fn: Callable[[dict[str, str]], int | None],
    value_fn: Callable[[dict[str, str]], float | None],
    direction: str,
    missing_pred: int | None = None,
) -> tuple[EvalResult | None, list[dict[str, Any]]]:
    labeled = []
    values = []
    for row in rows:
        label = label_fn(row)
        value = value_fn(row)
        if label is None:
            continue
        labeled.append((row, label, value))
        if value is not None:
            values.append(value)
    best: EvalResult | None = None
    best_threshold: float | None = None
    for threshold in threshold_candidates(values):
        pairs = []
        for _row, label, value in labeled:
            if value is None:
                if missing_pred is None:
                    continue
                pred = missing_pred
            elif direction == "ge":
                pred = int(value >= threshold)
            elif direction == "le":
                pred = int(value <= threshold)
            else:
                raise ValueError(direction)
            pairs.append((label, pred))
        if not pairs:
            continue
        result = evaluate_predictions(task, rule, f"{threshold:.4f}", pairs)
        if best is None or (result.balanced_accuracy, result.f1, result.accuracy) > (best.balanced_accuracy, best.f1, best.accuracy):
            best = result
            best_threshold = threshold
    errors = []
    if best is not None and best_threshold is not None:
        for row, label, value in labeled:
            if value is None:
                if missing_pred is None:
                    continue
                pred = missing_pred
            elif direction == "ge":
                pred = int(value >= best_threshold)
            else:
                pred = int(value <= best_threshold)
            if pred != label:
                errors.append(error_row(task, rule, row, label, pred, value, f"{best_threshold:.4f}"))
    return best, errors


def error_row(
    task: str,
    rule: str,
    row: dict[str, str],
    label: int,
    pred: int,
    value: float | None,
    threshold: str,
) -> dict[str, Any]:
    return {
        "task": task,
        "rule": rule,
        "threshold": threshold,
        "pilot_id": row.get("pilot_id", ""),
        "image_number": row.get("image_number", ""),
        "anatomical_region": row.get("anatomical_region", ""),
        "label": label,
        "prediction": pred,
        "value": "" if value is None else f"{value:.4f}",
        "violation_type": row.get("violation_type", ""),
        "annotation_notes": row.get("annotation_notes", ""),
        "study_folder": row.get("study_folder", ""),
    }


def fixed_rule_eval(
    rows: list[dict[str, str]],
    task: str,
    rule: str,
    label_fn: Callable[[dict[str, str]], int | None],
    pred_fn: Callable[[dict[str, str]], int | None],
) -> tuple[EvalResult, list[dict[str, Any]]]:
    pairs = []
    raw = []
    for row in rows:
        label = label_fn(row)
        pred = pred_fn(row)
        if label is None or pred is None:
            continue
        pairs.append((label, pred))
        raw.append((row, label, pred))
    result = evaluate_predictions(task, rule, "fixed", pairs)
    errors = [error_row(task, rule, row, label, pred, None, "fixed") for row, label, pred in raw if label != pred]
    return result, errors


def evaluate(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    done = [row for row in rows if row.get("annotation_status") == "done"]
    eval_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    spine_rows = [row for row in done if row.get("pilot_group") == "spine"]
    hip_rows = [row for row in done if row.get("pilot_group") == "hip"]

    # Fixed rules from task specification / annotation design.
    result, errors = fixed_rule_eval(
        spine_rows,
        "spine_positioning",
        "bad_if_l5_or_iliac_missing",
        lambda row: as_int(row.get("spine_positioning", "")),
        lambda row: int(not (bool_metric(row, "spine_l5_visible") and bool_metric(row, "spine_both_iliac_visible"))),
    )
    eval_rows.append(result_row(result))
    error_rows.extend(errors)

    result, errors = fixed_rule_eval(
        hip_rows,
        "hip_roi",
        "tz_margins_top30_bottom30_lateral20_missing_bad",
        lambda row: hip_label(row, "roi"),
        lambda row: hip_roi_prediction(row),
    )
    eval_rows.append(result_row(result))
    error_rows.extend(errors)

    # Threshold searches.
    threshold_specs = [
        (
            spine_rows,
            "spine_axis",
            "angle_deg_ge_threshold",
            lambda row: as_int(row.get("spine_axis", "")),
            lambda row: as_float(row.get("metric_spine_axis_angle_deg", "")),
            "ge",
            None,
        ),
        (
            spine_rows,
            "spine_axis",
            "max_deviation_mm_ge_threshold",
            lambda row: as_int(row.get("spine_axis", "")),
            lambda row: as_float(row.get("metric_spine_axis_max_deviation_mm", "")),
            "ge",
            None,
        ),
        (
            spine_rows,
            "spine_axis",
            "mean_deviation_mm_ge_threshold",
            lambda row: as_int(row.get("spine_axis", "")),
            lambda row: as_float(row.get("metric_spine_axis_mean_deviation_mm", "")),
            "ge",
            None,
        ),
        (
            hip_rows,
            "hip_positioning_rotation",
            "lesser_prominence_mm_ge_threshold_missing_bad",
            lambda row: hip_label(row, "positioning_rotation"),
            lambda row: as_float(row.get("metric_hip_lesser_prominence_mm", "")),
            "ge",
            1,
        ),
        (
            hip_rows,
            "hip_positioning_rotation",
            "lesser_ratio_ge_threshold_missing_bad",
            lambda row: hip_label(row, "positioning_rotation"),
            lambda row: as_float(row.get("metric_hip_lesser_prominence_ratio", "")),
            "ge",
            1,
        ),
        (
            hip_rows,
            "hip_roi",
            "top_margin_mm_le_threshold",
            lambda row: hip_label(row, "roi"),
            lambda row: as_float(row.get("metric_hip_top_margin_mm", "")),
            "le",
            1,
        ),
        (
            hip_rows,
            "hip_roi",
            "lateral_margin_mm_le_threshold",
            lambda row: hip_label(row, "roi"),
            lambda row: as_float(row.get("metric_hip_lateral_margin_mm", "")),
            "le",
            1,
        ),
        (
            hip_rows,
            "hip_roi",
            "bottom_margin_mm_le_threshold",
            lambda row: hip_label(row, "roi"),
            lambda row: as_float(row.get("metric_hip_bottom_margin_mm", "")),
            "le",
            1,
        ),
    ]
    for spec in threshold_specs:
        best, errors = best_threshold_rule(*spec)
        if best is not None:
            eval_rows.append(result_row(best))
            error_rows.extend(errors)

    best, errors = best_hip_roi_composite_rule(hip_rows)
    if best is not None:
        eval_rows.append(result_row(best))
        error_rows.extend(errors)

    eval_rows.sort(key=lambda row: (row["task"], -float(row["balanced_accuracy"]), -float(row["f1"]), row["rule"]))
    return eval_rows, error_rows


def hip_roi_prediction(row: dict[str, str]) -> int:
    top = as_float(row.get("metric_hip_top_margin_mm", ""))
    lateral = as_float(row.get("metric_hip_lateral_margin_mm", ""))
    bottom = as_float(row.get("metric_hip_bottom_margin_mm", ""))
    if top is None or lateral is None or bottom is None:
        return 1
    return int(top < ROI_TOP_THRESHOLD_MM or lateral < ROI_LATERAL_THRESHOLD_MM or bottom < ROI_BOTTOM_THRESHOLD_MM)




def best_hip_roi_composite_rule(rows: list[dict[str, str]]) -> tuple[EvalResult | None, list[dict[str, Any]]]:
    labeled = []
    top_values = []
    lateral_values = []
    bottom_values = []
    for row in rows:
        label = hip_label(row, "roi")
        if label is None:
            continue
        top = as_float(row.get("metric_hip_top_margin_mm", ""))
        lateral = as_float(row.get("metric_hip_lateral_margin_mm", ""))
        bottom = as_float(row.get("metric_hip_bottom_margin_mm", ""))
        labeled.append((row, label, top, lateral, bottom))
        if top is not None:
            top_values.append(top)
        if lateral is not None:
            lateral_values.append(lateral)
        if bottom is not None:
            bottom_values.append(bottom)

    best: EvalResult | None = None
    best_thresholds: tuple[float, float, float] | None = None
    for top_t in threshold_candidates(top_values):
        for lateral_t in threshold_candidates(lateral_values):
            for bottom_t in threshold_candidates(bottom_values):
                pairs = []
                for _row, label, top, lateral, bottom in labeled:
                    pred = int(
                        top is None
                        or lateral is None
                        or bottom is None
                        or top <= top_t
                        or lateral <= lateral_t
                        or bottom <= bottom_t
                    )
                    pairs.append((label, pred))
                result = evaluate_predictions(
                    "hip_roi",
                    "composite_any_margin_le_threshold_missing_bad",
                    f"top<={top_t:.3f};lateral<={lateral_t:.3f};bottom<={bottom_t:.3f}",
                    pairs,
                )
                if best is None or (result.balanced_accuracy, result.f1, result.accuracy) > (best.balanced_accuracy, best.f1, best.accuracy):
                    best = result
                    best_thresholds = (top_t, lateral_t, bottom_t)

    errors = []
    if best is not None and best_thresholds is not None:
        top_t, lateral_t, bottom_t = best_thresholds
        threshold_text = f"top<={top_t:.3f};lateral<={lateral_t:.3f};bottom<={bottom_t:.3f}"
        for row, label, top, lateral, bottom in labeled:
            pred = int(
                top is None
                or lateral is None
                or bottom is None
                or top <= top_t
                or lateral <= lateral_t
                or bottom <= bottom_t
            )
            if pred != label:
                value_parts = [
                    "" if top is None else f"top={top:.3f}",
                    "" if lateral is None else f"lateral={lateral:.3f}",
                    "" if bottom is None else f"bottom={bottom:.3f}",
                ]
                errors.append(error_row("hip_roi", "composite_any_margin_le_threshold_missing_bad", row, label, pred, None, threshold_text))
                errors[-1]["value"] = ";".join(part for part in value_parts if part)
    return best, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-csv", type=Path, default=DEFAULT_PILOT_CSV)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merged_rows(read_csv(args.pilot_csv), read_csv(args.metrics_csv))
    eval_rows, error_rows = evaluate(rows)
    write_csv(args.eval_csv, eval_rows)
    write_csv(
        args.errors_csv,
        error_rows,
        fieldnames=["task", "rule", "threshold", "pilot_id", "image_number", "anatomical_region", "label", "prediction", "value", "violation_type", "annotation_notes", "study_folder"],
    )
    print(f"Rule rows: {len(eval_rows)}")
    print(f"Error rows: {len(error_rows)}")
    print(f"Wrote: {args.eval_csv}")
    print(f"Wrote: {args.errors_csv}")


if __name__ == "__main__":
    main()
