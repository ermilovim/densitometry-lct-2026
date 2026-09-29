from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_CSV = ROOT / "artifacts" / "submission.csv"
DEFAULT_WORK_DIR = ROOT / "artifacts" / "full_inference"


def run_step(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_for(row: dict[str, str]) -> str:
    if row.get("quality_class", "") == "":
        return "Failure"
    if row.get("processing_status", "").startswith("quality_predicted"):
        return "Success"
    return "Failure"


def image_processing_times(paths: list[Path]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for path in paths:
        for row in read_csv(path):
            try:
                seconds = float(row.get("processing_seconds", "0"))
            except ValueError:
                seconds = 0.0
            member = row.get("member", "")
            totals[member] = totals.get(member, 0.0) + max(seconds, 0.0)
    return totals


def export_submission(
    debug_csv: Path,
    out_csv: Path,
    elapsed_seconds: float,
    timing_csvs: list[Path] | None = None,
) -> None:
    rows = read_csv(debug_csv)
    timings = image_processing_times(timing_csvs or [])
    fallback_seconds = elapsed_seconds / max(len(rows), 1)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out_rows.append(
            {
                "path_to_study": row.get("path_to_study", "") or row.get("member", ""),
                "study_uid": row.get("dicom_study_uid", "") or row.get("study_key", ""),
                "image_uid": row.get("image_uid", ""),
                "anatomical_region": row.get("anatomical_region", ""),
                "quality_class": row.get("quality_class", ""),
                "violation_type": row.get("violation_type", ""),
                "processing_status": status_for(row),
                "time_of_processing": f"{timings.get(row.get('member', ''), fallback_seconds):.6f}",
            }
        )
    write_csv(
        out_csv,
        out_rows,
        [
            "path_to_study",
            "study_uid",
            "image_uid",
            "anatomical_region",
            "quality_class",
            "violation_type",
            "processing_status",
            "time_of_processing",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Zip archive or folder with DICOM files.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Final CSV in the minimal TZ format.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Directory for intermediate/debug CSV files.")
    parser.add_argument("--device", default=None, help="Torch device for neural models, for example cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    skeleton_csv = work_dir / "inference_skeleton.csv"
    heatmap_csv = work_dir / "inference_landmark_predictions_heatmap.csv"
    knn_csv = work_dir / "inference_landmark_predictions_knn.csv"
    hybrid_csv = work_dir / "inference_landmark_predictions_hybrid.csv"
    spine_csv = work_dir / "inference_spine_classifier_predictions.csv"
    debug_quality_csv = work_dir / "quality_predictions_debug.csv"

    py = sys.executable
    device_args = [] if args.device is None else ["--device", args.device]

    run_step([py, "scripts/infer_pipeline.py", "--input", str(args.input), "--out-csv", str(skeleton_csv)])
    run_step(
        [
            py,
            "scripts/predict_landmarks_heatmap.py",
            "--input-csv",
            str(skeleton_csv),
            "--dicom-input",
            str(args.input),
            "--out-csv",
            str(heatmap_csv),
            *device_args,
        ]
    )
    run_step(
        [
            py,
            "scripts/predict_landmarks_knn.py",
            "--input-csv",
            str(skeleton_csv),
            "--dicom-input",
            str(args.input),
            "--out-csv",
            str(knn_csv),
        ]
    )
    run_step(
        [
            py,
            "scripts/merge_landmark_predictions.py",
            "--primary-csv",
            str(heatmap_csv),
            "--fallback-csv",
            str(knn_csv),
            "--out-csv",
            str(hybrid_csv),
        ]
    )
    run_step(
        [
            py,
            "scripts/predict_spine_classifiers.py",
            "--input-csv",
            str(skeleton_csv),
            "--dicom-input",
            str(args.input),
            "--out-csv",
            str(spine_csv),
            *device_args,
        ]
    )
    run_step(
        [
            py,
            "scripts/apply_quality_rules.py",
            "--inference-csv",
            str(skeleton_csv),
            "--landmarks-csv",
            str(hybrid_csv),
            "--spine-classifiers-csv",
            str(spine_csv),
            "--out-csv",
            str(debug_quality_csv),
            "--model-name",
            "hybrid",
        ]
    )

    elapsed = time.perf_counter() - started
    export_submission(debug_quality_csv, args.out_csv, elapsed, [heatmap_csv, knn_csv, spine_csv])
    print(f"Wrote final CSV: {args.out_csv}")
    print(f"Wrote debug CSV: {debug_quality_csv}")
    print(f"Elapsed seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
