from __future__ import annotations

import argparse
import csv
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from prepare_artifacts import DEFAULT_TRAIN_ZIP, image_from_dicom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_CSV = ROOT / "artifacts" / "manual_image_review.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "laterality_baseline_eval.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "laterality_baseline_report.csv"
DEFAULT_MODEL_NPZ = ROOT / "artifacts" / "laterality_template_model.npz"

LABELS = ("left_hip", "right_hip")


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


def split_by_study(rows: list[dict[str, str]], val_fraction: float, seed: int) -> dict[str, str]:
    studies = sorted({row["study_folder"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(studies)
    n_val = max(1, round(len(studies) * val_fraction))
    val = set(studies[:n_val])
    return {study: "val" if study in val else "train" for study in studies}


def image_vector(image: Image.Image, size: int) -> np.ndarray:
    arr = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    arr = arr / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    if std > 1e-6:
        arr = (arr - mean) / std
    else:
        arr = arr - mean
    return arr.reshape(-1)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def canonical_hip_rows(review_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in review_rows
        if row.get("manual_is_canonical") == "1"
        and row.get("manual_region") in LABELS
        and row.get("split") == "train"
    ]


def load_vectors(rows: list[dict[str, str]], train_zip: Path, size: int) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(train_zip) as zf:
        for row in rows:
            raw = zf.read(row["member"])
            image, _meta = image_from_dicom(raw)
            vectors[row["member"]] = image_vector(image, size)
    return vectors


def train_templates(rows: list[dict[str, str]], vectors: dict[str, np.ndarray], split_by_uid: dict[str, str]) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for label in LABELS:
        label_vectors = [
            vectors[row["member"]]
            for row in rows
            if split_by_uid[row["study_folder"]] == "train" and row["manual_region"] == label
        ]
        if not label_vectors:
            raise ValueError(f"No train vectors for {label}")
        templates[label] = np.mean(np.stack(label_vectors, axis=0), axis=0)
    return templates


def predict(vector: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float, float, float]:
    left_dist = distance(vector, templates["left_hip"])
    right_dist = distance(vector, templates["right_hip"])
    pred = "left_hip" if left_dist <= right_dist else "right_hip"
    margin = abs(left_dist - right_dist)
    return pred, left_dist, right_dist, margin


def evaluate(
    review_csv: Path,
    train_zip: Path,
    out_csv: Path,
    report_csv: Path,
    model_npz: Path,
    size: int,
    val_fraction: float,
    seed: int,
    train_all: bool,
) -> None:
    rows = canonical_hip_rows(read_csv(review_csv))
    split_by_uid = (
        {row["study_folder"]: "train" for row in rows}
        if train_all
        else split_by_study(rows, val_fraction=val_fraction, seed=seed)
    )
    vectors = load_vectors(rows, train_zip, size=size)
    templates = train_templates(rows, vectors, split_by_uid)

    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        split = split_by_uid[row["study_folder"]]
        pred, left_dist, right_dist, margin = predict(vectors[row["member"]], templates)
        eval_rows.append(
            {
                "split": split,
                "study_folder": row["study_folder"],
                "member": row["member"],
                "filename": row["filename"],
                "rows": row["rows"],
                "columns": row["columns"],
                "label": row["manual_region"],
                "prediction": pred,
                "is_correct": int(pred == row["manual_region"]),
                "left_distance": f"{left_dist:.6f}",
                "right_distance": f"{right_dist:.6f}",
                "margin": f"{margin:.6f}",
                "manual_notes": row.get("manual_notes", ""),
            }
        )

    report: list[dict[str, Any]] = []
    for split in ["train", "val"]:
        part = [row for row in eval_rows if row["split"] == split]
        correct = sum(int(row["is_correct"]) for row in part)
        report.append(
            {
                "split": split,
                "n": len(part),
                "correct": correct,
                "accuracy": f"{correct / len(part):.3f}" if part else "",
            }
        )

    write_csv(out_csv, eval_rows)
    write_csv(report_csv, report, fieldnames=["split", "n", "correct", "accuracy"])
    model_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_npz,
        image_size=np.array([size], dtype=np.int32),
        left_template=templates["left_hip"],
        right_template=templates["right_hip"],
    )
    print(f"Rows: {len(eval_rows)}")
    for row in report:
        print(f"{row['split']}: {row['correct']}/{row['n']} accuracy={row['accuracy']}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {report_csv}")
    print(f"Wrote: {model_npz}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--model-npz", type=Path, default=DEFAULT_MODEL_NPZ)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-all", action="store_true", help="Build left/right hip templates from all canonical hip rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        review_csv=args.review_csv,
        train_zip=args.train_zip,
        out_csv=args.out_csv,
        report_csv=args.report_csv,
        model_npz=args.model_npz,
        size=args.size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        train_all=args.train_all,
    )


if __name__ == "__main__":
    main()
