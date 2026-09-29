from __future__ import annotations

import argparse
import csv
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from prepare_artifacts import (
    canonical_rank,
    image_from_dicom,
    infer_region,
    is_dicom_member,
    member_series_folder,
    member_study_folder,
    read_dicom_lite,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "examples" / "example_input.zip"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "inference_skeleton.csv"
DEFAULT_LATERALITY_MODEL = ROOT / "artifacts" / "laterality_template_model.npz"

OUTPUT_COLUMNS = [
    "input_source",
    "path_to_study",
    "study_key",
    "dicom_study_uid",
    "series_uid",
    "image_uid",
    "member",
    "filename",
    "rows",
    "columns",
    "region_hint",
    "anatomical_region",
    "laterality_source",
    "candidate_rank_in_region",
    "is_canonical_for_region",
    "is_duplicate_or_derived",
    "canonical_member",
    "quality_class",
    "violation_type",
    "processing_status",
    "debug_message",
]


class LateralityModel:
    def __init__(self, image_size: int, left_template: np.ndarray, right_template: np.ndarray) -> None:
        self.image_size = image_size
        self.left_template = left_template
        self.right_template = right_template


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_member_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def is_dicom_path(path: Path, raw: bytes | None = None) -> bool:
    return path.is_file() and is_dicom_member(path.name, raw)


def load_laterality_model(path: Path | None) -> LateralityModel | None:
    if path is None or not path.exists():
        return None
    data = np.load(path)
    return LateralityModel(
        image_size=int(data["image_size"][0]),
        left_template=data["left_template"].astype(np.float32),
        right_template=data["right_template"].astype(np.float32),
    )


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


def template_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))


def infer_side_from_image(raw: bytes, model: LateralityModel | None) -> tuple[str, str, str]:
    if model is None:
        return "", "", ""
    try:
        image, _meta = image_from_dicom(raw)
    except Exception as exc:
        return "", "", f"laterality image model failed: {exc}"
    vector = image_vector(image, model.image_size)
    left_dist = template_distance(vector, model.left_template)
    right_dist = template_distance(vector, model.right_template)
    side = "left_hip" if left_dist <= right_dist else "right_hip"
    debug = f"laterality_template left_dist={left_dist:.4f} right_dist={right_dist:.4f}"
    return side, "template_model", debug


def infer_side_from_text(text: str) -> tuple[str, str]:
    name = text.upper()
    if "ППОБ" in name or "RHIP" in name or "RIGHT" in name or "_R" in name:
        return "right_hip", "filename"
    if "ЛПОБ" in name or "LHIP" in name or "LEFT" in name or "_L" in name:
        return "left_hip", "filename"
    return "", ""


def infer_side_from_dicom(meta: dict[str, Any]) -> tuple[str, str]:
    value = str(meta.get("ImageLaterality") or meta.get("Laterality") or "").strip().upper()
    if value == "R":
        return "right_hip", "dicom_laterality"
    if value == "L":
        return "left_hip", "dicom_laterality"
    return "", ""


def study_key_for_member(member: str, fallback: str) -> str:
    study = member_study_folder(member)
    if study:
        return study
    parts = member.split("/")
    if len(parts) > 1:
        return parts[0]
    return fallback


def record_from_raw(
    raw: bytes,
    member: str,
    input_source: str,
    fallback_study: str,
    laterality_model: LateralityModel | None,
) -> dict[str, Any]:
    ds = read_dicom_lite(raw)
    meta = ds.meta
    rows = meta.get("Rows")
    columns = meta.get("Columns")
    filename = Path(member).name
    region_hint = infer_region(columns, rows, filename)
    side, side_source = infer_side_from_text(filename)
    if not side:
        side, side_source = infer_side_from_dicom(meta)
    image_side_debug = ""
    if not side and region_hint == "hip_unknown_side":
        side, side_source, image_side_debug = infer_side_from_image(raw, laterality_model)

    anatomical_region = region_hint
    if region_hint in {"right_hip", "left_hip"}:
        anatomical_region = region_hint
        side_source = side_source or "region_hint"
    elif region_hint == "hip_unknown_side" and side:
        anatomical_region = side
    elif region_hint == "hip_unknown_side":
        anatomical_region = "hip_unknown_side"
        side_source = "unknown"
    elif region_hint == "spine":
        anatomical_region = "spine"
        side_source = ""
    else:
        anatomical_region = "unknown"
        side_source = ""

    return {
        "input_source": input_source,
        "path_to_study": study_key_for_member(member, fallback_study),
        "study_key": study_key_for_member(member, fallback_study),
        "dicom_study_uid": meta.get("StudyInstanceUID", ""),
        "series_uid": meta.get("SeriesInstanceUID", ""),
        "image_uid": meta.get("SOPInstanceUID", ""),
        "member": member,
        "series_folder": member_series_folder(member),
        "filename": filename,
        "file_size": len(raw),
        "rows": rows or "",
        "columns": columns or "",
        "region_hint": region_hint,
        "anatomical_region": anatomical_region,
        "laterality_source": side_source,
        "quality_class": "",
        "violation_type": "",
        "processing_status": "",
        "debug_message": image_side_debug,
    }


def iter_input_records(input_path: Path, laterality_model: LateralityModel | None) -> list[dict[str, Any]]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        records: list[dict[str, Any]] = []
        with zipfile.ZipFile(input_path) as zf:
            for info in sorted(zf.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                raw = zf.read(info)
                if not is_dicom_member(info.filename, raw):
                    continue
                records.append(
                    record_from_raw(
                        raw,
                        info.filename,
                        input_path.name,
                        input_path.stem,
                        laterality_model,
                    )
                )
        return records

    if input_path.is_dir():
        records = []
        for path in sorted(input_path.rglob("*")):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if not is_dicom_path(path, raw):
                continue
            member = clean_member_name(path, input_path)
            records.append(
                record_from_raw(
                    raw,
                    member,
                    input_path.as_posix(),
                    input_path.name,
                    laterality_model,
                )
            )
        return records

    raise ValueError(f"Unsupported input: {input_path}")


def assign_unknown_hip_sides(rows: list[dict[str, Any]]) -> None:
    by_study: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["anatomical_region"] == "hip_unknown_side":
            by_study[row["study_key"]].append(row)

    for study_rows in by_study.values():
        unknown_canonical_candidates = sorted(
            study_rows,
            key=lambda row: canonical_rank(
                int(row["file_size"]),
                int(row["rows"]) if row["rows"] != "" else None,
                int(row["columns"]) if row["columns"] != "" else None,
            ),
            reverse=True,
        )
        for row in unknown_canonical_candidates:
            if not row["debug_message"]:
                row["debug_message"] = "hip side is unknown: no filename, DICOM laterality, or image model signal"


def assign_canonical(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["study_key"], row["anatomical_region"])].append(row)

    canonical_by_group: dict[tuple[str, str], str] = {}
    for key, group_rows in grouped.items():
        ranked = sorted(
            group_rows,
            key=lambda row: canonical_rank(
                int(row["file_size"]),
                int(row["rows"]) if row["rows"] != "" else None,
                int(row["columns"]) if row["columns"] != "" else None,
            ),
            reverse=True,
        )
        canonical = ranked[0]
        canonical_by_group[key] = canonical["member"]
        for rank, row in enumerate(ranked, start=1):
            row["candidate_rank_in_region"] = rank
            row["is_canonical_for_region"] = int(rank == 1 and row["anatomical_region"] not in {"unknown", "hip_unknown_side"})
            row["is_duplicate_or_derived"] = int(rank != 1 and row["anatomical_region"] not in {"unknown", "hip_unknown_side"})

    for row in rows:
        row["canonical_member"] = canonical_by_group.get((row["study_key"], row["anatomical_region"]), "")


def assign_quality_placeholders(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["anatomical_region"] == "unknown":
            row["quality_class"] = ""
            row["violation_type"] = "unknown_anatomical_region"
            row["processing_status"] = "needs_region_review"
            continue
        if row["anatomical_region"] == "hip_unknown_side":
            row["quality_class"] = ""
            row["violation_type"] = "unknown_hip_laterality"
            row["processing_status"] = "needs_laterality_model"
            continue
        row["quality_class"] = ""
        row["violation_type"] = ""
        row["processing_status"] = "not_evaluated_quality_model_missing"


def build_inference_table(input_path: Path, laterality_model_path: Path | None) -> list[dict[str, Any]]:
    laterality_model = load_laterality_model(laterality_model_path)
    rows = iter_input_records(input_path, laterality_model=laterality_model)
    assign_unknown_hip_sides(rows)
    assign_canonical(rows)
    assign_quality_placeholders(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--laterality-model", type=Path, default=DEFAULT_LATERALITY_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_inference_table(args.input, laterality_model_path=args.laterality_model)
    write_csv(args.out_csv, rows, OUTPUT_COLUMNS)
    status_counts: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["processing_status"]] = status_counts.get(row["processing_status"], 0) + 1
        region_counts[row["anatomical_region"]] = region_counts.get(row["anatomical_region"], 0) + 1
    print(f"Rows: {len(rows)}")
    print("Regions:")
    for key, value in sorted(region_counts.items()):
        print(f"  {key}: {value}")
    print("Statuses:")
    for key, value in sorted(status_counts.items()):
        print(f"  {key}: {value}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
