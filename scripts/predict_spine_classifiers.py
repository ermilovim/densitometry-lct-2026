from __future__ import annotations

import argparse
import csv
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from prepare_artifacts import image_from_dicom
from train_spine_classifier import SmallSpineCNN, normalize_image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT / "artifacts" / "inference_skeleton.csv"
DEFAULT_DICOM_INPUT = ROOT / "examples" / "example_input.zip"
DEFAULT_POSITIONING_MODEL = ROOT / "artifacts" / "spine_classifiers" / "positioning" / "best.pt"
DEFAULT_ARTIFACT_MODEL = ROOT / "artifacts" / "spine_classifiers" / "artifact" / "best.pt"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "inference_spine_classifier_predictions.csv"
VALIDATED_SPINE_POSITIONING_THRESHOLD = 0.7809473276138306
VALIDATED_SPINE_ARTIFACT_THRESHOLD = 0.38129135966300964
VALIDATED_THRESHOLDS = {
    "positioning": VALIDATED_SPINE_POSITIONING_THRESHOLD,
    "artifact": VALIDATED_SPINE_ARTIFACT_THRESHOLD,
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


def read_member_raw(dicom_input: Path, member: str) -> bytes:
    if dicom_input.is_file() and dicom_input.suffix.lower() == ".zip":
        with zipfile.ZipFile(dicom_input) as zf:
            return zf.read(member)
    return (dicom_input / member).read_bytes()


def image_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    resized = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = normalize_image(np.asarray(resized, dtype=np.float32))
    return torch.from_numpy(arr[None, None, :, :].astype(np.float32)).to(device)


def load_classifier(path: Path, device: torch.device, threshold_override: float | None = None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    image_size = int(checkpoint["image_size"])
    base_channels = int(checkpoint.get("base_channels", 24))
    model = SmallSpineCNN(base=base_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    report = checkpoint.get("report", {})
    task = str(checkpoint.get("task", ""))
    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(VALIDATED_THRESHOLDS.get(task, report.get("threshold", 0.5)))
    )
    return {
        "path": path,
        "model": model,
        "image_size": image_size,
        "threshold": threshold,
    }


@torch.no_grad()
def predict_bad_probability(image: Image.Image, bundle: dict[str, Any], device: torch.device) -> float:
    tensor = image_tensor(image, int(bundle["image_size"]), device)
    logits = bundle["model"](tensor)
    return float(torch.sigmoid(logits)[0].detach().cpu().item())


def empty_output(row: dict[str, str]) -> dict[str, Any]:
    return {
        "input_source": row.get("input_source", ""),
        "study_key": row.get("study_key", ""),
        "member": row.get("member", ""),
        "filename": row.get("filename", ""),
        "anatomical_region": row.get("anatomical_region", ""),
        "is_canonical_for_region": row.get("is_canonical_for_region", ""),
        "spine_classifier_status": "",
        "spine_positioning_prob_bad": "",
        "spine_positioning_threshold": "",
        "spine_positioning_pred_bad": "",
        "spine_artifact_prob_bad": "",
        "spine_artifact_threshold": "",
        "spine_artifact_pred_bad": "",
    }


def predict_row(
    row: dict[str, str],
    dicom_input: Path,
    positioning_bundle: dict[str, Any] | None,
    artifact_bundle: dict[str, Any] | None,
    device: torch.device,
) -> dict[str, Any]:
    out = empty_output(row)
    if row.get("anatomical_region") != "spine":
        out["spine_classifier_status"] = "skipped_non_spine"
        return out
    if positioning_bundle is None or artifact_bundle is None:
        out["spine_classifier_status"] = "model_missing"
        return out

    try:
        image, _meta = image_from_dicom(read_member_raw(dicom_input, row["member"]))
    except Exception as exc:
        out["spine_classifier_status"] = f"render_failed: {exc}"
        return out

    positioning_prob = predict_bad_probability(image, positioning_bundle, device)
    artifact_prob = predict_bad_probability(image, artifact_bundle, device)
    positioning_threshold = float(positioning_bundle["threshold"])
    artifact_threshold = float(artifact_bundle["threshold"])

    out["spine_classifier_status"] = "predicted"
    out["spine_positioning_prob_bad"] = f"{positioning_prob:.6f}"
    out["spine_positioning_threshold"] = f"{positioning_threshold:.6f}"
    out["spine_positioning_pred_bad"] = int(positioning_prob + 1e-6 >= positioning_threshold)
    out["spine_artifact_prob_bad"] = f"{artifact_prob:.6f}"
    out["spine_artifact_threshold"] = f"{artifact_threshold:.6f}"
    out["spine_artifact_pred_bad"] = int(artifact_prob + 1e-6 >= artifact_threshold)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--dicom-input", type=Path, default=DEFAULT_DICOM_INPUT)
    parser.add_argument("--positioning-model", type=Path, default=DEFAULT_POSITIONING_MODEL)
    parser.add_argument("--artifact-model", type=Path, default=DEFAULT_ARTIFACT_MODEL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--positioning-threshold", type=float, default=VALIDATED_SPINE_POSITIONING_THRESHOLD)
    parser.add_argument("--artifact-threshold", type=float, default=VALIDATED_SPINE_ARTIFACT_THRESHOLD)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    positioning_bundle = load_classifier(args.positioning_model, device, threshold_override=args.positioning_threshold)
    artifact_bundle = load_classifier(args.artifact_model, device, threshold_override=args.artifact_threshold)
    rows = []
    for row in read_csv(args.input_csv):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = predict_row(row, args.dicom_input, positioning_bundle, artifact_bundle, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["processing_seconds"] = f"{time.perf_counter() - started:.6f}"
        rows.append(result)
    fieldnames = [
        "input_source",
        "study_key",
        "member",
        "filename",
        "anatomical_region",
        "is_canonical_for_region",
        "spine_classifier_status",
        "processing_seconds",
        "spine_positioning_prob_bad",
        "spine_positioning_threshold",
        "spine_positioning_pred_bad",
        "spine_artifact_prob_bad",
        "spine_artifact_threshold",
        "spine_artifact_pred_bad",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["spine_classifier_status"]] = status_counts.get(row["spine_classifier_status"], 0) + 1
    print(f"Rows: {len(rows)}")
    for key, value in sorted(status_counts.items()):
        print(f"{key}: {value}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
