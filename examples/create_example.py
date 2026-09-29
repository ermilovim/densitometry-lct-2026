"""Create a small synthetic DICOM archive with no patient data."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage


ROOT = Path(__file__).resolve().parent
UID_ROOT = "1.2.826.0.1.3680043.10.543.20260929"
STUDY_UID = f"{UID_ROOT}.1"


def image(rows: int, columns: int, center_x: float) -> np.ndarray:
    yy, xx = np.mgrid[:rows, :columns]
    background = 35 + 25 * (yy / rows)
    shaft = 150 * np.exp(-(((xx - center_x) / 20) ** 2 + ((yy - rows * 0.55) / (rows * 0.42)) ** 8))
    head = 135 * np.exp(-(((xx - center_x) / 52) ** 2 + ((yy - rows * 0.22) / 48) ** 2))
    return np.clip(background + shaft + head, 0, 255).astype(np.uint8)


def dicom_bytes(name: str, pixels: np.ndarray, instance: int, laterality: str = "") -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = f"{UID_ROOT}.{instance}.2"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(name, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = f"{UID_ROOT}.{instance}.1"
    ds.Modality = "DX"
    ds.PatientName = "SYNTHETIC^DEMO"
    ds.PatientID = "SYNTHETIC-DEMO"
    ds.ImageLaterality = laterality
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def main() -> None:
    cases = [
        ("spine.dcm", image(390, 320, 160), ""),
        ("right_hip.dcm", image(346, 280, 118), "R"),
        ("left_hip.dcm", image(346, 280, 162), "L"),
    ]
    with zipfile.ZipFile(ROOT / "example_input.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for instance, (name, pixels, laterality) in enumerate(cases, start=1):
            archive.writestr(f"synthetic-study/{name}", dicom_bytes(name, pixels, instance, laterality))


if __name__ == "__main__":
    main()
