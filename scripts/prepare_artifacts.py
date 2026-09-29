from __future__ import annotations

import argparse
import csv
import io
import math
import re
import struct
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ZIP = ROOT / "data" / "train.zip"
DEFAULT_TEST_ZIP = ROOT / "data" / "test.zip"
DEFAULT_OUT_DIR = ROOT / "artifacts"

PIXEL_SPACING_Y_MM = 1.05
PIXEL_SPACING_X_MM = 0.60
HIP_ROI_VERTICAL_MARGIN_PX = 30 / PIXEL_SPACING_Y_MM
HIP_ROI_HORIZONTAL_MARGIN_PX = 20 / PIXEL_SPACING_X_MM

EXPLICIT_LONG_VR = {"OB", "OD", "OF", "OL", "OV", "OW", "SQ", "SV", "UC", "UR", "UT", "UN", "UV"}
TEXT_VR = {
    "AE",
    "AS",
    "CS",
    "DA",
    "DS",
    "DT",
    "IS",
    "LO",
    "LT",
    "PN",
    "SH",
    "ST",
    "TM",
    "UC",
    "UI",
    "UR",
    "UT",
}

TAG_NAMES = {
    (0x0002, 0x0010): "TransferSyntaxUID",
    (0x0008, 0x0016): "SOPClassUID",
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x0070): "Manufacturer",
    (0x0008, 0x1030): "StudyDescription",
    (0x0008, 0x103E): "SeriesDescription",
    (0x0018, 0x1020): "SoftwareVersions",
    (0x0018, 0x1030): "ProtocolName",
    (0x0018, 0x1164): "ImagerPixelSpacing",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0020, 0x0011): "SeriesNumber",
    (0x0020, 0x0013): "InstanceNumber",
    (0x0020, 0x0060): "Laterality",
    (0x0020, 0x0062): "ImageLaterality",
    (0x0028, 0x0002): "SamplesPerPixel",
    (0x0028, 0x0004): "PhotometricInterpretation",
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
    (0x0028, 0x0030): "PixelSpacing",
    (0x0028, 0x0034): "PixelAspectRatio",
    (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0101): "BitsStored",
    (0x0028, 0x0102): "HighBit",
    (0x0028, 0x0103): "PixelRepresentation",
    (0x0028, 0x1050): "WindowCenter",
    (0x0028, 0x1051): "WindowWidth",
    (0x0028, 0x1052): "RescaleIntercept",
    (0x0028, 0x1053): "RescaleSlope",
}


@dataclass
class DicomLite:
    meta: dict[str, Any]
    pixel_data: bytes | None
    private_tag_count: int


def is_dicom_member(name: str, raw: bytes | None = None) -> bool:
    """Recognize DICOM files even when a PACS export omits ``.dcm``."""
    if name.endswith("/"):
        return False
    if name.lower().endswith(".dcm"):
        return True
    if raw is None:
        return False
    try:
        meta = read_dicom_lite(raw).meta
        return bool(meta.get("Rows") and meta.get("Columns"))
    except (ValueError, IndexError, struct.error):
        return False


def clean_text(value: bytes) -> str:
    raw = value.rstrip(b" \x00")
    if not raw:
        return ""
    for encoding in ("utf-8", "cp1251", "latin1"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace").strip()


def parse_numeric(value: bytes, fmt: str) -> int | None:
    if len(value) < struct.calcsize(fmt):
        return None
    return struct.unpack(fmt, value[: struct.calcsize(fmt)])[0]


def parse_value(tag: tuple[int, int], vr: str | None, value: bytes) -> Any:
    if vr in TEXT_VR or tag in {
        (0x0002, 0x0010),
        (0x0008, 0x0016),
        (0x0008, 0x0018),
        (0x0020, 0x000D),
        (0x0020, 0x000E),
    }:
        return clean_text(value)
    if vr == "US" or tag in {
        (0x0028, 0x0002),
        (0x0028, 0x0010),
        (0x0028, 0x0011),
        (0x0028, 0x0100),
        (0x0028, 0x0101),
        (0x0028, 0x0102),
        (0x0028, 0x0103),
        (0x0020, 0x0011),
        (0x0020, 0x0013),
    }:
        return parse_numeric(value, "<H")
    if vr == "SS":
        return parse_numeric(value, "<h")
    if vr == "UL":
        return parse_numeric(value, "<I")
    if vr == "SL":
        return parse_numeric(value, "<i")
    return clean_text(value) if len(value) <= 128 else f"<{len(value)} bytes>"


def find_sequence_end(data: bytes, pos: int) -> int:
    marker = b"\xfe\xff\xdd\xe0"
    idx = data.find(marker, pos)
    if idx == -1:
        return len(data)
    return min(len(data), idx + 8)


def read_element(
    data: bytes,
    pos: int,
    explicit_vr: bool,
) -> tuple[tuple[int, int], str | None, int, int, bytes] | None:
    if pos + 8 > len(data):
        return None
    group, element = struct.unpack_from("<HH", data, pos)
    pos += 4
    vr: str | None = None
    if explicit_vr:
        vr = data[pos : pos + 2].decode("ascii", errors="replace")
        pos += 2
        if vr in EXPLICIT_LONG_VR:
            pos += 2
            if pos + 4 > len(data):
                return None
            length = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        else:
            if pos + 2 > len(data):
                return None
            length = struct.unpack_from("<H", data, pos)[0]
            pos += 2
    else:
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4

    value_start = pos
    if length == 0xFFFFFFFF:
        value_end = find_sequence_end(data, value_start)
        value = data[value_start:value_end]
        next_pos = value_end
    else:
        value_end = min(len(data), value_start + length)
        value = data[value_start:value_end]
        next_pos = value_end
    return (group, element), vr, length, next_pos, value


def read_dicom_lite(raw: bytes, stop_after_pixel: bool = True) -> DicomLite:
    pos = 132 if len(raw) >= 132 and raw[128:132] == b"DICM" else 0
    meta: dict[str, Any] = {}
    pixel_data: bytes | None = None
    private_tag_count = 0

    # File meta header is always explicit little endian.
    while pos + 8 <= len(raw):
        item = read_element(raw, pos, explicit_vr=True)
        if item is None:
            break
        tag, vr, _length, next_pos, value = item
        if tag[0] != 0x0002:
            break
        name = TAG_NAMES.get(tag)
        if name:
            meta[name] = parse_value(tag, vr, value)
        pos = next_pos

    transfer_syntax = str(meta.get("TransferSyntaxUID") or "")
    explicit_vr = transfer_syntax not in {"1.2.840.10008.1.2"}

    while pos + 8 <= len(raw):
        item = read_element(raw, pos, explicit_vr=explicit_vr)
        if item is None:
            break
        tag, vr, _length, next_pos, value = item
        if tag[0] % 2 == 1:
            private_tag_count += 1
        if tag == (0x7FE0, 0x0010):
            pixel_data = value
            if stop_after_pixel:
                break
        name = TAG_NAMES.get(tag)
        if name:
            meta[name] = parse_value(tag, vr, value)
        pos = next_pos

    return DicomLite(meta=meta, pixel_data=pixel_data, private_tag_count=private_tag_count)


def member_study_folder(member: str) -> str:
    parts = member.split("/")
    if len(parts) >= 2 and parts[0] == "Исследования":
        return parts[1]
    if len(parts) >= 2:
        return parts[0]
    return ""


def member_series_folder(member: str) -> str:
    parts = member.split("/")
    for part in parts:
        if part.startswith("series_"):
            return part
    return ""


def infer_region(columns: int | None, rows: int | None, filename: str) -> str:
    name = filename.upper()
    if "ПОП" in name or "SPINE" in name:
        return "spine"
    if "ППОБ" in name or "RHIP" in name or "RIGHT" in name:
        return "right_hip"
    if "ЛПОБ" in name or "LHIP" in name or "LEFT" in name:
        return "left_hip"
    if columns is None:
        return "unknown"
    if columns >= 295:
        return "spine"
    if 240 <= columns <= 290:
        return "hip_unknown_side"
    return "unknown"


def canonical_rank(size: int, rows: int | None, columns: int | None) -> tuple[int, int, int]:
    area = (rows or 0) * (columns or 0)
    return (area, size, -(rows or 0))


def iter_dicom_records(zip_path: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if not is_dicom_member(info.filename):
                continue
            raw = zf.read(info)
            ds = read_dicom_lite(raw)
            meta = ds.meta
            rows = meta.get("Rows")
            columns = meta.get("Columns")
            filename = Path(info.filename).name
            record = {
                "split": split,
                "archive": zip_path.name,
                "member": info.filename,
                "path_to_study": member_study_folder(info.filename),
                "study_folder": member_study_folder(info.filename),
                "series_folder": member_series_folder(info.filename),
                "filename": filename,
                "file_size": info.file_size,
                "study_uid": meta.get("StudyInstanceUID", ""),
                "series_uid": meta.get("SeriesInstanceUID", ""),
                "image_uid": meta.get("SOPInstanceUID", ""),
                "modality": meta.get("Modality", ""),
                "manufacturer": meta.get("Manufacturer", ""),
                "software_versions": meta.get("SoftwareVersions", ""),
                "study_description": meta.get("StudyDescription", ""),
                "series_description": meta.get("SeriesDescription", ""),
                "protocol_name": meta.get("ProtocolName", ""),
                "laterality": meta.get("Laterality", ""),
                "image_laterality": meta.get("ImageLaterality", ""),
                "rows": rows or "",
                "columns": columns or "",
                "photometric": meta.get("PhotometricInterpretation", ""),
                "bits_allocated": meta.get("BitsAllocated", ""),
                "bits_stored": meta.get("BitsStored", ""),
                "pixel_spacing": meta.get("PixelSpacing", ""),
                "imager_pixel_spacing": meta.get("ImagerPixelSpacing", ""),
                "pixel_aspect_ratio": meta.get("PixelAspectRatio", ""),
                "private_tag_count": ds.private_tag_count,
                "region_hint": infer_region(columns, rows, filename),
            }
            records.append(record)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["study_folder"], record["region_hint"])].append(record)
    for rows in grouped.values():
        ranked = sorted(
            rows,
            key=lambda r: canonical_rank(
                int(r["file_size"]),
                int(r["rows"]) if r["rows"] != "" else None,
                int(r["columns"]) if r["columns"] != "" else None,
            ),
            reverse=True,
        )
        for i, record in enumerate(ranked, start=1):
            record["canonical_rank_in_region"] = i
            record["is_largest_in_region"] = int(i == 1)
    return records


def column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value


def row_index(cell_ref: str) -> int:
    digits = re.sub(r"[^0-9]", "", cell_ref)
    return int(digits)


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall(".//m:si", ns):
        strings.append("".join(t.text or "" for t in item.findall(".//m:t", ns)))
    return strings


def read_xlsx_sheet_rows(xlsx_bytes: bytes) -> list[dict[int, str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as zf:
        shared = read_shared_strings(zf)
        sheet_name = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(zf.read(sheet_name))
        rows: list[dict[int, str]] = []
        for row in root.findall(".//m:row", ns):
            cells: dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "")
                if not ref:
                    continue
                idx = column_index(ref)
                value = ""
                if cell.attrib.get("t") == "s":
                    node = cell.find("m:v", ns)
                    if node is not None and node.text not in (None, ""):
                        value = shared[int(node.text)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                else:
                    node = cell.find("m:v", ns)
                    value = "" if node is None or node.text is None else node.text
                cells[idx] = value.strip()
            if cells:
                rows.append(cells)
    return rows


def yes_no(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def or_nullable(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return int(any(present))


def extract_markup_xlsx(train_zip: Path) -> bytes:
    with zipfile.ZipFile(train_zip) as zf:
        matches = [name for name in zf.namelist() if name.lower().endswith(".xlsx")]
        if not matches:
            raise FileNotFoundError(f"No .xlsx markup found in {train_zip}")
        return zf.read(matches[0])


def build_labels(train_zip: Path) -> list[dict[str, Any]]:
    rows = read_xlsx_sheet_rows(extract_markup_xlsx(train_zip))
    records: list[dict[str, Any]] = []
    for cells in rows:
        row_no = yes_no(cells.get(1, ""))
        study = cells.get(2, "")
        if row_no is None or not study or study == "study":
            continue
        spine_positioning = yes_no(cells.get(3, ""))
        spine_axis = yes_no(cells.get(4, ""))
        spine_artifact = yes_no(cells.get(5, ""))
        right_hip_positioning_rotation = yes_no(cells.get(6, ""))
        right_hip_roi = yes_no(cells.get(7, ""))
        left_hip_positioning_rotation = yes_no(cells.get(8, ""))
        left_hip_roi = yes_no(cells.get(9, ""))
        raw_spine_quality = yes_no(cells.get(10, ""))
        raw_right_hip_quality = yes_no(cells.get(11, ""))
        raw_left_hip_quality = yes_no(cells.get(12, ""))
        spine_quality = or_nullable(spine_positioning, spine_axis, spine_artifact)
        right_hip_quality = or_nullable(right_hip_positioning_rotation, right_hip_roi)
        left_hip_quality = or_nullable(left_hip_positioning_rotation, left_hip_roi)
        records.append(
            {
                "row_no": row_no,
                "study_uid": study,
                "spine_positioning": spine_positioning,
                "spine_axis": spine_axis,
                "spine_artifact": spine_artifact,
                "right_hip_positioning_rotation": right_hip_positioning_rotation,
                "right_hip_roi": right_hip_roi,
                "left_hip_positioning_rotation": left_hip_positioning_rotation,
                "left_hip_roi": left_hip_roi,
                "spine_quality_or": spine_quality,
                "right_hip_quality_or": right_hip_quality,
                "left_hip_quality_or": left_hip_quality,
                "raw_spine_quality": raw_spine_quality,
                "raw_right_hip_quality": raw_right_hip_quality,
                "raw_left_hip_quality": raw_left_hip_quality,
                "spine_quality_differs_from_raw": int(
                    spine_quality is not None
                    and raw_spine_quality is not None
                    and spine_quality != raw_spine_quality
                ),
                "right_hip_quality_differs_from_raw": int(
                    right_hip_quality is not None
                    and raw_right_hip_quality is not None
                    and right_hip_quality != raw_right_hip_quality
                ),
                "left_hip_quality_differs_from_raw": int(
                    left_hip_quality is not None
                    and raw_left_hip_quality is not None
                    and left_hip_quality != raw_left_hip_quality
                ),
                "comment": cells.get(13, ""),
            }
        )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_u8(pixels: bytes) -> bytes:
    if not pixels:
        return pixels
    values = list(pixels)
    hist = [0] * 256
    for value in values:
        hist[value] += 1
    total = len(values)
    lo_target = max(0, int(total * 0.01))
    hi_target = max(0, int(total * 0.99))
    acc = 0
    lo = 0
    for i, count in enumerate(hist):
        acc += count
        if acc >= lo_target:
            lo = i
            break
    acc = 0
    hi = 255
    for i, count in enumerate(hist):
        acc += count
        if acc >= hi_target:
            hi = i
            break
    if hi <= lo:
        lo, hi = min(values), max(values)
    if hi <= lo:
        return bytes([0] * len(values))
    scale = 255 / (hi - lo)
    return bytes(max(0, min(255, round((v - lo) * scale))) for v in values)


def image_from_dicom(raw: bytes) -> tuple[Image.Image, dict[str, Any]]:
    ds = read_dicom_lite(raw)
    rows = int(ds.meta.get("Rows") or 0)
    columns = int(ds.meta.get("Columns") or 0)
    bits = int(ds.meta.get("BitsAllocated") or 8)
    if rows <= 0 or columns <= 0 or ds.pixel_data is None:
        raise ValueError("DICOM has no readable pixel data")
    if bits != 8:
        raise ValueError(f"Unsupported BitsAllocated={bits}; use pydicom for this file")
    expected = rows * columns
    pixels = ds.pixel_data[:expected]
    if len(pixels) < expected:
        raise ValueError(f"PixelData is too short: {len(pixels)} < {expected}")
    image = Image.frombytes("L", (columns, rows), normalize_u8(pixels))
    return image, ds.meta


def safe_study_name(study: str, index: int) -> str:
    suffix = re.sub(r"[^0-9A-Za-z_.-]+", "_", study)[:90]
    return f"{index:03d}_{suffix or 'study'}.jpg"


def make_contact_sheets(
    train_zip: Path,
    index_rows: list[dict[str, Any]],
    out_dir: Path,
    max_sheets: int | None,
    thumb_width: int = 180,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [r for r in index_rows if r["split"] == "train"]
    by_study: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_study[row["study_folder"]].append(row)
    studies = sorted(by_study)
    if max_sheets is not None:
        studies = studies[:max_sheets]

    member_to_raw: dict[str, bytes] = {}
    with zipfile.ZipFile(train_zip) as zf:
        for study_i, study in enumerate(studies, start=1):
            rows = sorted(by_study[study], key=lambda r: r["member"])
            thumbs: list[tuple[Image.Image, str]] = []
            for row in rows:
                raw = member_to_raw.get(row["member"])
                if raw is None:
                    raw = zf.read(row["member"])
                    member_to_raw[row["member"]] = raw
                try:
                    img, _meta = image_from_dicom(raw)
                    scale = thumb_width / img.width
                    thumb = img.resize((thumb_width, max(1, round(img.height * scale))))
                except Exception as exc:
                    thumb = Image.new("L", (thumb_width, thumb_width), 0)
                    label = f"{row['filename']} ERROR: {exc}"
                else:
                    label = (
                        f"{row['filename']}  {row['columns']}x{row['rows']}  "
                        f"{row['region_hint']}  rank={row.get('canonical_rank_in_region', '')}"
                    )
                thumbs.append((thumb.convert("RGB"), label))

            cols = min(4, max(1, len(thumbs)))
            label_h = 38
            gap = 12
            cell_w = thumb_width
            max_thumb_h = max(img.height for img, _ in thumbs) if thumbs else thumb_width
            cell_h = max_thumb_h + label_h
            title_h = 42
            sheet_w = cols * cell_w + (cols + 1) * gap
            sheet_h = title_h + math.ceil(len(thumbs) / cols) * (cell_h + gap) + gap
            sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((gap, 10), f"{study_i:03d}. {study}   n={len(thumbs)}", fill=(0, 0, 0))

            for i, (thumb, label) in enumerate(thumbs):
                r, c = divmod(i, cols)
                x = gap + c * (cell_w + gap)
                y = title_h + r * (cell_h + gap)
                sheet.paste(thumb, (x, y))
                draw.rectangle((x, y, x + thumb.width - 1, y + thumb.height - 1), outline=(160, 160, 160))
                draw.text((x, y + max_thumb_h + 4), label[:64], fill=(0, 0, 0))

            sheet.save(out_dir / safe_study_name(study, study_i), quality=92)
    return len(studies)


def build_study_summary(index_rows: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels_by_uid = {str(row["study_uid"]): row for row in labels}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        grouped[(row["split"], row["study_folder"])].append(row)
    summary = []
    for (split, study), rows in sorted(grouped.items()):
        region_counts = Counter(row["region_hint"] for row in rows)
        label = labels_by_uid.get(study, {})
        summary.append(
            {
                "split": split,
                "study_uid": study,
                "n_dicoms": len(rows),
                "n_spine_hint": region_counts.get("spine", 0),
                "n_hip_unknown_side_hint": region_counts.get("hip_unknown_side", 0),
                "n_right_hip_hint": region_counts.get("right_hip", 0),
                "n_left_hip_hint": region_counts.get("left_hip", 0),
                "n_unknown_hint": region_counts.get("unknown", 0),
                "spine_quality_or": label.get("spine_quality_or", ""),
                "right_hip_quality_or": label.get("right_hip_quality_or", ""),
                "left_hip_quality_or": label.get("left_hip_quality_or", ""),
                "comment": label.get("comment", ""),
            }
        )
    return summary


def build_manual_review(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_rows = []
    for row in index_rows:
        review_rows.append(
            {
                "split": row["split"],
                "study_folder": row["study_folder"],
                "member": row["member"],
                "filename": row["filename"],
                "columns": row["columns"],
                "rows": row["rows"],
                "region_hint": row["region_hint"],
                "candidate_is_canonical": row.get("is_largest_in_region", ""),
                "candidate_rank_in_region": row.get("canonical_rank_in_region", ""),
                "manual_region": "",
                "manual_is_canonical": "",
                "manual_notes": "",
            }
        )
    return review_rows


def print_summary(index_rows: list[dict[str, Any]], labels: list[dict[str, Any]], sheets: int) -> None:
    print(f"DICOM rows: {len(index_rows)}")
    print("By split:", dict(Counter(row["split"] for row in index_rows)))
    print("By region_hint:", dict(Counter(row["region_hint"] for row in index_rows)))
    print("By size:", dict(Counter((row["columns"], row["rows"]) for row in index_rows).most_common()))
    print(f"Label rows: {len(labels)}")
    print(
        "Raw spine final differs from OR:",
        sum(row["spine_quality_differs_from_raw"] for row in labels),
    )
    print(
        "Raw right hip final differs from OR:",
        sum(row["right_hip_quality_differs_from_raw"] for row in labels),
    )
    print(
        "Raw left hip final differs from OR:",
        sum(row["left_hip_quality_differs_from_raw"] for row in labels),
    )
    print(
        "Hip ROI thresholds:",
        f"vertical={HIP_ROI_VERTICAL_MARGIN_PX:.1f}px",
        f"horizontal={HIP_ROI_HORIZONTAL_MARGIN_PX:.1f}px",
    )
    print(f"Contact sheets written: {sheets}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    parser.add_argument("--test-zip", type=Path, default=DEFAULT_TEST_ZIP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-contact-sheets", type=int, default=None)
    parser.add_argument("--no-contact-sheets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = iter_dicom_records(args.train_zip, "train")
    if args.test_zip.exists():
        index_rows.extend(iter_dicom_records(args.test_zip, "test"))

    labels = build_labels(args.train_zip)
    study_summary = build_study_summary(index_rows, labels)
    manual_review = build_manual_review(index_rows)

    write_csv(out_dir / "dicom_index.csv", index_rows)
    write_csv(out_dir / "labels.csv", labels)
    write_csv(out_dir / "study_summary.csv", study_summary)
    write_csv(out_dir / "manual_image_review.csv", manual_review)

    sheets = 0
    if not args.no_contact_sheets:
        sheets = make_contact_sheets(
            args.train_zip,
            index_rows,
            out_dir / "contact_sheets",
            max_sheets=args.max_contact_sheets,
        )

    print_summary(index_rows, labels, sheets)
    print(f"Wrote: {out_dir / 'dicom_index.csv'}")
    print(f"Wrote: {out_dir / 'labels.csv'}")
    print(f"Wrote: {out_dir / 'study_summary.csv'}")
    print(f"Wrote: {out_dir / 'manual_image_review.csv'}")


if __name__ == "__main__":
    main()
