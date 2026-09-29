from __future__ import annotations

import argparse
import csv
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from prepare_artifacts import DEFAULT_TRAIN_ZIP, image_from_dicom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANDMARK_CSV = ROOT / "artifacts" / "landmark_all.csv"
DEFAULT_ERRORS_CSV = ROOT / "artifacts" / "landmark_all_rule_errors.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "rule_error_sheets"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if len(candidate) <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word[:width]
    if line:
        lines.append(line)
    return lines or [""]


def error_kind(row: dict[str, str]) -> str:
    label = row.get("label", "")
    prediction = row.get("prediction", "")
    if label == "1" and prediction == "0":
        return "FN"
    if label == "0" and prediction == "1":
        return "FP"
    return "ERR"


def sheet_name(task: str, rule: str) -> str:
    name = f"{task}__{rule}".replace("/", "_").replace(" ", "_")
    return f"{name[:150]}.jpg"


def draw_sheet(
    rows: list[dict[str, str]],
    rows_by_pilot_id: dict[str, dict[str, str]],
    train_zip: Path,
    out_path: Path,
    thumb_width: int,
    cols: int,
) -> None:
    cells: list[tuple[Image.Image, list[str]]] = []
    with zipfile.ZipFile(train_zip) as zf:
        for err in rows:
            item = rows_by_pilot_id.get(err["pilot_id"])
            if item is None:
                continue
            try:
                raw = zf.read(item["member"])
                image, _meta = image_from_dicom(raw)
                scale = thumb_width / image.width
                thumb = image.resize((thumb_width, max(1, round(image.height * scale))))
            except Exception as exc:
                thumb = Image.new("L", (thumb_width, thumb_width), 0)
                label = f"render error: {exc}"
            else:
                label = (
                    f"{error_kind(err)} pilot={err['pilot_id']} img={err['image_number']} "
                    f"{err['anatomical_region']} label={err['label']} pred={err['prediction']} "
                    f"value={err.get('value', '')} note={err.get('annotation_notes', '')}"
                )
            cells.append((thumb.convert("RGB"), wrap_text(label, 36)))

    if not cells:
        return

    cols = min(cols, len(cells))
    gap = 12
    title_h = 42
    label_line_h = 13
    max_thumb_h = max(thumb.height for thumb, _label in cells)
    label_h = 6 * label_line_h
    cell_w = thumb_width
    cell_h = max_thumb_h + label_h
    sheet_w = cols * cell_w + (cols + 1) * gap
    sheet_h = title_h + math.ceil(len(cells) / cols) * (cell_h + gap) + gap

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 12), out_path.stem[:120], fill=(0, 0, 0))

    for i, (thumb, label_lines) in enumerate(cells):
        row_i, col_i = divmod(i, cols)
        x = gap + col_i * (cell_w + gap)
        y = title_h + row_i * (cell_h + gap)
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb.width - 1, y + thumb.height - 1), outline=(160, 160, 160))
        for line_i, line in enumerate(label_lines[:6]):
            draw.text((x, y + max_thumb_h + 4 + line_i * label_line_h), line, fill=(0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def build_sheets(
    landmark_csv: Path,
    errors_csv: Path,
    train_zip: Path,
    out_dir: Path,
    thumb_width: int,
    cols: int,
) -> list[Path]:
    landmark_rows = read_csv(landmark_csv)
    error_rows = read_csv(errors_csv)
    rows_by_pilot_id = {row["pilot_id"]: row for row in landmark_rows}

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in error_rows:
        grouped[(row["task"], row["rule"])].append(row)

    written: list[Path] = []
    for (task, rule), rows in sorted(grouped.items()):
        out_path = out_dir / sheet_name(task, rule)
        rows = sorted(rows, key=lambda row: (error_kind(row), int(row["pilot_id"] or 0)))
        draw_sheet(rows, rows_by_pilot_id, train_zip, out_path, thumb_width, cols)
        if out_path.exists():
            written.append(out_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark-csv", type=Path, default=DEFAULT_LANDMARK_CSV)
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_sheets(
        landmark_csv=args.landmark_csv,
        errors_csv=args.errors_csv,
        train_zip=args.train_zip,
        out_dir=args.out_dir,
        thumb_width=args.thumb_width,
        cols=args.cols,
    )
    print(f"Wrote {len(paths)} sheets to: {args.out_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
