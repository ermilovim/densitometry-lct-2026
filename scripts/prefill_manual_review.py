from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_artifacts import (
    DEFAULT_TEST_ZIP,
    DEFAULT_TRAIN_ZIP,
    read_dicom_lite,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_CSV = ROOT / "artifacts" / "manual_image_review.csv"
DEFAULT_INDEX_CSV = ROOT / "artifacts" / "dicom_index.csv"
DEFAULT_REPORT_CSV = ROOT / "artifacts" / "manual_prefill_report.csv"


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


def nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def append_note(existing: str, note: str) -> str:
    if not note:
        return existing
    parts = [part.strip() for part in existing.split(";") if part.strip()]
    if note not in parts:
        parts.append(note)
    return "; ".join(parts)


def pixel_hashes(index_rows: list[dict[str, str]], train_zip: Path, test_zip: Path) -> dict[str, str]:
    rows_by_archive: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in index_rows:
        rows_by_archive[row["archive"]].append(row)

    zip_by_name = {
        train_zip.name: train_zip,
        test_zip.name: test_zip,
    }
    hashes: dict[str, str] = {}

    for archive_name, rows in rows_by_archive.items():
        zip_path = zip_by_name.get(archive_name)
        if zip_path is None or not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as zf:
            for row in rows:
                member = row["member"]
                try:
                    ds = read_dicom_lite(zf.read(member))
                except Exception:
                    continue
                if ds.pixel_data is None:
                    continue
                payload = ds.pixel_data
                digest = hashlib.sha1(payload).hexdigest()
                hashes[member] = digest
    return hashes


def base_region(row: dict[str, str]) -> str:
    hint = row.get("region_hint", "")
    if hint in {"spine", "right_hip", "left_hip", "hip_unknown_side"}:
        return hint
    columns = row.get("columns", "")
    if columns == "300":
        return "spine"
    if columns == "280":
        return "hip_unknown_side"
    return "unknown"


def prefill_rows(
    review_rows: list[dict[str, str]],
    index_rows: list[dict[str, str]],
    hashes: dict[str, str],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    index_by_member = {row["member"]: row for row in index_rows}
    hash_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in index_rows:
        digest = hashes.get(row["member"], "")
        if not digest:
            continue
        region = base_region(row)
        key = (row["study_folder"], region, digest)
        hash_groups[key].append(row["member"])

    exact_duplicate_of: dict[str, str] = {}
    exact_group_representative: dict[str, str] = {}
    for members in hash_groups.values():
        canonical = sorted(
            members,
            key=lambda m: (
                index_by_member[m].get("is_largest_in_region", "") == "1",
                -int(index_by_member[m].get("canonical_rank_in_region") or 999999),
                m,
            ),
            reverse=True,
        )[0]
        for member in members:
            exact_group_representative[member] = canonical
        if len(members) <= 1:
            continue
        for member in members:
            if member != canonical:
                exact_duplicate_of[member] = canonical

    report: list[dict[str, Any]] = []
    changed_counts: Counter[str] = Counter()

    for row in review_rows:
        member = row["member"]
        idx = index_by_member.get(member, row)
        suggested_region = base_region(idx)
        if suggested_region == "hip_unknown_side":
            suggested_canonical = "1" if exact_group_representative.get(member) == member else "0"
        else:
            suggested_canonical = "1" if idx.get("is_largest_in_region", "") == "1" else "0"
        notes: list[str] = []

        if suggested_region == "hip_unknown_side":
            notes.append("проверить сторону бедра")
        if suggested_region == "unknown":
            notes.append("нестандартный размер")
        if member in exact_duplicate_of:
            notes.append(f"точный дубль: {exact_duplicate_of[member]}")
            if not nonempty(row.get("manual_is_canonical", "")):
                suggested_canonical = "0"
        if idx.get("columns", "") not in {"280", "300"}:
            notes.append(f"ширина={idx.get('columns', '')}")
        if idx.get("candidate_rank_in_region", row.get("candidate_rank_in_region", "")) not in {"", "1"}:
            notes.append(
                "ранг кандидата="
                + str(idx.get("canonical_rank_in_region", row.get("candidate_rank_in_region", "")))
            )

        if not nonempty(row.get("manual_region", "")):
            row["manual_region"] = suggested_region
            changed_counts["manual_region"] += 1
        if not nonempty(row.get("manual_is_canonical", "")):
            row["manual_is_canonical"] = suggested_canonical
            changed_counts["manual_is_canonical"] += 1
        for note in notes:
            before = row.get("manual_notes", "")
            after = append_note(before, note)
            if after != before:
                row["manual_notes"] = after
                changed_counts["manual_notes"] += 1

        if notes or row["manual_region"] in {"unknown", "duplicate", "hip_unknown_side"}:
            report.append(
                {
                    "split": row.get("split", ""),
                    "study_folder": row.get("study_folder", ""),
                    "member": member,
                    "filename": row.get("filename", ""),
                    "columns": row.get("columns", ""),
                    "rows": row.get("rows", ""),
                    "manual_region": row.get("manual_region", ""),
                    "manual_is_canonical": row.get("manual_is_canonical", ""),
                    "manual_notes": row.get("manual_notes", ""),
                }
            )

    print("Updated fields:", dict(changed_counts))
    print("Rows for visual review:", len(report))
    print("Manual region counts:", dict(Counter(row["manual_region"] for row in review_rows)))
    return review_rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--train-zip", type=Path, default=DEFAULT_TRAIN_ZIP)
    parser.add_argument("--test-zip", type=Path, default=DEFAULT_TEST_ZIP)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_rows = read_csv(args.review_csv)
    index_rows = read_csv(args.index_csv)
    hashes = pixel_hashes(index_rows, args.train_zip, args.test_zip)
    fieldnames = list(review_rows[0].keys()) if review_rows else []

    updated_rows, report = prefill_rows(review_rows, index_rows, hashes)

    if args.dry_run:
        print("Dry run: no files written")
        return

    if not args.no_backup and args.review_csv.exists():
        backup_path = args.review_csv.with_suffix(".before_prefill.csv")
        shutil.copy2(args.review_csv, backup_path)
        print(f"Backup written: {backup_path}")

    write_csv(args.review_csv, updated_rows, fieldnames=fieldnames)
    write_csv(args.report_csv, report)
    print(f"Wrote: {args.review_csv}")
    print(f"Wrote: {args.report_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
