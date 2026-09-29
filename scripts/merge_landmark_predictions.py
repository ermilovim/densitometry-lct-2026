from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY_CSV = ROOT / "artifacts" / "inference_landmark_predictions_heatmap.csv"
DEFAULT_FALLBACK_CSV = ROOT / "artifacts" / "inference_landmark_predictions_knn.csv"
DEFAULT_OUT_CSV = ROOT / "artifacts" / "inference_landmark_predictions_hybrid.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("study_key", ""), row.get("member", "")


def merge_predictions(primary_rows: list[dict[str, str]], fallback_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    fallback_by_key = {key(row): row for row in fallback_rows}
    merged: list[dict[str, Any]] = []
    for primary in primary_rows:
        fallback = fallback_by_key.get(key(primary), {})
        if primary.get("landmark_status") == "predicted":
            row = dict(primary)
            row["landmark_model"] = f"hybrid:{primary.get('landmark_model', '')}"
            row["hybrid_source"] = "primary"
        elif fallback:
            row = dict(fallback)
            row["landmark_model"] = f"hybrid:{fallback.get('landmark_model', '')}"
            row["hybrid_source"] = "fallback"
        else:
            row = dict(primary)
            row["landmark_model"] = "hybrid:none"
            row["hybrid_source"] = "missing"
        merged.append(row)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-csv", type=Path, default=DEFAULT_PRIMARY_CSV)
    parser.add_argument("--fallback-csv", type=Path, default=DEFAULT_FALLBACK_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_predictions(read_csv(args.primary_csv), read_csv(args.fallback_csv))
    fieldnames = list(rows[0].keys()) if rows else []
    if "hybrid_source" not in fieldnames:
        fieldnames.append("hybrid_source")
    write_csv(args.out_csv, rows, fieldnames)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("hybrid_source", "")] = counts.get(row.get("hybrid_source", ""), 0) + 1
    print(f"Rows: {len(rows)}")
    for source, count in sorted(counts.items()):
        print(f"{source}: {count}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
