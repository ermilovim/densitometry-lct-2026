from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "positioning": {
        "dataset_dir": ROOT / "artifacts" / "spine_positioning_dataset",
        "label_col": "positioning_bad",
        "out_dir": ROOT / "artifacts" / "spine_classifiers" / "positioning",
    },
    "artifact": {
        "dataset_dir": ROOT / "artifacts" / "spine_artifact_dataset",
        "label_col": "artifact_bad",
        "out_dir": ROOT / "artifacts" / "spine_classifiers" / "artifact",
    },
}


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




def review_id(row: dict[str, str]) -> str:
    base = f"img{int(row['image_number']):03d}_pilot{int(row['pilot_id']):04d}_{row['filename']}"
    if row.get("synthetic") == "1":
        return f"{base}_{row.get('augmentation', 'synthetic')}"
    return base


def apply_manual_positioning_review(dataset_dir: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    review_csv = dataset_dir / "manual_iliac_review.csv"
    if not review_csv.exists():
        return rows

    reviews = {
        row["review_id"]: row
        for row in read_csv(review_csv)
        if row.get("review_status") == "done" and row.get("review_use_for_training", "1") == "1"
    }
    filtered: list[dict[str, str]] = []
    for row in rows:
        review = reviews.get(review_id(row))
        if review is None:
            continue
        item = dict(row)
        item["positioning_bad"] = review.get("review_positioning_bad", item.get("positioning_bad", ""))
        item["iliac_visible"] = review.get("review_iliac_visible", item.get("iliac_visible", ""))
        item["manual_review_id"] = review["review_id"]
        item["manual_review_updated_at"] = review.get("updated_at", "")
        filtered.append(item)
    if not filtered:
        raise ValueError(f"Manual positioning review exists but no rows selected for training: {review_csv}")
    return filtered


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def as_int(value: str) -> int:
    if value == "":
        return 0
    return int(float(value))


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32) / 255.0
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    mean = float(arr.mean())
    std = float(arr.std())
    return (arr - mean) / max(std, 1e-6)


class SpineClassificationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        dataset_dir: Path,
        label_col: str,
        image_size: int,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.dataset_dir = dataset_dir
        self.label_col = label_col
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(self.dataset_dir / row["image_path"]).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)

        if self.augment:
            if random.random() < 0.5:
                arr = np.ascontiguousarray(arr[:, ::-1])
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-12.0, 12.0)
            arr = np.clip(arr * gain + bias, 0.0, 255.0)
            if random.random() < 0.5:
                noise = np.random.normal(0.0, 3.0, size=arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0.0, 255.0)

        arr = normalize_image(arr)
        x = torch.from_numpy(arr[None, :, :].astype(np.float32))
        y = torch.tensor(float(as_int(row[self.label_col])), dtype=torch.float32)
        return {"image": x, "label": y, "row": row}


class SmallSpineCNN(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            self._block(base, base * 2),
            nn.MaxPool2d(2),
            self._block(base * 2, base * 4),
            nn.MaxPool2d(2),
            self._block(base * 4, base * 6),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(base * 6, 1),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)




def make_balanced_positioning_holdout(
    rows: list[dict[str, str]],
    per_class: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    originals = [
        row
        for row in rows
        if row.get("synthetic") == "0" and row.get("augmentation") == "original"
    ]
    good = [row for row in originals if as_int(row.get("positioning_bad", "")) == 0]
    bad = [row for row in originals if as_int(row.get("positioning_bad", "")) == 1]
    n = min(per_class, len(good), len(bad))
    if n < 1:
        raise ValueError("Need both good and bad original positioning rows for balanced holdout")

    rng = random.Random(seed)
    good = sorted(good, key=lambda row: int(row["image_number"]))
    bad = sorted(bad, key=lambda row: int(row["image_number"]))
    rng.shuffle(good)
    rng.shuffle(bad)
    val_rows = sorted(bad[:n] + good[:n], key=lambda row: int(row["image_number"]))
    val_studies = {row["study_folder"] for row in val_rows}
    train_rows = [row for row in rows if row["study_folder"] not in val_studies]
    if not train_rows:
        raise ValueError("Balanced positioning holdout left no training rows")
    return train_rows, val_rows


def make_sampler(rows: list[dict[str, str]], label_col: str) -> WeightedRandomSampler:
    labels = [as_int(row[label_col]) for row in rows]
    counts = {0: max(1, labels.count(0)), 1: max(1, labels.count(1))}
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def confusion(labels: list[int], preds: list[int]) -> dict[str, int]:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ba = (recall + specificity) / 2.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": ba,
        "accuracy": accuracy,
    }


def threshold_report(labels: list[int], probs: list[float]) -> dict[str, Any]:
    thresholds = sorted(set([0.5, *probs]))
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        preds = [int(prob >= threshold) for prob in probs]
        c = confusion(labels, preds)
        m = metrics_from_confusion(c)
        report = {"threshold": float(threshold), **c, **m}
        if best is None:
            best = report
            continue
        key = (report["balanced_accuracy"], report["f1"], report["recall"], -abs(report["threshold"] - 0.5))
        best_key = (best["balanced_accuracy"], best["f1"], best["recall"], -abs(best["threshold"] - 0.5))
        if key > best_key:
            best = report
    assert best is not None
    return best


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_col: str,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    labels: list[int] = []
    probs: list[float] = []
    rows_out: list[dict[str, Any]] = []

    for batch in loader:
        images = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(images)
        total_loss += float(criterion(logits, y).item())
        p = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        batch_labels = y.detach().cpu().numpy().astype(int).tolist()
        rows = batch["row"]
        for i, prob in enumerate(p):
            label = int(batch_labels[i])
            labels.append(label)
            probs.append(float(prob))
            rows_out.append(
                {
                    "split": rows["split"][i],
                    "image_path": rows["image_path"][i],
                    "image_number": rows["image_number"][i],
                    "pilot_id": rows["pilot_id"][i],
                    "augmentation": rows["augmentation"][i],
                    "synthetic": rows["synthetic"][i],
                    label_col: label,
                    "prob_bad": float(prob),
                }
            )

    report = threshold_report(labels, probs)
    for item in rows_out:
        item["pred_bad"] = int(item["prob_bad"] >= report["threshold"])
        item["correct"] = int(item["pred_bad"] == int(item[label_col]))
    return total_loss / max(1, len(labels)), rows_out, report


def train(args: argparse.Namespace) -> None:
    task_cfg = TASKS[args.task]
    dataset_dir = args.dataset_dir or task_cfg["dataset_dir"]
    label_col = task_cfg["label_col"]
    out_dir = args.out_dir or task_cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    rows = read_csv(dataset_dir / "annotations.csv")
    if args.task == "positioning":
        rows = apply_manual_positioning_review(dataset_dir, rows)
    if args.task == "positioning" and args.balanced_positioning_val and not args.train_all:
        original_train_rows, original_val_rows = make_balanced_positioning_holdout(
            rows,
            per_class=args.balanced_positioning_val_per_class,
            seed=args.seed,
        )
    else:
        original_train_rows = [row for row in rows if row["split"] == "train"]
        original_val_rows = [row for row in rows if row["split"] == "val"]
    train_rows = original_train_rows + original_val_rows if args.train_all else original_train_rows
    val_rows = train_rows if args.train_all else original_val_rows
    if not train_rows or not val_rows:
        raise ValueError("Need non-empty train and val splits")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = SpineClassificationDataset(train_rows, dataset_dir, label_col, args.image_size, augment=True)
    val_ds = SpineClassificationDataset(val_rows, dataset_dir, label_col, args.image_size, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_rows, label_col),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SmallSpineCNN(base=args.base_channels).to(device)
    train_labels = [as_int(row[label_col]) for row in train_rows]
    neg = max(1, train_labels.count(0))
    pos = max(1, train_labels.count(1))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -math.inf
    best_report: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    mode = "train_all" if args.train_all else "train_val"
    print(
        f"task={args.task} mode={mode} device={device} train={len(train_rows)} "
        f"eval={len(val_rows)} original_train={len(original_train_rows)} "
        f"original_val={len(original_val_rows)} pos_weight={pos_weight.item():.3f}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.shape[0]
            seen += images.shape[0]
        scheduler.step()

        val_loss, val_predictions, report = evaluate(model, val_loader, device, label_col)
        train_loss /= max(1, seen)
        score = report["balanced_accuracy"] + 0.01 * report["f1"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **report,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"ba={report['balanced_accuracy']:.3f} f1={report['f1']:.3f} "
            f"recall={report['recall']:.3f} precision={report['precision']:.3f} "
            f"thr={report['threshold']:.3f}"
        )

        if score > best_score:
            best_score = score
            best_report = report
            torch.save(
                {
                    "task": args.task,
                    "label_col": label_col,
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "state_dict": model.state_dict(),
                    "report": report,
                },
                out_dir / "best.pt",
            )
            write_csv(out_dir / "val_predictions.csv", val_predictions)

    assert best_report is not None
    write_csv(out_dir / "history.csv", history)
    summary = {
        "task": args.task,
        "dataset_dir": str(dataset_dir),
        "label_col": label_col,
        "out_dir": str(out_dir),
        "device": str(device),
        "train_all": args.train_all,
        "train_rows": len(train_rows),
        "eval_rows": len(val_rows),
        "original_train_rows": len(original_train_rows),
        "original_val_rows": len(original_val_rows),
        "train_positive": pos,
        "train_negative": neg,
        "manual_review_csv": str(dataset_dir / "manual_iliac_review.csv") if args.task == "positioning" and (dataset_dir / "manual_iliac_review.csv").exists() else "",
        "balanced_positioning_val": bool(args.task == "positioning" and args.balanced_positioning_val and not args.train_all),
        "balanced_positioning_val_per_class": args.balanced_positioning_val_per_class,
        "best_report": best_report,
        "notes": (
            "Final train-all model; metrics are in-sample sanity checks, not holdout validation."
            if args.train_all
            else "Val set is tiny; treat this as a sanity check, not a stable leaderboard estimate."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--train-all", action="store_true", help="Train on previous train+val rows for final submission weights.")
    parser.add_argument("--balanced-positioning-val", action="store_true", help="For positioning calibration, build a balanced original-image holdout and exclude same studies from train.")
    parser.add_argument("--balanced-positioning-val-per-class", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
