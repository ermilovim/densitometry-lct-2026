from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "artifacts" / "keypoint_dataset"
DEFAULT_ANNOTATIONS_CSV = DEFAULT_DATASET_DIR / "annotations.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "landmark_heatmap"

PIXEL_SPACING_X_MM = 0.60
PIXEL_SPACING_Y_MM = 1.05

SPINE_KEYS = [
    "spine_th12_center",
    "spine_l1_center",
    "spine_l2_center",
    "spine_l3_center",
    "spine_l4_center",
    "spine_l5_center",
    "spine_left_iliac_crest",
    "spine_right_iliac_crest",
]

HIP_KEYS = [
    "hip_greater_trochanter_top_edge",
    "hip_greater_trochanter_lateral_edge",
    "hip_lesser_trochanter_upper",
    "hip_lesser_trochanter_tip",
    "hip_lesser_trochanter_lower",
    "hip_ischium_edge",
]

LESSER_KEYS = [
    "hip_lesser_trochanter_upper",
    "hip_lesser_trochanter_tip",
    "hip_lesser_trochanter_lower",
]


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


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def region_keys(region_group: str) -> list[str]:
    if region_group == "spine":
        return SPINE_KEYS
    if region_group == "hip":
        return HIP_KEYS
    raise ValueError(f"Unsupported region group: {region_group}")


def filter_rows(rows: list[dict[str, str]], region_group: str, split: str) -> list[dict[str, str]]:
    if region_group == "spine":
        regions = {"spine"}
    elif region_group == "hip":
        regions = {"left_hip", "right_hip"}
    else:
        raise ValueError(f"Unsupported region group: {region_group}")
    return [row for row in rows if row["split"] == split and row["anatomical_region"] in regions]


def gaussian_heatmaps(
    points: torch.Tensor,
    visible: torch.Tensor,
    size: int,
    sigma: float,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    xx = xx.float()
    yy = yy.float()
    maps = []
    denom = 2.0 * sigma * sigma
    for i in range(points.shape[0]):
        if visible[i] <= 0:
            maps.append(torch.zeros((size, size), dtype=torch.float32))
            continue
        x, y = points[i]
        maps.append(torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / denom))
    return torch.stack(maps, dim=0)


class LandmarkDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        dataset_dir: Path,
        keys: list[str],
        image_size: int,
        sigma: float,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.dataset_dir = dataset_dir
        self.keys = keys
        self.image_size = image_size
        self.sigma = sigma
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(self.dataset_dir / row["image_path"]).convert("L")
        width = float(row["width"])
        height = float(row["height"])
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0

        if self.augment:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.08, 0.08)
            arr = np.clip(arr * gain + bias, 0.0, 1.0)
            if random.random() < 0.5:
                noise = np.random.normal(0.0, 0.015, size=arr.shape).astype(np.float32)
                arr = np.clip(arr + noise, 0.0, 1.0)

        mean = float(arr.mean())
        std = float(arr.std())
        arr = (arr - mean) / max(std, 1e-6)
        image_tensor = torch.from_numpy(arr[None, :, :].astype(np.float32))

        points = []
        visible = []
        for key in self.keys:
            x_norm = as_float(row.get(f"{key}_x_norm", ""))
            y_norm = as_float(row.get(f"{key}_y_norm", ""))
            is_visible = row.get(f"{key}_visible") == "1" and x_norm is not None and y_norm is not None
            visible.append(1.0 if is_visible else 0.0)
            if is_visible:
                points.append([x_norm * (self.image_size - 1), y_norm * (self.image_size - 1)])
            else:
                points.append([0.0, 0.0])

        point_tensor = torch.tensor(points, dtype=torch.float32)
        visible_tensor = torch.tensor(visible, dtype=torch.float32)
        heatmaps = gaussian_heatmaps(point_tensor, visible_tensor, self.image_size, self.sigma)

        return {
            "image": image_tensor,
            "heatmaps": heatmaps,
            "visible": visible_tensor,
            "points": point_tensor,
            "width": torch.tensor(width, dtype=torch.float32),
            "height": torch.tensor(height, dtype=torch.float32),
            "pilot_id": row["pilot_id"],
            "image_number": row["image_number"],
            "anatomical_region": row["anatomical_region"],
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, out_channels: int, base: int = 24) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        mid = self.mid(self.pool2(e2))
        d2 = self.up2(mid)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def heatmap_loss(logits: torch.Tensor, target: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    point_weights = 1.0 + 50.0 * target
    weights = visible[:, :, None, None] * point_weights
    sq = (pred - target) ** 2 * weights
    denom = weights.sum()
    return sq.sum() / denom.clamp_min(1.0)


def spatial_softargmax(logits: torch.Tensor, beta: float = 8.0) -> torch.Tensor:
    batch, channels, height, width = logits.shape
    probs = torch.softmax((logits * beta).reshape(batch, channels, -1), dim=2)
    xs = torch.linspace(0.0, 1.0, width, device=logits.device)
    ys = torch.linspace(0.0, 1.0, height, device=logits.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = (probs * xx.reshape(1, 1, -1)).sum(dim=2)
    y = (probs * yy.reshape(1, 1, -1)).sum(dim=2)
    return torch.stack([x, y], dim=2)


def point_to_line_distance(point: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    ap = point - a
    cross = torch.abs(ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0])
    denom = torch.linalg.norm(ab, dim=1).clamp_min(1e-6)
    return cross / denom


def lesser_geometry_loss(
    logits: torch.Tensor,
    points: torch.Tensor,
    visible: torch.Tensor,
    keys: list[str],
    image_size: int,
) -> torch.Tensor:
    if not all(key in keys for key in LESSER_KEYS):
        return logits.new_tensor(0.0)

    idx = torch.tensor([keys.index(key) for key in LESSER_KEYS], device=logits.device)
    mask = (visible.index_select(1, idx).sum(dim=1) == len(LESSER_KEYS)).float()
    if mask.sum() <= 0:
        return logits.new_tensor(0.0)

    pred = spatial_softargmax(logits).index_select(1, idx)
    target = points.index_select(1, idx) / float(image_size - 1)

    upper = pred[:, 0]
    tip = pred[:, 1]
    lower = pred[:, 2]
    target_upper = target[:, 0]
    target_tip = target[:, 1]
    target_lower = target[:, 2]

    margin = 3.0 / float(image_size - 1)
    order = torch.relu(upper[:, 1] + margin - tip[:, 1]) ** 2
    order = order + torch.relu(tip[:, 1] + margin - lower[:, 1]) ** 2

    pred_pairwise = torch.stack(
        [
            torch.linalg.norm(upper - tip, dim=1),
            torch.linalg.norm(tip - lower, dim=1),
            torch.linalg.norm(upper - lower, dim=1),
        ],
        dim=1,
    )
    target_pairwise = torch.stack(
        [
            torch.linalg.norm(target_upper - target_tip, dim=1),
            torch.linalg.norm(target_tip - target_lower, dim=1),
            torch.linalg.norm(target_upper - target_lower, dim=1),
        ],
        dim=1,
    )
    pairwise = ((pred_pairwise - target_pairwise) ** 2).mean(dim=1)

    pred_prominence = point_to_line_distance(tip, upper, lower)
    target_prominence = point_to_line_distance(target_tip, target_upper, target_lower)
    prominence = (pred_prominence - target_prominence) ** 2

    min_base = 8.0 / float(image_size - 1)
    collapse = torch.relu(min_base - pred_pairwise[:, 2]) ** 2

    loss = 10.0 * order + pairwise + prominence + 2.0 * collapse
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def total_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    visible: torch.Tensor,
    points: torch.Tensor,
    keys: list[str],
    image_size: int,
    geometry_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heat = heatmap_loss(logits, target, visible)
    geom = lesser_geometry_loss(logits, points, visible, keys, image_size) if geometry_loss_weight > 0 else logits.new_tensor(0.0)
    return heat + geometry_loss_weight * geom, heat, geom


def predict_points(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    batch, channels, height, width = probs.shape
    flat_idx = probs.reshape(batch, channels, -1).argmax(dim=2)
    y = torch.div(flat_idx, width, rounding_mode="floor").float()
    x = (flat_idx % width).float()
    return torch.stack([x, y], dim=2)


def point_error_mm(pred: torch.Tensor, target: torch.Tensor, width: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    pred_x = pred[:, 0] / (pred.new_tensor(pred.shape[0]).fill_(0) + 1)
    _ = pred_x
    scale = pred.new_tensor([PIXEL_SPACING_X_MM, PIXEL_SPACING_Y_MM])
    original_pred = torch.stack(
        [
            pred[:, 0] / 255.0 * width,
            pred[:, 1] / 255.0 * height,
        ],
        dim=1,
    )
    original_target = torch.stack(
        [
            target[:, 0] / 255.0 * width,
            target[:, 1] / 255.0 * height,
        ],
        dim=1,
    )
    diff_mm = (original_pred - original_target) * scale
    return torch.sqrt((diff_mm * diff_mm).sum(dim=1))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, round((len(vals) - 1) * p))
    return vals[idx]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    keys: list[str],
    image_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    by_point: dict[str, list[float]] = defaultdict(list)

    for batch in loader:
        images = batch["image"].to(device)
        points = batch["points"].to(device)
        visible = batch["visible"].to(device)
        widths = batch["width"].to(device)
        heights = batch["height"].to(device)
        logits = model(images)
        preds = predict_points(logits)
        denom = float(image_size - 1)

        for b in range(images.shape[0]):
            for k, key in enumerate(keys):
                if visible[b, k].item() <= 0:
                    continue
                pred_orig_x = preds[b, k, 0].item() / denom * widths[b].item()
                pred_orig_y = preds[b, k, 1].item() / denom * heights[b].item()
                true_orig_x = points[b, k, 0].item() / denom * widths[b].item()
                true_orig_y = points[b, k, 1].item() / denom * heights[b].item()
                dx_mm = (pred_orig_x - true_orig_x) * PIXEL_SPACING_X_MM
                dy_mm = (pred_orig_y - true_orig_y) * PIXEL_SPACING_Y_MM
                error = math.hypot(dx_mm, dy_mm)
                errors.append(error)
                by_point[key].append(error)
                rows.append(
                    {
                        "pilot_id": batch["pilot_id"][b],
                        "image_number": batch["image_number"][b],
                        "anatomical_region": batch["anatomical_region"][b],
                        "point": key,
                        "pred_x": f"{pred_orig_x:.3f}",
                        "pred_y": f"{pred_orig_y:.3f}",
                        "true_x": f"{true_orig_x:.3f}",
                        "true_y": f"{true_orig_y:.3f}",
                        "error_mm": f"{error:.3f}",
                    }
                )

    metrics = {
        "mean_error_mm": mean(errors),
        "median_error_mm": percentile(errors, 0.5),
        "p90_error_mm": percentile(errors, 0.9),
        "n_points": float(len(errors)),
    }
    for key, vals in by_point.items():
        metrics[f"{key}_mean_error_mm"] = mean(vals)
    return rows, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["hip", "spine"], required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--geometry-loss-weight", type=float, default=0.005)
    parser.add_argument("--geometry-loss-start-epoch", type=int, default=20)
    parser.add_argument("--train-all", action="store_true", help="Train on previous train+val rows for final submission weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    keys = region_keys(args.region)
    all_rows = read_csv(args.annotations_csv)
    original_train_rows = filter_rows(all_rows, args.region, "train")
    original_val_rows = filter_rows(all_rows, args.region, "val")
    train_rows = original_train_rows + original_val_rows if args.train_all else original_train_rows
    val_rows = train_rows if args.train_all else original_val_rows
    if not train_rows or not val_rows:
        raise RuntimeError(f"Empty train/val split for region={args.region}")

    train_ds = LandmarkDataset(train_rows, args.dataset_dir, keys, args.image_size, args.sigma, augment=True)
    val_ds = LandmarkDataset(val_rows, args.dataset_dir, keys, args.image_size, args.sigma, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyUNet(out_channels=len(keys)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = args.out_dir / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")
    history: list[dict[str, Any]] = []

    print(f"Device: {device}")
    mode = "train_all" if args.train_all else "train_val"
    print(
        f"Mode: {mode} | Train rows: {len(train_rows)} | Eval rows: {len(val_rows)} "
        f"| Original train: {len(original_train_rows)} | Original val: {len(original_val_rows)} | Points: {len(keys)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device)
            target = batch["heatmaps"].to(device)
            visible = batch["visible"].to(device)
            points = batch["points"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            geometry_weight = (
                args.geometry_loss_weight
                if args.region == "hip" and epoch >= args.geometry_loss_start_epoch
                else 0.0
            )
            loss, heat_loss, geom_loss = total_loss(
                logits=logits,
                target=target,
                visible=visible,
                points=points,
                keys=keys,
                image_size=args.image_size,
                geometry_loss_weight=geometry_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        _rows, metrics = evaluate(model, val_loader, device, keys, args.image_size)
        row = {
            "epoch": epoch,
            "train_loss": f"{mean(losses):.6f}",
            "val_mean_error_mm": f"{metrics['mean_error_mm']:.3f}",
            "val_median_error_mm": f"{metrics['median_error_mm']:.3f}",
            "val_p90_error_mm": f"{metrics['p90_error_mm']:.3f}",
        }
        history.append(row)
        if metrics["mean_error_mm"] < best_metric:
            best_metric = metrics["mean_error_mm"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "region": args.region,
                    "keys": keys,
                    "image_size": args.image_size,
                    "sigma": args.sigma,
                    "metrics": metrics,
                },
                out_dir / "best.pt",
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}: loss={row['train_loss']} "
                f"val_mean={row['val_mean_error_mm']}mm "
                f"val_median={row['val_median_error_mm']}mm "
                f"val_p90={row['val_p90_error_mm']}mm"
            )

    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    eval_rows, metrics = evaluate(model, val_loader, device, keys, args.image_size)

    write_csv(out_dir / "history.csv", history)
    write_csv(out_dir / "val_predictions.csv", eval_rows)
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "region": args.region,
                "keys": keys,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "image_size": args.image_size,
                "epochs": args.epochs,
                "train_all": args.train_all,
                "geometry_loss_weight": args.geometry_loss_weight if args.region == "hip" else 0.0,
                "geometry_loss_start_epoch": args.geometry_loss_start_epoch if args.region == "hip" else 0,
                "original_train_rows": len(original_train_rows),
                "original_val_rows": len(original_val_rows),
                "best_metrics": metrics,
                "notes": (
                    "Final train-all model; metrics are in-sample sanity checks, not holdout validation."
                    if args.train_all
                    else "Metrics are measured on the held-out validation split."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Best val mean error: {metrics['mean_error_mm']:.3f} mm")
    print(f"Best val median error: {metrics['median_error_mm']:.3f} mm")
    print(f"Best val p90 error: {metrics['p90_error_mm']:.3f} mm")
    print(f"Wrote: {out_dir / 'best.pt'}")
    print(f"Wrote: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
