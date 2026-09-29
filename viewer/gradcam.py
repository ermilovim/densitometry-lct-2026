from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "positioning": ROOT / "artifacts" / "spine_classifiers" / "positioning" / "best.pt",
    "artifact": ROOT / "artifacts" / "spine_classifiers" / "artifact" / "best.pt",
}
VALIDATED_THRESHOLDS = {
    "positioning": 0.7809473276138306,
    "artifact": 0.38129135966300964,
}
Task = Literal["positioning", "artifact"]


def normalize_heatmap(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(values.min())
    max_value = float(values.max())
    if max_value <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return values / max_value


def colorize_heatmap(heatmap: np.ndarray, task: str) -> Image.Image:
    heatmap = normalize_heatmap(heatmap)
    alpha = np.clip((heatmap ** 1.35) * 170.0, 0.0, 170.0).astype(np.uint8)
    rgba = np.zeros((*heatmap.shape, 4), dtype=np.uint8)
    if task == "artifact":
        rgba[..., 0] = 255
        rgba[..., 1] = np.clip(70 + heatmap * 90, 0, 255).astype(np.uint8)
        rgba[..., 2] = 42
    else:
        rgba[..., 0] = 245
        rgba[..., 1] = np.clip(125 + heatmap * 80, 0, 255).astype(np.uint8)
        rgba[..., 2] = 28
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


@lru_cache(maxsize=2)
def load_bundle(task: str):
    if task not in MODEL_PATHS:
        raise ValueError("Unsupported Grad-CAM task")
    import torch
    from scripts.train_spine_classifier import SmallSpineCNN

    checkpoint = torch.load(MODEL_PATHS[task], map_location="cpu", weights_only=False)
    model = SmallSpineCNN(base=int(checkpoint.get("base_channels", 24)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "target_layer": model.features[11],
        "image_size": int(checkpoint["image_size"]),
        "threshold": float(VALIDATED_THRESHOLDS.get(task, checkpoint.get("report", {}).get("threshold", 0.5))),
    }


def image_tensor(image: Image.Image, image_size: int):
    import torch
    from scripts.train_spine_classifier import normalize_image

    resized = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = normalize_image(np.asarray(resized, dtype=np.float32))
    return torch.from_numpy(arr[None, None, :, :].astype(np.float32))


def gradcam_overlay(image: Image.Image, task: str) -> tuple[Image.Image, float, float]:
    import torch

    bundle = load_bundle(task)
    model = bundle["model"]
    target_layer = bundle["target_layer"]
    activations = None

    def save_activation(_module, _inputs, output):
        nonlocal activations
        activations = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(save_activation)
    try:
        tensor = image_tensor(image, int(bundle["image_size"]))
        model.zero_grad(set_to_none=True)
        logit = model(tensor)[0]
        logit.backward()
        if activations is None or activations.grad is None:
            raise RuntimeError("Grad-CAM activations are unavailable")
        grads = activations.grad.detach()
        maps = activations.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * maps).sum(dim=1))[0]
        cam_np = normalize_heatmap(cam.cpu().numpy())
    finally:
        handle.remove()

    heatmap = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L").resize(image.size, Image.Resampling.BICUBIC)
    overlay = colorize_heatmap(np.asarray(heatmap, dtype=np.float32) / 255.0, task)
    prob_bad = float(torch.sigmoid(logit.detach()).cpu().item())
    return overlay, prob_bad, float(bundle["threshold"])


def gradcam_png_bytes(image: Image.Image, task: str) -> bytes:
    overlay, prob_bad, threshold = gradcam_overlay(image, task)
    if task == "artifact" and prob_bad < threshold:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    output = BytesIO()
    overlay.save(output, format="PNG")
    return output.getvalue()
