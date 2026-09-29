"""Compatibility inference for the checkpoint published by DXA-to-3D.

The upstream checkpoint belongs to its original ResNet50 + Bottleneck
Transformer implementation (1254 outputs), while the current upstream
``inference.py`` expects model classes that are absent from the repository.
The attention block follows lucidrains/bottleneck-transformer-pytorch (MIT).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import einsum, nn
from torchvision.models import resnet50


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


def _relative_to_absolute(value):
    batch, heads, length, _ = value.shape
    padded = torch.cat((value, value.new_zeros(batch, heads, length, 1)), dim=3)
    padded = padded.reshape(batch, heads, -1)
    padded = torch.cat((padded, value.new_zeros(batch, heads, length - 1)), dim=2)
    return padded.reshape(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1:]


def _relative_logits_1d(query, relative_keys):
    batch, heads, height, width, _ = query.shape
    logits = einsum("bhxyd,rd->bhxyr", query, relative_keys)
    logits = _relative_to_absolute(logits.reshape(batch, heads * height, width, -1))
    logits = logits.reshape(batch, heads, height, width, width).unsqueeze(3)
    return logits.expand(-1, -1, -1, height, -1, -1)


class RelativePosition(nn.Module):
    def __init__(self, feature_size, head_dim):
        super().__init__()
        height, width = _pair(feature_size)
        scale = head_dim ** -0.5
        self.feature_size = (height, width)
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, head_dim) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, head_dim) * scale)

    def forward(self, query):
        height, width = self.feature_size
        query = query.reshape(*query.shape[:2], height, width, query.shape[-1])
        width_logits = _relative_logits_1d(query, self.rel_width)
        width_logits = width_logits.permute(0, 1, 2, 4, 3, 5).reshape(query.shape[0], query.shape[1], height * width, height * width)
        height_logits = _relative_logits_1d(query.permute(0, 1, 3, 2, 4), self.rel_height)
        height_logits = height_logits.permute(0, 1, 4, 2, 5, 3).reshape(query.shape[0], query.shape[1], height * width, height * width)
        return width_logits + height_logits


class Attention(nn.Module):
    def __init__(self, dim, feature_size, heads=4, head_dim=128):
        super().__init__()
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Conv2d(dim, heads * head_dim * 3, 1, bias=False)
        self.pos_emb = RelativePosition(feature_size, head_dim)

    def forward(self, feature_map):
        batch, _, height, width = feature_map.shape
        query, key, value = self.to_qkv(feature_map).chunk(3, dim=1)
        reshape = lambda item: item.reshape(batch, self.heads, -1, height * width).transpose(2, 3)
        query, key, value = map(reshape, (query, key, value))
        similarity = einsum("bhid,bhjd->bhij", query * self.scale, key) + self.pos_emb(query)
        output = einsum("bhij,bhjd->bhid", similarity.softmax(dim=-1), value)
        return output.transpose(2, 3).reshape(batch, -1, height, width)


class BottleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.shortcut = nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            Attention(512, (7, 7)), nn.Identity(), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 2048, 1, bias=False), nn.BatchNorm2d(2048),
        )
        self.activation = nn.ReLU()

    def forward(self, value):
        return self.activation(self.net(value) + self.shortcut(value))


class BottleStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*(BottleBlock() for _ in range(3)))

    def forward(self, value):
        return self.net(value)


class PublishedDXA3DModel(nn.Module):
    """Architecture matching the official 2024 checkpoint exactly."""

    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.layer = BottleStack()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(2048, 209 * 6)

    def forward(self, value):
        return self.linear(self.flatten(self.pool(self.layer(self.model(value)))))


def predict_curves(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    model = PublishedDXA3DModel().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        return model(tensor.to(device)).reshape(209, 6).cpu().numpy()
