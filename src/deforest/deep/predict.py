"""Deep inference.

Runs :class:`ChangeUNet` over every preprocessed test tile using overlapping
sliding-window inference with Hann-window overlap-add stitching to avoid
seams at patch boundaries. Produces a ``(prob, month)`` pair per tile where

* ``prob``   — float32 probability raster on the Sentinel-2 grid.
* ``month``  — int32 YYMM raster (0 when the change head is inactive at a
  pixel or when the month head never fires).

Post-processing (threshold, morphology, polygonize, area filter, time_step
attachment) happens in :mod:`deforest.postprocess` exactly as for the CPU
pipeline.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from ..runtime import autoscale_defaults, detect_hardware
from .dataset import CachedTile, hann_window_2d, iterate_tile_patches
from .model import ChangeUNetConfig, build_model


def load_checkpoint(path: str | Path, device: str) -> tuple["torch.nn.Module", dict]:
    if torch is None:
        raise ImportError("PyTorch required")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = ChangeUNetConfig(**ckpt["cfg"])
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, ckpt


def predict_tile(
    tile: CachedTile,
    model,
    *,
    patch_size: int = 256,
    overlap: int = 64,
    batch_size: int = 32,
    amp_dtype: str = "bfloat16",
    device: str = "cuda:0",
    month_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run overlapping inference over one tile, return (prob, month_yymm)."""
    if torch is None:
        raise ImportError("PyTorch required")

    _, h, w = tile.features.shape
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    month_accum = np.zeros((h, w), dtype=np.float32)  # soft-argmax accumulator
    window = hann_window_2d(patch_size)

    # Calendar: month index 0 = start['2020-01']. We keep it implicit — the
    # caller passes in ``start_year`` / ``start_month`` from config when
    # translating ``month_accum`` to YYMM.
    dtype_t = _torch_dtype(amp_dtype)
    use_amp = dtype_t != torch.float32

    batch_feats, batch_pos = [], []
    with torch.no_grad():
        for y, x, feats, forest in iterate_tile_patches(tile, patch_size, overlap):
            batch_feats.append(feats)
            batch_pos.append((y, x, forest))
            if len(batch_feats) >= batch_size:
                _flush(batch_feats, batch_pos, model, device, dtype_t, use_amp, window,
                       prob_sum, weight_sum, month_accum)
                batch_feats.clear(); batch_pos.clear()
        if batch_feats:
            _flush(batch_feats, batch_pos, model, device, dtype_t, use_amp, window,
                   prob_sum, weight_sum, month_accum)

    prob = prob_sum / np.maximum(weight_sum, 1e-6)
    # ``month_accum`` holds the **expected month index** (soft argmax) weighted
    # by the Hann window. Dividing by weight_sum yields a per-pixel expected
    # month index; convert to YYMM with the caller's month-calendar start.
    return prob.astype(np.float32), month_accum / np.maximum(weight_sum, 1e-6)


def month_idx_to_yymm(expected_idx: np.ndarray, start_year: int, start_month: int) -> np.ndarray:
    idx = np.rint(expected_idx).astype(np.int32)
    year = start_year + (idx + start_month - 1) // 12
    month = (idx + start_month - 1) % 12 + 1
    yymm = (year % 100) * 100 + month
    yymm[idx < 0] = 0
    return yymm.astype(np.int32)


def autoscale_infer(cfg: dict) -> tuple[str, str, int]:
    """Return (device_str, amp_dtype, batch_size) for inference."""
    hw = detect_hardware()
    scales = autoscale_defaults(hw, patch_size=int(cfg.get("patch_size", 256)))
    device = scales.torch_device if hw.device != "cpu" else "cpu"
    amp = scales.amp_dtype
    batch = cfg.get("batch_size") or scales.batch_size
    return device, amp, int(batch)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _flush(
    feats_list,
    pos_list,
    model,
    device: str,
    dtype_t,
    use_amp: bool,
    window: np.ndarray,
    prob_sum: np.ndarray,
    weight_sum: np.ndarray,
    month_accum: np.ndarray,
) -> None:
    x = torch.from_numpy(np.stack(feats_list, axis=0)).to(device, non_blocking=True)
    if use_amp:
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype_t):
            out = model(x)
    else:
        out = model(x)

    change_prob = torch.sigmoid(out["change_logits"]).float().cpu().numpy()
    month_logits = out["month_logits"].float().cpu().numpy()
    # Expected month index = softmax over month channel then weighted sum.
    m = _softmax(month_logits, axis=1)
    idx_grid = np.arange(m.shape[1], dtype=np.float32)[None, :, None, None]
    expected_idx = (m * idx_grid).sum(axis=1)  # (B, H, W)

    for (y, x_, forest), p_patch, ei_patch in zip(pos_list, change_prob, expected_idx):
        ph, pw = p_patch.shape
        w_ = window[:ph, :pw]
        prob_sum[y : y + ph, x_ : x_ + pw] += p_patch * w_
        weight_sum[y : y + ph, x_ : x_ + pw] += w_
        month_accum[y : y + ph, x_ : x_ + pw] += ei_patch * w_


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def _torch_dtype(name: str):
    if torch is None:
        raise ImportError("PyTorch required")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name.lower()]
