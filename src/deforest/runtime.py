"""Hardware & environment detection with sensible autoscaled defaults.

The same code runs on a MacBook (CPU only), a CUDA box, or an AMD MI300X box
with ROCm. Choices that change per host should come from here, not from a
hard‑coded config.

Detection order
---------------
1. ``DEFOREST_DEVICE`` env var — explicit override (``cpu`` | ``cuda`` | ``rocm``).
2. PyTorch ``torch.cuda.is_available()``.
   - On AMD GPUs with the ROCm PyTorch wheel this returns ``True`` and
     ``torch.version.hip`` is a version string. We expose that as ``rocm``.
3. Fallback: ``cpu``.

Scales returned by :func:`autoscale_defaults` are intentionally conservative
upper bounds — callers may still clamp them for particular jobs. They are
tuned for a single MI300X (1 GPU, 192 GB HBM, ~240 vCPUs, 5 TB scratch).
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Device = Literal["cpu", "cuda", "rocm"]


@dataclass(frozen=True)
class Hardware:
    device: Device
    device_name: str
    gpu_count: int
    gpu_vram_gib: float  # per-device
    cpu_count: int
    total_ram_gib: float
    scratch_dir: Path
    scratch_free_gib: float
    supports_bf16: bool

    def summary(self) -> str:
        return (
            f"device={self.device} ({self.device_name}) × {self.gpu_count}, "
            f"vram={self.gpu_vram_gib:.0f} GiB/gpu, "
            f"cpu={self.cpu_count} cores, "
            f"ram={self.total_ram_gib:.0f} GiB, "
            f"scratch={self.scratch_dir} ({self.scratch_free_gib:.0f} GiB free), "
            f"bf16={self.supports_bf16}"
        )


def detect_hardware(
    scratch_dir_env: str = "DEFOREST_SCRATCH",
    default_scratch: str = "/mnt/scratch",
) -> Hardware:
    device, name, n_gpu, vram, bf16 = _detect_device()
    cpu_count = os.cpu_count() or 1
    total_ram_gib = _total_ram_gib()

    scratch = Path(os.environ.get(scratch_dir_env, default_scratch))
    if not scratch.exists():
        # Fall back to the workspace if /mnt/scratch isn't available (e.g. on
        # a laptop). 5 TB NVMe only matters on the actual server.
        scratch = Path.cwd() / ".cache" / "deforest"
    scratch.mkdir(parents=True, exist_ok=True)
    free = _disk_free_gib(scratch)

    return Hardware(
        device=device,
        device_name=name,
        gpu_count=n_gpu,
        gpu_vram_gib=vram,
        cpu_count=cpu_count,
        total_ram_gib=total_ram_gib,
        scratch_dir=scratch,
        scratch_free_gib=free,
        supports_bf16=bf16,
    )


# ---------------------------------------------------------------------------
# Autoscaling heuristics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scales:
    """Resource budgets derived from :class:`Hardware`."""

    torch_device: str         # "cuda", "cuda:0" etc.
    amp_dtype: str            # "bfloat16" or "float16" or "float32"
    batch_size: int           # per-GPU batch size
    grad_accum: int           # gradient accumulation steps
    num_workers: int          # DataLoader workers
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    lgbm_threads: int
    preprocess_workers: int
    max_cached_patches: int   # how many preprocessed patches to keep on scratch
    patch_size: int           # H=W of training crops


def autoscale_defaults(hw: Hardware, patch_size: int = 256) -> Scales:
    if hw.device == "cpu":
        return Scales(
            torch_device="cpu",
            amp_dtype="float32",
            batch_size=4,
            grad_accum=1,
            num_workers=min(4, hw.cpu_count),
            prefetch_factor=2,
            pin_memory=False,
            persistent_workers=False,
            lgbm_threads=max(1, hw.cpu_count - 1),
            preprocess_workers=max(1, hw.cpu_count - 1),
            max_cached_patches=5_000,
            patch_size=patch_size,
        )

    # GPU path (ROCm or CUDA).
    amp = "bfloat16" if hw.supports_bf16 else "float16"

    # Batch size — AEF-only features are ~193 channels, S1/S2 stats add ~15.
    # A 256×256 patch × ~220 ch × 2 bytes (bf16) × batch × ~4 (activations) ≈
    # 100 MB per sample at fwd+bwd. At 192 GB we can push batch_size very
    # high; we clamp to 256 to keep learning-rate tuning sane.
    if hw.gpu_vram_gib >= 120:
        batch_size = 128
    elif hw.gpu_vram_gib >= 40:
        batch_size = 64
    elif hw.gpu_vram_gib >= 24:
        batch_size = 32
    else:
        batch_size = 16

    num_workers = min(32, max(4, hw.cpu_count // 8))
    preprocess_workers = max(1, hw.cpu_count - 4)
    lgbm_threads = max(1, hw.cpu_count - 4)

    return Scales(
        torch_device=f"{hw.device}:0" if hw.device == "cuda" else "cuda:0",  # ROCm still uses "cuda" in torch
        amp_dtype=amp,
        batch_size=batch_size,
        grad_accum=1,
        num_workers=num_workers,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
        lgbm_threads=lgbm_threads,
        preprocess_workers=preprocess_workers,
        max_cached_patches=200_000,
        patch_size=patch_size,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_device() -> tuple[Device, str, int, float, bool]:
    override = os.environ.get("DEFOREST_DEVICE", "").strip().lower()
    if override in {"cpu", "cuda", "rocm"}:
        if override == "cpu":
            return "cpu", platform.processor() or "cpu", 0, 0.0, False
        return _probe_torch_gpu(force_as=override)
    return _probe_torch_gpu()


def _probe_torch_gpu(force_as: str | None = None) -> tuple[Device, str, int, float, bool]:
    try:
        import torch  # noqa: WPS433 — lazy import, stays optional
    except ImportError:
        return "cpu", platform.processor() or "cpu", 0, 0.0, False

    if not torch.cuda.is_available():
        return "cpu", platform.processor() or "cpu", 0, 0.0, False

    is_rocm = bool(getattr(torch.version, "hip", None))
    device: Device = (
        "rocm" if (force_as == "rocm" or (force_as is None and is_rocm))
        else "cuda"
    )

    n = torch.cuda.device_count()
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = device
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        vram = total_bytes / (1024**3)
    except Exception:
        vram = 0.0

    try:
        bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        bf16 = False

    return device, name, n, vram, bf16


def _total_ram_gib() -> float:
    # os.sysconf works on Linux/macOS; the fallback is only for Windows.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024**3)
    except (AttributeError, ValueError, OSError):
        try:
            import psutil  # type: ignore

            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 0.0


def _disk_free_gib(p: Path) -> float:
    try:
        return shutil.disk_usage(p).free / (1024**3)
    except OSError:
        return 0.0
