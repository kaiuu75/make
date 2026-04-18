"""Sanity checks for :mod:`deforest.runtime`.

These tests intentionally do not require a GPU to be present — they verify
that autoscaling *does the right thing* when hardware is simulated, and
that the real :func:`detect_hardware` returns a usable object on the host
that the test suite is running on (laptop, CI, or the MI300X droplet).
"""

from __future__ import annotations

import os

from deforest.runtime import Hardware, autoscale_defaults, detect_hardware


def test_detect_hardware_runs_on_any_host(tmp_path):
    os.environ["DEFOREST_SCRATCH"] = str(tmp_path)
    hw = detect_hardware()
    assert hw.cpu_count >= 1
    assert hw.scratch_dir == tmp_path or hw.scratch_dir.is_dir()
    # Regardless of device, summary must be printable.
    assert isinstance(hw.summary(), str)


def test_autoscale_cpu_is_conservative(tmp_path):
    hw = Hardware(
        device="cpu", device_name="Mac",
        gpu_count=0, gpu_vram_gib=0.0,
        cpu_count=8, total_ram_gib=16.0,
        scratch_dir=tmp_path, scratch_free_gib=100.0,
        supports_bf16=False,
    )
    s = autoscale_defaults(hw)
    assert s.torch_device == "cpu"
    assert s.amp_dtype == "float32"
    assert s.batch_size <= 8
    assert 1 <= s.num_workers <= 8
    assert s.lgbm_threads == 7


def test_autoscale_mi300x_is_aggressive(tmp_path):
    hw = Hardware(
        device="rocm", device_name="AMD Instinct MI300X",
        gpu_count=1, gpu_vram_gib=192.0,
        cpu_count=240, total_ram_gib=1800.0,
        scratch_dir=tmp_path, scratch_free_gib=5000.0,
        supports_bf16=True,
    )
    s = autoscale_defaults(hw)
    assert s.amp_dtype == "bfloat16"
    assert s.batch_size >= 64
    assert s.lgbm_threads >= 200
    assert s.preprocess_workers >= 200
    assert s.pin_memory is True
    assert s.persistent_workers is True
