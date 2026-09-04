from __future__ import annotations

import re

import torch


CUDA_DEVICE_RE = re.compile(r"^cuda(?::(?P<index>\d+))?$")


def resolve_device(device: str = "auto", allow_cpu_fallback: bool = True) -> str:
    """Resolve a user device string using the project policy: cuda -> cpu."""
    normalized = (device or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "gpu":
        normalized = "cuda"
    if normalized == "cpu":
        return "cpu"
    if CUDA_DEVICE_RE.match(normalized):
        if torch.cuda.is_available():
            return normalized
        if allow_cpu_fallback:
            return "cpu"
        raise RuntimeError("CUDA was requested but is not available")
    raise ValueError(f"Unsupported device '{device}'. Use auto, cpu, cuda, or cuda:N.")


def validate_multi_gpu(enabled: bool, device_ids: list[int] | None) -> list[int]:
    """Validate multi-GPU ids and return the effective list."""
    if not enabled:
        return []
    ids = list(device_ids or range(torch.cuda.device_count()))
    if not ids:
        raise ValueError("multi-GPU is enabled but no CUDA device ids were provided or detected")
    available = torch.cuda.device_count()
    missing = [idx for idx in ids if idx < 0 or idx >= available]
    if missing:
        raise ValueError(
            f"unavailable CUDA device ids {missing}; detected {available} CUDA device(s)"
        )
    return ids


def pick_task_device(base_device: str, task_index: int, device_ids: list[int]) -> str:
    """Assign a task to a CUDA device round-robin when multi-GPU is enabled."""
    if not device_ids or not base_device.startswith("cuda"):
        return base_device
    return f"cuda:{device_ids[task_index % len(device_ids)]}"
