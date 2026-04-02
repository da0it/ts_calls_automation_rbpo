from __future__ import annotations

import os
from functools import lru_cache

AUTO_DEVICE = "auto"


def normalize_whisperx_device(device: str | None) -> str:
    value = (device or "").strip().lower()
    if not value:
        return AUTO_DEVICE
    if value == "gpu":
        return "cuda"
    return value


@lru_cache(maxsize=1)
def _cuda_is_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_whisperx_device(device: str | None = None) -> str:
    normalized = normalize_whisperx_device(device)
    if normalized != AUTO_DEVICE:
        return normalized
    return "cuda" if _cuda_is_available() else "cpu"


def get_whisperx_device_from_env(name: str = "WHISPERX_DEVICE") -> str:
    return resolve_whisperx_device(os.getenv(name, AUTO_DEVICE))
