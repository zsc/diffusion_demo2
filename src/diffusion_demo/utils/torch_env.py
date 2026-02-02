"""PyTorch environment utilities: device, dtype, seed."""

import os
import random
from typing import Tuple

import numpy as np
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Get torch device based on preference.
    
    Args:
        preference: "auto", "cuda", "cpu", or specific device string
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def is_bf16_supported() -> bool:
    """Check if bfloat16 is supported on current device."""
    if not torch.cuda.is_available():
        return False
    # Check GPU capability - bfloat16 requires SM80+ (Ampere, Ada, Hopper)
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 8


def get_dtype(use_bfloat16: bool = False) -> torch.dtype:
    """Get appropriate dtype based on availability and preference."""
    if use_bfloat16 and is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_info() -> dict:
    """Get PyTorch environment info."""
    info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": is_bf16_supported(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    return info


def get_env_config() -> dict:
    """Get environment configuration from env vars."""
    return {
        "port": int(os.environ.get("PORT", 5000)),
        "host": os.environ.get("HOST", "127.0.0.1"),
        "device": os.environ.get("DEVICE", "auto"),
        "data_limit_default": int(os.environ.get("DATA_LIMIT_DEFAULT", 2048)),
    }
