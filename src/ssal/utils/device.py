from __future__ import annotations

import torch


def get_device(device_name: str = "auto") -> torch.device:
    """Select the requested computation device."""

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        return torch.device("cuda")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

        return torch.device("mps")

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {device_name}. "
        "Expected one of: auto, cpu, cuda, mps."
    )
