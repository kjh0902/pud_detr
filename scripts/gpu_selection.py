"""Early CUDA visibility handling for the PUD-DETR training CLI."""

from __future__ import annotations

import os
from typing import Sequence


GENERIC_DEVICES = {"auto", "cpu", "cuda"}


def normalize_device_argument(device: str) -> str:
    """Return a canonical device value accepted by the training CLI."""
    value = device.strip().lower()
    if value in GENERIC_DEVICES:
        return value
    if value.isdigit():
        return str(int(value))
    if value.startswith("cuda:") and value[5:].isdigit():
        return str(int(value[5:]))
    raise ValueError("device must be auto, cpu, cuda, a GPU ID, or cuda:<GPU_ID>")


def concrete_cuda_index(device: str) -> int | None:
    normalized = normalize_device_argument(device)
    return int(normalized) if normalized.isdigit() else None


def device_argument_from_argv(argv: Sequence[str]) -> str | None:
    """Extract the last --device value without importing CUDA libraries."""
    result: str | None = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--device" and index + 1 < len(argv):
            result = argv[index + 1]
            index += 2
            continue
        if token.startswith("--device="):
            result = token.split("=", maxsplit=1)[1]
        index += 1
    return result


def configure_cuda_visibility(argv: Sequence[str]) -> int | None:
    """Expose only the requested physical GPU before PyTorch is imported."""
    value = device_argument_from_argv(argv)
    if value is None:
        return None
    try:
        physical_index = concrete_cuda_index(value)
    except ValueError:
        # The full argparse validation later provides the user-facing error.
        return None
    if physical_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    return physical_index
