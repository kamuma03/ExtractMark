"""GPU monitoring via nvidia-smi."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_gpu_memory_mb(gpu_index: int = 0) -> float | None:
    """Get current GPU memory usage in MB via nvidia-smi.

    Returns None if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        logger.debug("nvidia-smi query failed: %s", e)
    return None


def get_gpu_info() -> dict | None:
    """Get GPU name, total memory, and driver version."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                return {
                    "name": parts[0],
                    "memory_total_mb": float(parts[1]),
                    "driver_version": parts[2],
                }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        logger.debug("nvidia-smi info query failed: %s", e)
    return None
