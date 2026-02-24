"""CUDA memory management utilities."""

import gc
import torch


def clear_cuda_cache_for_current_process(sync: bool = True) -> None:
    """Clear CUDA cache for all visible devices in this process."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    if sync:
        torch.cuda.synchronize()
    for dev in range(torch.cuda.device_count()):
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
