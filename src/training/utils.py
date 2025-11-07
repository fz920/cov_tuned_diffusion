"""Training-time utilities such as gradient clipping queues."""

from __future__ import annotations

import numpy as np
import torch


class Queue:
    """Sliding window helper used to track gradient-norm statistics."""

    def __init__(self, max_len: int = 50):
        self.items: list[float] = []
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: float) -> None:
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self) -> float:
        return float(np.mean(self.items)) if self.items else 0.0

    def std(self) -> float:
        return float(np.std(self.items)) if self.items else 0.0


def gradient_clipping(flow: torch.nn.Module, gradnorm_queue: Queue) -> torch.Tensor:
    """Clip gradients relative to recent history and record the resulting norm."""
    max_grad_norm = 1.0 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=max_grad_norm, norm_type=2.0)

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    if float(grad_norm) > max_grad_norm:
        print(f"Clipped gradient with value {grad_norm:.1f} while allowed {max_grad_norm:.1f}")
    return grad_norm


__all__ = ["Queue", "gradient_clipping"]
