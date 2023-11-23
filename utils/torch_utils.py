from typing import Optional, Sequence
import torch


def to_torch_size(*size) -> torch.Size:
    if len(size) == 1 and isinstance(size[0], Sequence):
        torch_size = size[0]
    else:
        torch_size = list(size)
    return torch.Size(torch_size)
