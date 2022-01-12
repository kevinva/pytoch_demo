import torch
from torch import nn, Tensor
from typing import Tuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BPTT = 35

def batchify(data: Tensor, bsz: int) -> Tensor:
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(DEVICE)

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(BPTT, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target