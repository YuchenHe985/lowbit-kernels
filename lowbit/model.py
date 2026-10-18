"""Swap a Hugging Face model's linear layers for INT4 weight-only ones (decode: GEMV kernel; prefill: dequant + cuBLAS)."""
import torch
from torch import nn

from . import ops

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class W4Linear(nn.Module):
    def __init__(self, lin: nn.Linear, group=128):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.wq = ops.W4Weight.from_float(lin.weight, group)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().half(), requires_grad=False)

    def forward(self, x):
        shape = x.shape
        y = ops.linear(x.reshape(-1, self.in_features).half(), self.wq)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)


def quantize_model(model, group=128):
    """Replace every decoder-layer projection in place; embeddings and lm_head stay FP16 (they are tied in Qwen2.5
    up to 3B, so quantizing lm_head would also change the embedding). Returns (layers replaced, int4 bytes)."""
    count, nbytes = 0, 0
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in TARGETS:
                q = W4Linear(child, group)
                setattr(module, child_name, q)
                count += 1
                nbytes += q.wq.nbytes()
                del child
    torch.cuda.empty_cache()
    return count, nbytes
