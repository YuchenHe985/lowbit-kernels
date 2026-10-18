"""Swap a Hugging Face model's linear layers for INT4 weight-only ones (decode: GEMV kernel; prefill: dequant + cuBLAS).

Each layer is quantized once (quantize_layers); the same packed values then back either a "fake-quant" model
(ordinary nn.Linear holding the dequantized FP16 weights, so PyTorch does the math) or the INT4 kernels. Equal
quality from the two separates kernel error from quantization error.
"""
import torch
from torch import nn

from . import ops

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class W4Linear(nn.Module):
    def __init__(self, wq, bias, in_features, out_features, gemv_version=0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.wq, self.gemv_version = wq, gemv_version
        self.bias = None if bias is None else nn.Parameter(bias.detach().half(), requires_grad=False)

    def forward(self, x):
        shape = x.shape
        y = ops.linear(x.reshape(-1, self.in_features).half(), self.wq, version=self.gemv_version)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)


def quantize_layers(model, group=128, clip_grid=None):
    """Quantize every decoder-layer projection once. Embeddings and lm_head stay FP16 (tied in Qwen2.5 up to 3B, so
    quantizing lm_head would also change the embedding). Returns [(parent, name, linear, packed, scales)]."""
    out = []
    for _, module in list(model.named_modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in TARGETS:
                packed, scales = ops.quantize_w4_gpu(child.weight, group, clip_grid)
                out.append((module, name, child, packed, scales))
    return out


def apply_fake_quant(layers, group=128):
    """Replace each weight by its dequantized FP16 value (exactly what the kernels multiply by)."""
    for _, _, lin, packed, scales in layers:
        lin.weight.data.copy_(ops.dequantize_w4_gpu(packed, scales, group))


def apply_int4(layers, group=128, gemv_version=0, layouts=("mma",)):
    """Swap each nn.Linear for a W4Linear on the same packed values; returns (layers replaced, INT4 bytes)."""
    count, nbytes = len(layers), 0
    for parent, name, lin, packed, scales in layers:
        wq = ops.W4Weight(packed.cpu().numpy(), scales.cpu().numpy(), group, layouts)
        setattr(parent, name, W4Linear(wq, lin.bias, lin.in_features, lin.out_features, gemv_version))
        nbytes += wq.nbytes()
    layers.clear()                      # drops the last references to the FP16 nn.Linear weights
    torch.cuda.empty_cache()
    return count, nbytes
