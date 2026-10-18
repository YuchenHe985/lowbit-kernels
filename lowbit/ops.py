"""PyTorch side of the CUDA kernels: build the extension once, hold quantized weights, dispatch decode vs prefill."""
import pathlib

import torch

from . import quant

_EXT = None
SRC = pathlib.Path(__file__).resolve().parent.parent / "csrc" / "w4_gemv.cu"


def ext():
    """Compile csrc/w4_gemv.cu for the visible GPU on first use (about a minute), then reuse it."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="lowbit_w4", sources=[str(SRC)], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _EXT


class W4Weight:
    """One linear layer's weight in the lowbit/quant.py format, on the GPU, in both layouts."""

    def __init__(self, packed, scales, group):
        self.group = group
        self.n, self.k = packed.shape[0], packed.shape[1] * 2
        self.packed = torch.from_numpy(packed).cuda()                                   # canonical: v1, v2, dequant
        self.packed_v3 = torch.from_numpy(quant.interleave_for_gpu(packed)).cuda()      # interleaved: v3
        self.scales = torch.from_numpy(scales).cuda()

    @classmethod
    def from_float(cls, w, group=128):
        packed, scales = quant.quantize_w4(w.detach().float().cpu().numpy(), group)
        return cls(packed, scales, group)

    def nbytes(self):
        return self.packed.numel() + self.scales.numel() * 2


def gemv(x, wq, version=3):
    """x [M, K] fp16, M <= 8 -> [M, N] fp16."""
    w = wq.packed_v3 if version == 3 else wq.packed
    return ext().w4_gemv(x.contiguous(), w, wq.scales, wq.group, version)


def dequant(wq):
    return ext().w4_dequant(wq.packed, wq.scales, wq.group)


def linear(x, wq, max_gemv_rows=8):
    """Decode (few rows): the INT4 GEMV. Prefill (many rows): dequantize, then cuBLAS FP16 GEMM."""
    if x.shape[0] <= max_gemv_rows:
        return gemv(x, wq)
    return torch.nn.functional.linear(x, dequant(wq))
