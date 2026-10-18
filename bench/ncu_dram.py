"""Workload for Nsight Compute's DRAM byte counters: one call each of FP16 cuBLAS, v4, v6, v5 and dequantize, on
a narrow and a wide layer, M = 1. ncu flushes the caches before each profiled kernel (--cache-control all), so
the counters show what a cold call moves from DRAM - the check on the effective-bandwidth figures of gpu_run.py.

    ncu --csv --cache-control all --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
        python bench/ncu_dram.py > results/ncu_dram.csv
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lowbit import ops  # noqa: E402

for k, n in ((896, 896), (2048, 11008)):
    wq = ops.W4Weight.from_float(torch.randn(n, k))
    w16 = ops.dequant(wq)
    x = torch.randn(1, k).half().cuda()
    torch.nn.functional.linear(x, w16)
    ops.gemv(x, wq, 4)
    ops.gemv(x, wq, 6)
    ops.gemv(x, wq, 5)
    ops.dequant(wq, "mma")
torch.cuda.synchronize()
print("ncu workload done", file=sys.stderr)
