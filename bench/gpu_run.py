"""A1 run on one GPU: correctness of every kernel version, then the measured ladder and the prefill crossover.

    python bench/gpu_run.py --out results/a1.json [--quick]

Exits non-zero if any correctness check fails, so a broken kernel never produces a results table.
"""
import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lowbit import ops, quant  # noqa: E402

# (name, K, N): real linear-layer shapes. Qwen2.5 0.5B/1.5B/3B, plus a 7B-class MLP.
SHAPES = [
    ("qwen0.5b.q_proj", 896, 896), ("qwen0.5b.kv_proj", 896, 128), ("qwen0.5b.gate_up", 896, 4864),
    ("qwen0.5b.down", 4864, 896),
    ("qwen1.5b.q_proj", 1536, 1536), ("qwen1.5b.gate_up", 1536, 8960), ("qwen1.5b.down", 8960, 1536),
    ("qwen3b.q_proj", 2048, 2048), ("qwen3b.gate_up", 2048, 11008), ("qwen3b.down", 11008, 2048),
    ("7b-class.gate_up", 3584, 18944), ("7b-class.down", 18944, 3584),
]
EDGE_SHAPES = [("tiny", 128, 1), ("odd_n", 256, 1001), ("odd_n2", 384, 37)]   # tails: N not a multiple of the 4 rows per block
GROUP = 128
# kernel variants: name -> (version, rows_per_warp); v4 picks R from N, the v4_r* rows show what each R would give
KERNELS = {"v1": (1, 0), "v2": (2, 0), "v3": (3, 0), "v4": (4, 0), "v4_r1": (4, 1), "v4_r2": (4, 2), "v4_r4": (4, 4)}
L2_FLUSH_BYTES = 64 << 20      # cycle through enough weight copies that each call reads cold from DRAM (T4 L2: 4 MiB)


def env():
    p = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.memory,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    commit = os.environ.get("LOWBIT_COMMIT", "")
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return {"gpu": p.name, "sm": f"{p.major}.{p.minor}", "sms": p.multi_processor_count, "nvidia_smi": smi,
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "commit": commit, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def timed(fn, warmup=20, iters=100, rounds=5):
    """Median, p10, p90 of per-call GPU time in microseconds over `rounds` replays of a CUDA graph holding `iters`
    calls. The graph removes Python and launch overhead (several microseconds per call), which would otherwise hide
    the kernel on small layers; fall back to plain launches if the call cannot be captured."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        run, mode = g.replay, "cuda_graph"
    except RuntimeError:
        def run():
            for _ in range(iters):
                fn()
        mode = "eager"
    run()
    torch.cuda.synchronize()
    per = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        per.append(a.elapsed_time(b) * 1000 / iters)
    per.sort()
    return {"us": float(np.median(per)), "p10": float(np.percentile(per, 10)), "p90": float(np.percentile(per, 90)),
            "timing": mode}


def bandwidth():
    """Achievable DRAM bandwidth, two ways: a 256 MiB device-to-device copy (read + write every byte) and a
    256 MiB read-only reduction. A GEMV mostly reads, so the read figure is its ceiling."""
    src = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t = timed(lambda: dst.copy_(src), warmup=5, iters=20, rounds=5)
    f = src.view(torch.float32).normal_()
    r = timed(lambda: f.sum(), warmup=5, iters=20, rounds=5)
    return {"copy_gbps": 2 * src.numel() / (t["us"] * 1e-6) / 1e9, "read_gbps": src.numel() / (r["us"] * 1e-6) / 1e9}


def check(shapes, rows):
    """Each version vs an fp64 reference on the same quantized weights; cuBLAS FP16 on the dequantized weights is
    the yardstick for acceptable fp16 rounding. A kernel bug gives errors orders of magnitude above it."""
    out, ok = [], True
    for name, k, n in shapes:
        g = torch.Generator(device="cpu").manual_seed(k * 7 + n)
        wq = ops.W4Weight.from_float(torch.randn(n, k, generator=g), GROUP)
        w16 = ops.dequant(wq)
        ref_w = torch.from_numpy(quant.dequantize_w4(wq.packed.cpu().numpy(), wq.scales.cpu().numpy(), GROUP)).double().cuda()
        dq_err = (w16.double() - ref_w).abs().max().item()
        # q * scale is exact in fp32; the kernel rounds it to fp16 once (round to nearest), so it must equal the
        # fp64 reference rounded to fp16 bit for bit
        ok &= bool(torch.equal(w16, ref_w.half()))
        for m in rows:
            x = torch.randn(m, k, generator=g).half().cuda()
            ref = x.double() @ ref_w.T
            scale = ref.pow(2).mean().sqrt().item()
            cublas = (torch.nn.functional.linear(x, w16).double() - ref).abs().max().item() / scale
            # the output itself is fp16: rounding the exact result to fp16 already costs this much
            floor = (ref.half().double() - ref).abs().max().item() / scale
            row = {"shape": name, "K": k, "N": n, "M": m, "cublas_rel": cublas, "fp16_out_floor_rel": floor,
                   "dequant_max_abs_err": dq_err}
            # v3 keeps fp16 partials over 8 products (HFMA2), so it may sit a few times above cuBLAS's fp32
            # accumulation; a layout or indexing bug gives O(1) relative error, far beyond this bound
            for kname, (v, rpw) in KERNELS.items():
                e = (ops.gemv(x, wq, v, rpw).double() - ref).abs().max().item() / scale
                row[f"{kname}_rel"] = e
                ok &= e <= max(4 * cublas, 4 * floor, 3e-3)
            out.append(row)
    return ok, out


def bench(shapes, rows_decode, rows_prefill, peak):
    """Every call reads its weights cold: enough copies of the layer are cycled that they overflow L2, as in a
    real decode step, where the other layers' weights evict this one between uses."""
    res = []
    for name, k, n in shapes:
        copies = max(1, -(-L2_FLUSH_BYTES // (n * k * 2)))
        wqs = [ops.W4Weight.from_float(torch.randn(n, k), GROUP) for _ in range(copies)]
        w16s = [ops.dequant(wq) for wq in wqs]
        nxt = iter(range(1 << 62))
        pick = lambda seq: seq[next(nxt) % copies]
        for m in sorted(set(rows_decode) | set(rows_prefill)):
            x = torch.randn(m, k).half().cuda()
            r = {"shape": name, "K": k, "N": n, "M": m, "weight_copies": copies}
            act = 2 * (m * k + m * n)
            w4_bytes, w16_bytes = wqs[0].nbytes(), n * k * 2
            r["fp16_roofline_us"] = (w16_bytes + act) / (peak * 1e9) * 1e6
            r["int4_roofline_us"] = (w4_bytes + act) / (peak * 1e9) * 1e6
            r["fp16_cublas"] = timed(lambda: torch.nn.functional.linear(x, pick(w16s)))
            r["dequant_cublas"] = timed(lambda: torch.nn.functional.linear(x, ops.dequant(pick(wqs))))
            r["dequant_only"] = timed(lambda: ops.dequant(pick(wqs)))
            if m <= 8:
                for kname, (v, rpw) in KERNELS.items():
                    r[kname] = timed(lambda v=v, rpw=rpw: ops.gemv(x, pick(wqs), v, rpw))
            for key in [kk for kk in r if isinstance(r[kk], dict)]:
                moved = (w16_bytes if key == "fp16_cublas" else w4_bytes) + act
                if key == "dequant_cublas":
                    moved = w4_bytes + 2 * w16_bytes + act    # dequant writes fp16 W, the GEMM reads it back
                if key == "dequant_only":
                    moved = w4_bytes + w16_bytes
                r[key]["gbps"] = moved / (r[key]["us"] * 1e-6) / 1e9
                r[key]["pct_peak"] = 100 * r[key]["gbps"] / peak
            res.append(r)
            print(f"{name:18s} M={m:<4d} " + "  ".join(f"{kk}={r[kk]['us']:.1f}us" for kk in r if isinstance(r[kk], dict)),
                  flush=True)
        wqs = w16s = None
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/a1.json")
    ap.add_argument("--quick", action="store_true", help="four shapes (small, wide, long-K, 7B-class), decode only")
    a = ap.parse_args()
    torch.manual_seed(0)
    shapes = [SHAPES[i] for i in (0, 5, 9, 10)] if a.quick else SHAPES
    t0 = time.time()
    ops.ext()
    build_s = time.time() - t0
    ok, checks = check(shapes + EDGE_SHAPES, range(1, 9))
    worst = {kname: max(c[f"{kname}_rel"] for c in checks) for kname in KERNELS}
    worst["cublas"] = max(c["cublas_rel"] for c in checks)
    print("correctness", "PASS" if ok else "FAIL", worst, flush=True)
    result = {"env": env(), "build_seconds": build_s, "group": GROUP, "correct": ok, "worst_rel_err": worst,
              "checks": checks}
    if ok:
        bw = bandwidth()
        peak = bw["read_gbps"]
        result["bandwidth"] = bw
        result["spec_bandwidth_gbps"] = 320.0 if "T4" in result["env"]["gpu"] else None
        print(f"bandwidth {bw}", flush=True)
        result["bench"] = bench(shapes, [1, 2, 4, 8], [] if a.quick else [16, 32, 64, 128, 256, 512, 1024], peak)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(result, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
