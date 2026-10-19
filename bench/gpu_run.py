"""A1 run on one GPU: correctness of every kernel version, then the measured ladder and the prefill crossover.

    python bench/gpu_run.py --out results/a1.json [--quick]

Exits non-zero if any correctness check fails, so a broken kernel never produces a results table.
"""
import argparse
import hashlib
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
SOURCES = ("lowbit/quant.py", "lowbit/ops.py", "lowbit/model.py", "csrc/w4_gemv.cu", "bench/gpu_run.py", "bench/e2e.py")
# kernel variants: name -> (version, rows_per_warp); v4 picks R from N, the v4_r* rows show what each R would give
KERNELS = {"v1": (1, 0), "v2": (2, 0), "v3": (3, 0), "v4": (4, 0), "v4_r1": (4, 1), "v4_r2": (4, 2), "v4_r4": (4, 4),
           "v5": (5, 0), "v5_t1": (5, 1), "v5_t2": (5, 2), "v5_t4": (5, 4), "v6": (6, 0)}      # v5_tN: N 16-row tiles per warp
TC_ROWS = (12, 16, 33, 64)     # beyond the GEMV range: only the tensor-core kernel (v5) takes these
L2_FLUSH_BYTES = 64 << 20      # every timed graph cycles through >= 64 MiB of copies of what it reads: 16x the T4's 4 MiB L2


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
    root = pathlib.Path(__file__).resolve().parent.parent
    src = hashlib.sha256()
    for rel in SOURCES:                       # the measured code itself, whatever the commit label says
        src.update(rel.encode() + b"\0" + (root / rel).read_bytes())
    vers = {}
    for mod in ("transformers", "datasets", "numpy"):
        try:
            vers[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return {"gpu": p.name, "sm": f"{p.major}.{p.minor}", "sms": p.multi_processor_count, "nvidia_smi": smi,
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(), **vers,
            "commit": commit, "source_sha256": src.hexdigest()[:16],
            "dispatch": {"gemv_rows": ops.AUTO_GEMV_ROWS, "v5_rows": ops.AUTO_V5_ROWS, "v5_wide_rows": ops.AUTO_V5_WIDE_ROWS,
                         "wide_n": ops.AUTO_WIDE_N},
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


_HEAT = None


def heat(ms=60):
    """Keep the GPU busy for about `ms` so its clocks are at their loaded level before a measurement. A T4 left idle
    for a moment drops its SM clock, and a small layer's time then moves by up to 1.7x (measured: the same v6 call
    5.4 vs 9.2 us depending on what ran before it)."""
    global _HEAT
    if _HEAT is None:
        _HEAT = torch.randn(2048, 2048, device="cuda", dtype=torch.half)
    t0 = time.time()
    while time.time() - t0 < ms / 1000:
        for _ in range(4):
            _HEAT @ _HEAT
        torch.cuda.synchronize()


def sm_clock_mhz():
    try:
        return int(torch.cuda.clock_rate())
    except Exception:
        return None


def timed(fn, warmup=20, iters=100, rounds=5):
    """Per-call GPU time in microseconds: each of `rounds` replays of a CUDA graph holding `iters` calls gives one
    mean; we report the median of those means and their spread (`p10`/`p90` across the rounds, not per-call
    percentiles). The graph removes Python and launch overhead (several microseconds per call), which would
    otherwise hide the kernel on small layers; fall back to plain launches if the call cannot be captured."""
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
    heat()
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
            "timing": mode, "sm_mhz": sm_clock_mhz()}


def bandwidth():
    """Achievable DRAM bandwidth, two ways: a 256 MiB device-to-device copy (read + write every byte) and a
    256 MiB read-only reduction. A GEMV mostly reads, so the read figure is its ceiling."""
    src = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t = timed(lambda: dst.copy_(src), warmup=5, iters=20, rounds=5)
    f = src.view(torch.float32).normal_()
    r = timed(lambda: f.sum(), warmup=5, iters=20, rounds=5)
    return {"copy_gbps": 2 * src.numel() / (t["us"] * 1e-6) / 1e9, "read_gbps": src.numel() / (r["us"] * 1e-6) / 1e9}


def overflow_case():
    """Large activations against small weights: q = 7, scale = 2^-10, x = 4096 over one group of 128 gives exactly
    3584. Scaling after an fp16 partial sum of 7 * x overflowed this to inf (v3/v4/v6 before the fix)."""
    k = 128
    wq = ops.W4Weight(np.full((16, k // 2), 0xFF, np.uint8), np.full((16, 1), 2.0 ** -10, np.float16), k)
    out, ok = {}, True
    for xv, want in ((4096.0, 3584.0), (-8192.0, -7168.0)):
        x = torch.full((1, k), xv, dtype=torch.half, device="cuda")
        for kname, (v, rpw) in KERNELS.items():
            y = ops.gemv(x, wq, v, rpw).float()
            good = bool(torch.isfinite(y).all()) and float((y - want).abs().max()) <= 2.0    # fp16 output step at 4096 is 4
            out[f"{kname}@{xv:g}"] = float(y[0, 0])
            ok &= good
            if not good:
                print(f"OVERFLOW {kname} x={xv}: {y[0, 0].item()} (want {want})", flush=True)
    return ok, out


def dispatch_matrix():
    """linear() as a model calls it, against the fp64 reference: the mma-only copy a model keeps across the auto
    rule's row boundaries, G = 32/64 weights (no mma layout: v4 then dequantize), G = 256 on the mma layout, a
    fixed version past its row limit (must fall back, not fail), a side stream, and inputs it must refuse."""
    rows, ok = [], True
    cases = [("mma_only_g128", 1024, 3072, 128, ("mma",)), ("mma_only_narrow", 1536, 1000, 128, ("mma",)),
             ("canonical_g32", 96, 200, 32, None), ("canonical_g64", 320, 130, 64, None), ("mma_g256", 512, 4096, 256, ("mma",))]
    for name, k, n, g, layouts in cases:
        gen = torch.Generator().manual_seed(k + n + g)
        wf = torch.randn(n, k, generator=gen)
        wq = ops.W4Weight(*quant.quantize_w4(wf.numpy(), g), g, layouts)
        ref_w = torch.from_numpy(quant.dequantize_w4(*quant.quantize_w4(wf.numpy(), g), g)).double().cuda()
        for m in (1, 2, 8, 9, 17, 31, 32, 33, 63, 64, 65, 130):
            x = torch.randn(m, k, generator=gen).half().cuda()
            ref = x.double() @ ref_w.T
            e = ((ops.linear(x, wq).double() - ref).abs().max() / ref.pow(2).mean().sqrt()).item()
            rows.append({"case": name, "M": m, "path": ops.auto_version(m, wq) or "dequant+cublas", "rel": e})
            ok &= e <= 3e-3
    wq = ops.W4Weight.from_float(torch.randn(256, 512))
    x = torch.randn(9, 512).half().cuda()
    ok &= bool(torch.isfinite(ops.linear(x, wq, max_gemv_rows=16, version=6)).all())     # v6 takes 8 rows: falls back
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        y_side = ops.linear(x[:1], wq)
    s.synchronize()
    ok &= bool(torch.equal(y_side, ops.linear(x[:1], wq)))
    refused = 0
    for bad in (lambda: ops.ext().w4_dequant(wq.packed, wq.scales.cpu(), 128),            # CPU scales
                lambda: ops.ext().w4_gemv(x[:1], wq.packed, wq.scales, 0, 1, 0),           # group 0
                lambda: ops.ext().w4_gemv(x, wq.packed, wq.scales, 128, 1, 0)):            # 9 rows into a GEMV
        try:
            bad()
        except RuntimeError:
            refused += 1
    ok &= refused == 3
    print("dispatch", "PASS" if ok else "FAIL", f"worst {max(r['rel'] for r in rows):.2e}, refused {refused}/3", flush=True)
    return ok, rows


def check(shapes, rows):
    """Each version vs an fp64 reference on the same quantized weights; cuBLAS FP16 on the dequantized weights is
    the yardstick for acceptable fp16 rounding. A kernel bug gives errors orders of magnitude above it."""
    out, ok = [], True
    for name, k, n in shapes:
        g = torch.Generator(device="cpu").manual_seed(k * 7 + n)
        wf = torch.randn(n, k, generator=g)
        wq = ops.W4Weight.from_float(wf, GROUP)
        # the GPU quantizer (used for whole models) must give the golden model's bits
        pk, sc = ops.quantize_w4_gpu(wf.cuda(), GROUP)
        same = {"gpu_quant": bool(torch.equal(pk, wq.packed) and torch.equal(sc, wq.scales)),
                "gpu_dequant": bool(torch.equal(ops.dequantize_w4_gpu(pk, sc, GROUP), ops.dequant(wq)))}
        for key in ("gpu_quant", "gpu_dequant"):
            if not same[key]:
                print(f"MISMATCH {key} on {name}", flush=True)
                ok = False
        w16 = ops.dequant(wq)
        ref_w = torch.from_numpy(quant.dequantize_w4(wq.packed.cpu().numpy(), wq.scales.cpu().numpy(), GROUP)).double().cuda()
        dq_err = (w16.double() - ref_w).abs().max().item()
        # q * scale is exact in fp32; the kernel rounds it to fp16 once (round to nearest), so it must equal the
        # fp64 reference rounded to fp16 bit for bit
        if not torch.equal(w16, ref_w.half()):
            print(f"MISMATCH dequant_kernel on {name}", flush=True)
            ok = False
        # the model keeps only the v5 layout; its dequantize kernel must give the same fp16 weights
        if not torch.equal(ops.ext().w4_dequant(wq.packed_mma, wq.scales, GROUP, True), w16):
            print(f"MISMATCH dequant_mma_kernel on {name}", flush=True)
            ok = False
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
                if m > 8 and v != 5:                      # v5 alone goes past 8 rows
                    continue
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
        w4_bytes, w16_bytes = n * k // 2 + n * (k // GROUP) * 2, n * k * 2
        c4, c16 = (max(2, -(-L2_FLUSH_BYTES // b)) for b in (w4_bytes, w16_bytes))
        wqs = [ops.W4Weight.from_float(torch.randn(n, k), GROUP) for _ in range(c4)]
        w16s = [ops.dequant(wq) for wq in wqs[:c16]]
        # pick() runs while the graph is captured, so the graph holds one call per copy: capture whole rotations
        nxt = iter(range(1 << 62))
        pick = lambda seq: seq[next(nxt) % len(seq)]
        it4, it16 = max(100, c4), max(100, c16)
        for m in sorted(set(rows_decode) | set(rows_prefill)):
            x = torch.randn(m, k).half().cuda()
            r = {"shape": name, "K": k, "N": n, "M": m, "copies_int4": c4, "copies_fp16": c16,
                 "graph_working_set_mib": {"int4": c4 * w4_bytes / 2**20, "fp16": c16 * w16_bytes / 2**20}}
            act = 2 * (m * k + m * n)
            r["fp16_roofline_us"] = (w16_bytes + act) / (peak * 1e9) * 1e6
            r["int4_roofline_us"] = (w4_bytes + act) / (peak * 1e9) * 1e6
            r["fp16_cublas"] = timed(lambda: torch.nn.functional.linear(x, pick(w16s)), iters=it16)
            # the fallback a model runs: it keeps only the mma copy, so it dequantizes with dequant_mma_kernel
            r["dequant_cublas"] = timed(lambda: torch.nn.functional.linear(x, ops.dequant(pick(wqs), "mma")), iters=it4)
            r["dequant_cublas_canonical"] = timed(lambda: torch.nn.functional.linear(x, ops.dequant(pick(wqs), "canonical")), iters=it4)
            r["dequant_only"] = timed(lambda: ops.dequant(pick(wqs), "mma"), iters=it4)
            r["linear"] = timed(lambda: ops.linear(x, pick(wqs)), iters=it4)      # the deployed path (auto rule)
            r["linear_path"] = ops.auto_version(m, wqs[0]) or "dequant+cublas"
            for kname, (v, rpw) in KERNELS.items():
                if m <= 8 or (v == 5 and m <= 64):
                    r[kname] = timed(lambda v=v, rpw=rpw: ops.gemv(x, pick(wqs), v, rpw), iters=it4)
                    if v in (4, 6) and rpw == 0:
                        r[f"{kname}_R"] = ops.pick_rows(n, m)
            if 64 < m <= 512:
                r["v5_chunked"] = timed(lambda: ops.gemm_chunked(x, pick(wqs)), iters=it4)
            # effective bandwidth = the minimum bytes the operation must move / its time (not a DRAM counter: an
            # intermediate that stays in L2 makes this exceed the DRAM ceiling; bench/ncu_dram.py counts DRAM bytes)
            for key in [kk for kk in r if isinstance(r[kk], dict) and "us" in r[kk]]:
                moved = (w16_bytes if key == "fp16_cublas" else w4_bytes) + act
                if key.startswith("dequant_cublas") or (key == "linear" and r["linear_path"] == "dequant+cublas"):
                    moved = w4_bytes + 2 * w16_bytes + act    # dequant writes fp16 W, the GEMM reads it back
                if key == "v5_chunked":
                    moved = -(-m // 64) * w4_bytes + act      # each 64-row slice reads the INT4 weight again
                if key == "dequant_only":
                    moved = w4_bytes + w16_bytes
                r[key]["eff_gbps"] = moved / (r[key]["us"] * 1e-6) / 1e9
                r[key]["eff_pct_bw"] = 100 * r[key]["eff_gbps"] / peak
            res.append(r)
            print(f"{name:18s} M={m:<4d} " + "  ".join(f"{kk}={r[kk]['us']:.1f}us" for kk in r if isinstance(r[kk], dict) and "us" in r[kk]),
                  flush=True)
        wqs = w16s = None
        torch.cuda.empty_cache()
    return res


def order_check(peak_unused=None, names=("qwen0.5b.q_proj", "qwen0.5b.down", "qwen1.5b.q_proj", "qwen3b.gate_up")):
    """M = 1: the same v6 call timed directly, through linear() and through the raw extension, interleaved three
    times, so a gap between them is a real cost and not GPU clock state left by the timing before it."""
    out = []
    for name, k, n in [s for s in SHAPES if s[0] in names]:
        w4_bytes = n * k // 2 + n * (k // GROUP) * 2
        wqs = [ops.W4Weight.from_float(torch.randn(n, k), GROUP, ("mma",)) for _ in range(max(2, -(-L2_FLUSH_BYTES // w4_bytes)))]
        nxt = iter(range(1 << 62))
        pick = lambda: wqs[next(nxt) % len(wqs)]
        x = torch.randn(1, k).half().cuda()
        it = max(100, len(wqs))
        calls = {"gemv_v6": lambda: ops.gemv(x, pick(), 6), "linear": lambda: ops.linear(x, pick()),
                 "ext_raw": lambda: (lambda w: ops.ext().w4_gemv(x, w.packed_mma, w.scales, w.group, 6, 0))(pick())}
        seq = [(rep, key, timed(fn, iters=it)["us"]) for rep in range(3) for key, fn in calls.items()]
        out.append({"shape": name, "K": k, "N": n, "sequence": seq})
        print(name, " ".join(f"{key}={us:.1f}" for _, key, us in seq), flush=True)
        wqs = None
        torch.cuda.empty_cache()
    return out


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
    ok, checks = check(shapes + EDGE_SHAPES, list(range(1, 9)) + list(TC_ROWS))
    ok_over, over = overflow_case()
    ok_disp, disp = dispatch_matrix()
    print("overflow", "PASS" if ok_over else "FAIL", flush=True)
    ok = ok and ok_over and ok_disp
    worst = {kname: max(c[f"{kname}_rel"] for c in checks if f"{kname}_rel" in c) for kname in KERNELS}
    worst["cublas"] = max(c["cublas_rel"] for c in checks)
    print("correctness", "PASS" if ok else "FAIL", worst, flush=True)
    result = {"env": env(), "build_seconds": build_s, "group": GROUP, "correct": ok, "worst_rel_err": worst,
              "checks": checks, "overflow_case": over, "dispatch_matrix": disp}
    if ok:
        bw = bandwidth()
        peak = bw["read_gbps"]
        result["bandwidth"] = bw
        result["spec_bandwidth_gbps"] = 320.0 if "T4" in result["env"]["gpu"] else None
        print(f"bandwidth {bw}", flush=True)
        result["order_check"] = order_check()
        result["bench"] = bench(shapes, [1, 2, 4, 8], [16, 32, 64] if a.quick else [16, 32, 64, 65, 96, 128, 192, 256, 384, 512, 1024], peak)
        result["order_check_after"] = order_check()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(result, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
