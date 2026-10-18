"""Render the measured results as Markdown tables (the README's numbers come only from here).

    python bench/report.py results/a1.json [results/e2e_*.json] > results/RESULTS.md
"""
import json
import sys


def fmt_env(e):
    return (f"{e['gpu']} (sm_{e['sm'].replace('.', '')}, {e['sms']} SMs), torch {e['torch']}, CUDA {e['cuda']}, "
            f"commit `{e['commit']}`, {e['time_utc']}")


def best_r(x):
    return min((1, 2, 4), key=lambda r: x[f"v4_r{r}"]["us"])


def a1(r):
    bw, w = r["bandwidth"], r["worst_rel_err"]
    out = ["## A1: kernels", "", f"Environment: {fmt_env(r['env'])}", "",
           f"Measured DRAM bandwidth: **{bw['read_gbps']:.0f} GB/s read-only**, {bw['copy_gbps']:.0f} GB/s copy "
           f"(spec {r['spec_bandwidth_gbps']:.0f} GB/s). Rooflines and `% of BW` use the read figure. Weights are "
           "read cold on every call (copies cycled past the 4 MiB L2), as in a real decode step.", "",
           "Correctness: every kernel against an fp64 reference on the same quantized weights, M = 1..8, all shapes "
           "plus odd-N tails (v5 also at M = 12..64). Worst max-abs error / RMS(output): " +
           ", ".join(f"{k} {v:.2e}" for k, v in w.items() if not k.startswith(("v4_", "v5_"))) + ".", ""]
    b = r["bench"]
    out += ["### Decode (M = 1): median latency, microseconds", "",
            "| layer | K x N | FP16 cuBLAS | FP16 roofline | dequant + cuBLAS | v1 | v2 | v3 | v4 (R) | INT4 roofline | "
            "v4 vs FP16 cuBLAS | v4 vs FP16 roofline | v4 % of BW |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in b:
        if x["M"] != 1:
            continue
        out.append(f"| {x['shape']} | {x['K']} x {x['N']} | {x['fp16_cublas']['us']:.1f} | {x['fp16_roofline_us']:.1f} | "
                   f"{x['dequant_cublas']['us']:.1f} | {x['v1']['us']:.1f} | {x['v2']['us']:.1f} | {x['v3']['us']:.1f} | "
                   f"{x['v4']['us']:.1f} ({best_r(x)}) | {x['int4_roofline_us']:.1f} | "
                   f"{x['fp16_cublas']['us'] / x['v4']['us']:.2f}x | {x['fp16_roofline_us'] / x['v4']['us']:.2f}x | "
                   f"{x['v4']['pct_peak']:.0f}% |")
    out += ["", "(R) = the fastest rows-per-warp among the v4_r1/r2/r4 runs; `v4` itself uses the R picked from N.", ""]
    ms = sorted({x["M"] for x in b if x["M"] <= 64})
    out += ["### Batch sweep: fastest INT4 kernel vs FP16 cuBLAS, microseconds", "",
            "Each cell: best INT4 time (which kernel) / FP16 cuBLAS time. v1-v4 run up to M = 8, the tensor-core v5 up "
            "to M = 64; v5 picks its row tiles (T) from N and M, the v5_tN runs fix them.", "",
            "| layer | " + " | ".join(f"M={m}" for m in ms) + " |", "|---" * (1 + len(ms)) + "|"]
    for name in dict.fromkeys(x["shape"] for x in b):
        cells = []
        for m in ms:
            x = next((y for y in b if y["shape"] == name and y["M"] == m), None)
            if not x:
                cells.append("-")
                continue
            k, us = best_int4(x)
            win = "**" if us < x["fp16_cublas"]["us"] else ""
            cells.append(f"{win}{us:.1f}{win} ({k}) / {x['fp16_cublas']['us']:.1f}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += ["", "Bold: the INT4 kernel beats FP16 cuBLAS at that batch size.", ""]
    out += ["### v4 / v6 / v5 by batch size (picked defaults), microseconds", "",
            "| layer | " + " | ".join(f"M={m}" for m in ms if m <= 8) + " |", "|---" * (1 + len([m for m in ms if m <= 8])) + "|"]
    for name in dict.fromkeys(x["shape"] for x in b):
        cells = []
        for m in (m for m in ms if m <= 8):
            x = next((y for y in b if y["shape"] == name and y["M"] == m), None)
            cells.append(" / ".join(f"{x[k]['us']:.1f}" if k in x else "-" for k in ("v4", "v6", "v5")) if x else "-")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    pre = [x for x in b if x["M"] > 64]
    if pre:
        pm = sorted({x["M"] for x in pre})
        out += ["", "### Prefill (M > 64): dequant + cuBLAS time / FP16 cuBLAS time", "",
                "| layer | " + " | ".join(f"M={m}" for m in pm) + " |", "|---" * (1 + len(pm)) + "|"]
        for name in dict.fromkeys(x["shape"] for x in pre):
            row = sorted((x for x in pre if x["shape"] == name), key=lambda x: x["M"])
            out.append(f"| {name} | " + " | ".join(
                f"{x['dequant_cublas']['us'] / x['fp16_cublas']['us']:.2f}x" for x in row) + " |")
    return out


KERNELS = ("v2", "v3", "v4", "v4_r1", "v4_r2", "v4_r4", "v5", "v5_t1", "v5_t2", "v5_t4", "v6")


def best_int4(x):
    k = min((k for k in KERNELS if k in x), key=lambda k: x[k]["us"])
    return k, x[k]["us"]


def e2e(rs):
    out = ["## A2: end to end (Hugging Face, batch 1, greedy)", "", f"Environment: {fmt_env(rs[0]['env'])}", "",
           "Quality on WikiText-2 (40 windows x 1024 tokens). Fake-quant = FP16 layers holding exactly the dequantized "
           "INT4 weights: its gap to FP16 is the cost of 4-bit weights, its gap to INT4 is kernel error.", "",
           "| model | scales | weights FP16 -> INT4 | PPL FP16 | PPL fake-quant | PPL INT4 | next-token agreement with FP16 | "
           "INT4 vs fake-quant agreement |", "|---|---|---|---|---|---|---|---|"]
    for r in rs:
        f, z, q = r["fp16"], r["fake_quant"], r["int4"]
        out.append(f"| {name(r)} | {'clip search' if r.get('clip_search') else 'absmax'} | "
                   f"{f['weight_bytes'] / 2**30:.2f} -> {q['weight_bytes'] / 2**30:.2f} GiB | {f['wikitext2']['ppl']:.3f} | "
                   f"{z['wikitext2']['ppl']:.3f} | {q['wikitext2']['ppl']:.3f} | "
                   f"{100 * q['wikitext2']['next_token_agreement_with_fp16']:.1f}% | "
                   f"{100 * r['int4_vs_fake_quant_next_token_agreement']:.2f}% |")
    out += ["", "Speed at batch 1 (decode tokens/s; TTFT in ms). Eager = plain `generate`; CUDA graph = each decode step "
            "replayed as one graph over a static KV cache.", "",
            "| model | scales | eager tok/s FP16 -> INT4 | CUDA graph tok/s FP16 -> INT4 | TTFT ms FP16 -> INT4 | peak mem GB |",
            "|---|---|---|---|---|---|"]
    for r in rs:
        f, q = r["fp16"], r["int4"]
        g = (f"{f['cuda_graph']['decode_tokens_per_s']:.1f} -> {q['cuda_graph']['decode_tokens_per_s']:.1f} "
             f"({r['speedup_decode_cuda_graph']:.2f}x)" if "speedup_decode_cuda_graph" in r else
             "failed: " + str(q.get("cuda_graph", f.get("cuda_graph", {})).get("error", "?"))[:60])
        out.append(f"| {name(r)} | {'clip' if r.get('clip_search') else 'absmax'} | "
                   f"{f['eager']['decode_tokens_per_s']:.1f} -> {q['eager']['decode_tokens_per_s']:.1f} "
                   f"({r['speedup_decode_eager']:.2f}x) | {g} | {f['eager']['ttft_ms']:.0f} -> {q['eager']['ttft_ms']:.0f} | "
                   f"{f['peak_mem_bytes'] / 1e9:.2f} -> {q['peak_mem_bytes'] / 1e9:.2f} |")
    out += ["", "Where eager decode GPU time goes (profiler, 32 steps):", "",
            "| model | run | GPU busy | INT4 kernels | other GEMM | dequant | other |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        for lab in ("fp16", "int4"):
            p = r[lab]["profile"]
            s = p.get("share_pct", {})
            out.append(f"| {name(r)} | {lab} | {p['gpu_busy_pct']:.0f}% | {s.get('int4_kernels', 0):.0f}% | "
                       f"{s.get('gemm', 0):.0f}% | {s.get('dequant', 0):.0f}% | {s.get('other', 0):.0f}% |")
    return out


def name(r):
    return r["model"].split("/")[-1]


def main():
    docs = [json.load(open(p)) for p in sys.argv[1:]]
    lines = []
    for d in docs:
        if "bench" in d:
            lines += a1(d) + [""]
    e = [d for d in docs if "fake_quant" in d]
    if e:
        lines += e2e(e)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
