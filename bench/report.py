"""Render the measured results as Markdown tables (the README's numbers come only from here).

    python bench/report.py results/raw/a1.json [results/raw/e2e_*.json] [results/raw/ncu_dram.csv] > results/RESULTS.md
"""
import csv
import io
import json
import sys


def fmt_env(e):
    extra = ", ".join(f"{k} {e[k]}" for k in ("transformers", "datasets") if k in e)
    return (f"{e['gpu']} (sm_{e['sm'].replace('.', '')}, {e['sms']} SMs), torch {e['torch']}, CUDA {e['cuda']}"
            f"{', ' + extra if extra else ''}, commit `{e['commit']}`, source sha256 `{e.get('source_sha256', '?')}`, {e['time_utc']}")


def spread(t):
    return f"{t['us']:.1f} ({t['p10']:.1f}-{t['p90']:.1f})"


KERNELS = ("v2", "v3", "v4", "v4_r1", "v4_r2", "v4_r4", "v5", "v5_t1", "v5_t2", "v5_t4", "v6")


def best_variant(x):
    k = min((k for k in KERNELS if k in x), key=lambda k: x[k]["us"])
    return k, x[k]["us"]


def a1(r):
    bw, w = r["bandwidth"], r["worst_rel_err"]
    b = r["bench"]
    ws = min(min(x["graph_working_set_mib"].values()) for x in b)
    out = ["## A1: kernels", "", f"Environment: {fmt_env(r['env'])}", "",
           f"Measured DRAM bandwidth: **{bw['read_gbps']:.0f} GB/s read-only**, {bw['copy_gbps']:.0f} GB/s copy "
           f"(spec {r['spec_bandwidth_gbps']:.0f} GB/s). Rooflines and `eff. % of BW` use the read figure.", "",
           "Timing: median of five CUDA-graph replays; each replay calls the operation once per copy of its operand, "
           f"cycling through at least {ws:.0f} MiB of distinct copies (T4 L2: 4 MiB), so weights are not reused from "
           "L2 between calls. Spread in parentheses = range of the five replay means (p10-p90), not per-call percentiles.", "",
           "Effective bandwidth = the minimum bytes an operation must move / its time. It is not a DRAM counter: an "
           "intermediate that stays in L2 (dequantize + cuBLAS at small sizes) can push it past the DRAM ceiling. "
           "The DRAM counters are in the Nsight Compute table below.", "",
           "Correctness: every kernel against an fp64 reference on the same quantized weights, M = 1..8, all shapes "
           "plus odd-N tails (v5 also at M = 12..64); each kernel's worst max-abs error / RMS(output) over all cases, "
           "next to cuBLAS's worst on the same cases (independent maxima, not a per-case comparison): " +
           ", ".join(f"{k} {v:.2e}" for k, v in w.items() if not k.startswith(("v4_", "v5_"))) + "."]
    if "overflow_case" in r:
        out += ["", "Overflow regression (q = 7, scale = 2^-10, x = +-4096/8192, exact output 3584 / -7168): every "
                "kernel finite and exact. Dispatch matrix (`linear()` on the mma-only copy a model keeps, G = 32/64/256, "
                f"M = 1..130 across every boundary of the auto rule): worst error "
                f"{max(d['rel'] for d in r['dispatch_matrix']):.2e}; bad inputs refused."]
    out += ["", "### Decode (M = 1): median latency, microseconds", "",
            "| layer | K x N | FP16 cuBLAS | v1 | v2 | v3 | v4 [R] | v6 [R] | `linear()` (deployed: v6) | INT4 roofline | "
            "`linear()` vs FP16 cuBLAS | v4 eff. % of BW |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in b:
        if x["M"] != 1:
            continue
        out.append(f"| {x['shape']} | {x['K']} x {x['N']} | {spread(x['fp16_cublas'])} | {x['v1']['us']:.1f} | "
                   f"{x['v2']['us']:.1f} | {x['v3']['us']:.1f} | {x['v4']['us']:.1f} [{x['v4_R']}] | "
                   f"{x['v6']['us']:.1f} [{x['v6_R']}] | {spread(x['linear'])} | {x['int4_roofline_us']:.1f} | "
                   f"{x['fp16_cublas']['us'] / x['linear']['us']:.2f}x | {x['v4']['eff_pct_bw']:.0f}% |")
    out += ["", "[R] = rows per warp the kernel picked from N.", ""]
    ms = sorted({x["M"] for x in b})
    out += ["### Deployed `linear()` vs FP16 cuBLAS by batch size", "",
            "Each cell: `linear()` time / FP16 cuBLAS time (path: v6, v5 or dequantize + cuBLAS, as the auto rule "
            "picks it on the mma-only copy). Bold: `linear()` is faster.", "",
            "| layer | " + " | ".join(f"M={m}" for m in ms) + " |", "|---" * (1 + len(ms)) + "|"]
    short = {6: "v6", 5: "v5", "dequant+cublas": "dq"}
    for name in dict.fromkeys(x["shape"] for x in b):
        cells = []
        for m in ms:
            x = next((y for y in b if y["shape"] == name and y["M"] == m), None)
            if not x:
                cells.append("-")
                continue
            ratio = x["linear"]["us"] / x["fp16_cublas"]["us"]
            win = "**" if ratio < 1 else ""
            cells.append(f"{win}{ratio:.2f}x{win} ({short[x['linear_path']]})")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    ms64 = [m for m in ms if m <= 64]
    out += ["", "### Every kernel variant (oracle: the fastest of all variants measured, not what `linear()` runs)", "",
            "Each cell: fastest variant time (which) / FP16 cuBLAS time, microseconds. v5_tN fix the row tiles; v4_rN "
            "fix rows per warp.", "",
            "| layer | " + " | ".join(f"M={m}" for m in ms64) + " |", "|---" * (1 + len(ms64)) + "|"]
    for name in dict.fromkeys(x["shape"] for x in b):
        cells = []
        for m in ms64:
            x = next((y for y in b if y["shape"] == name and y["M"] == m), None)
            if not x:
                cells.append("-")
                continue
            k, us = best_variant(x)
            cells.append(f"{us:.1f} ({k}) / {x['fp16_cublas']['us']:.1f}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    pre = [x for x in b if x["M"] > 64]
    if pre:
        pm = sorted({x["M"] for x in pre})
        out += ["", "### Prefill (M > 64): dequantize (model's mma copy) + cuBLAS / FP16 cuBLAS; canonical-copy "
                "dequantize in parentheses; 64-row v5 slices in brackets", "",
                "| layer | " + " | ".join(f"M={m}" for m in pm) + " |", "|---" * (1 + len(pm)) + "|"]
        for name in dict.fromkeys(x["shape"] for x in pre):
            row = sorted((x for x in pre if x["shape"] == name), key=lambda x: x["M"])
            f = lambda x, k: x[k]["us"] / x["fp16_cublas"]["us"]
            out.append(f"| {name} | " + " | ".join(
                f"{f(x, 'dequant_cublas'):.2f}x ({f(x, 'dequant_cublas_canonical'):.2f}x)"
                + (f" [{f(x, 'v5_chunked'):.2f}x]" if "v5_chunked" in x else "") for x in row) + " |")
    return out


def ncu(text):
    body = text.split("\n# stderr")[0]
    rows = [r for r in csv.reader(io.StringIO(body)) if len(r) > 5]
    if not rows or "Metric Name" not in rows[0]:
        tail = text.split("# stderr")[-1].strip().splitlines()[-3:]
        return ["", "### DRAM counters (Nsight Compute)", "", "Not available on this VM: " + " / ".join(tail)[:300]]
    h = rows[0]
    ki, mi, vi, ui = (h.index(c) for c in ("Kernel Name", "Metric Name", "Metric Value", "Metric Unit"))
    idx = h.index("ID")
    per = {}
    for r in rows[1:]:
        v = float(r[vi].replace(",", ""))
        scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "nsecond": 1e-3, "usecond": 1, "msecond": 1e3}.get(r[ui], 1)
        per.setdefault(int(r[idx]), {"kernel": r[ki][:60]})[r[mi]] = v * scale
    out = ["", "### DRAM counters (Nsight Compute, caches flushed before each kernel, M = 1)", "",
           "Narrow layer 896 x 896, then wide layer 2048 x 11008; per kernel in call order: FP16 cuBLAS, v4, v6, v5, "
           "dequantize.", "", "| # | kernel | DRAM read MB | DRAM write MB | time us (under ncu) |", "|---|---|---|---|---|"]
    for i, d in sorted(per.items()):
        out.append(f"| {i} | `{d['kernel']}` | {d.get('dram__bytes_read.sum', 0) / 1e6:.2f} | "
                   f"{d.get('dram__bytes_write.sum', 0) / 1e6:.2f} | {d.get('gpu__time_duration.sum', 0):.1f} |")
    return out


def e2e(rs):
    out = ["## A2: end to end (Hugging Face, batch 1, greedy)", "", f"Environment: {fmt_env(rs[0]['env'])}", "",
           "Models: " + ", ".join(f"{name(r)} @ `{(r.get('model_revision') or '?')[:10]}`" for r in rs[::2]) +
           f"; WikiText-2 fingerprint `{rs[0].get('dataset_fingerprint')}`.", "",
           "### Quality", "",
           "WikiText-2 perplexity (40 windows x 1024 tokens; these long inputs run the prefill path, dequantize + "
           "cuBLAS). Fake-quant = FP16 layers holding exactly the dequantized INT4 weights: its gap to FP16 is the cost "
           "of 4-bit weights.", "",
           "| model | scales | weights FP16 -> INT4 | PPL FP16 | PPL fake-quant | PPL INT4 | next-token agreement with FP16 |",
           "|---|---|---|---|---|---|---|"]
    for r in rs:
        f, z, q = r["fp16"], r["fake_quant"], r["int4"]
        out.append(f"| {name(r)} | {scales(r)} | {f['weight_bytes'] / 2**30:.2f} -> {q['weight_bytes'] / 2**30:.2f} GiB | "
                   f"{f['wikitext2']['ppl']:.3f} | {z['wikitext2']['ppl']:.3f} | {q['wikitext2']['ppl']:.3f} | "
                   f"{100 * q['wikitext2']['next_token_agreement_with_fp16']:.1f}% |")
    if "decode_path_int4_vs_fake_quant" in rs[0]:
        out += ["", "Decode kernels vs fake-quant on identical inputs: the same 512 WikiText tokens fed through the KV "
                "cache 1 token per step (INT4: v6), 16 per step (v5) and 48 per step (v5 on layers with N >= 3072, "
                "dequantize + cuBLAS on the rest). Logit difference over the first 32 positions, relative to the logits' RMS.", "",
                "| model | scales | tokens/step | PPL fake-quant | PPL INT4 | argmax agreement | worst per-token NLL diff | "
                "worst logit diff / RMS |", "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            for c, d in r["decode_path_int4_vs_fake_quant"].items():
                out.append(f"| {name(r)} | {scales(r)} | {c} | {d['ppl_fake']:.3f} | {d['ppl_int4']:.3f} | "
                           f"{100 * d['argmax_agreement']:.2f}% | {d['max_abs_nll_diff']:.4f} | {d['max_logit_diff_rel']:.2e} |")
    out += ["", "### Speed", "", "Decode tokens/s and TTFT (ms) on a 65-token chat prompt, 256 new tokens. Eager = plain "
            "`generate`; CUDA graph = each decode step replayed as one graph over a static KV cache.", "",
            "| model | scales | eager tok/s FP16 -> INT4 | CUDA graph tok/s FP16 -> INT4 | TTFT ms FP16 -> INT4 | peak mem GB | "
            "graph vs eager on the same tokens: argmax agreement, worst logit diff / RMS |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        f, q = r["fp16"], r["int4"]
        g = (f"{f['cuda_graph']['decode_tokens_per_s']:.1f} -> {q['cuda_graph']['decode_tokens_per_s']:.1f} "
             f"({r['speedup_decode_cuda_graph']:.2f}x)" if "speedup_decode_cuda_graph" in r else
             "failed: " + str(q.get("cuda_graph", f.get("cuda_graph", {})).get("error", "?"))[:60])
        fc = q.get("graph_vs_eager_forced")
        same = (f"{100 * fc['argmax_agreement']:.1f}% of {fc['positions']}, {fc['max_logit_diff_rel']:.1e}" if fc else
                None if q.get("graph_vs_eager_identical_prefix") is None else
                f"first differing generated token: {q['graph_vs_eager_identical_prefix']}")
        out.append(f"| {name(r)} | {scales(r)} | "
                   f"{f['eager']['decode_tokens_per_s']:.1f} -> {q['eager']['decode_tokens_per_s']:.1f} "
                   f"({r['speedup_decode_eager']:.2f}x) | {g} | {f['eager']['ttft_ms']:.0f} -> {q['eager']['ttft_ms']:.0f} | "
                   f"{f['peak_mem_bytes'] / 1e9:.2f} -> {q['peak_mem_bytes'] / 1e9:.2f} | "
                   f"{same or '-'} |")
    out += ["", "Where eager decode time goes (decode only, after prefill). Graph / eager = CUDA-graph step time / eager "
            "step time: the graph step is about the GPU's own work, so the rest of an eager step is CPU launch time. "
            "Kernel shares come from the profiler and are shown only where its kernel time covers at least 90% of "
            "the graph step (it can drop records on long windows).", "",
            "| model | scales | run | eager ms/step | graph ms/step | graph / eager | profiler coverage | INT4 kernels | "
            "dense GEMM | dequantize | other |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        for lab in ("fp16", "int4"):
            x, p = r[lab], r[lab]["profile"]
            s = p.get("share_pct", {})
            eager = p.get("wall_ms_per_step", p["wall_ms_unprofiled"] / p["steps"])
            graph = p.get("graph_ms_per_step") or (1000 / x["cuda_graph"]["decode_tokens_per_s"]
                                                    if "decode_tokens_per_s" in x.get("cuda_graph", {}) else None)
            kern = p.get("gpu_kernel_ms_per_step", p["gpu_kernel_ms"] / p["steps"])
            cov = kern / graph if graph else None
            shares = (f"{s.get('int4_kernels', 0):.0f}% | {s.get('gemm', 0):.0f}% | {s.get('dequant', 0):.0f}% | "
                      f"{s.get('other', 0):.0f}%" if cov and cov >= 0.9 else "incomplete | | |")
            out.append(f"| {name(r)} | {scales(r)} | {lab} | {eager:.1f} | {graph:.1f} | {100 * graph / eager:.0f}% | "
                       f"{100 * cov:.0f}% | {shares} |" if graph else
                       f"| {name(r)} | {scales(r)} | {lab} | {eager:.1f} | - | - | - | {shares} |")
    return out


def name(r):
    return r["model"].split("/")[-1]


def scales(r):
    return "clip search" if r.get("clip_search") else "absmax"


def main():
    lines, e, csvs = [], [], []
    for p in sys.argv[1:]:
        if p.endswith(".csv"):
            csvs.append(open(p).read())
            continue
        d = json.load(open(p))
        if "bench" in d:
            lines += a1(d)
        elif "fake_quant" in d:
            e.append(d)
    for c in csvs:
        lines += ncu(c)
    if e:
        lines += [""] + e2e(e)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
