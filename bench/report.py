"""Render the measured results as Markdown tables (the README's numbers come only from here).

    python bench/report.py results/a1.json [results/e2e_*.json] > results/RESULTS.md
"""
import json
import sys


def fmt_env(e):
    return (f"{e['gpu']} (sm_{e['sm'].replace('.', '')}, {e['sms']} SMs), torch {e['torch']}, CUDA {e['cuda']}, "
            f"commit `{e['commit']}`, {e['time_utc']}")


def a1(r):
    out = ["## A1: kernels", "", f"Environment: {fmt_env(r['env'])}", "",
           f"Measured device-to-device copy bandwidth: **{r['copy_bandwidth_gbps']:.0f} GB/s** "
           f"(spec {r['spec_bandwidth_gbps']:.0f} GB/s); `% of copy BW` uses the measured figure.", "",
           "Correctness: every version against an fp64 reference on the same quantized weights, M = 1..8, all shapes "
           f"plus odd-N tails. Worst max-abs error / RMS(output): v1 {r['worst_rel_err']['v1']:.2e}, "
           f"v2 {r['worst_rel_err']['v2']:.2e}, v3 {r['worst_rel_err']['v3']:.2e}, "
           f"cuBLAS FP16 {r['worst_rel_err']['cublas']:.2e}.", ""]
    b = r["bench"]
    out += ["### Decode (M = 1): median latency, microseconds", "",
            "| layer | K x N | FP16 cuBLAS | dequant + cuBLAS | v1 | v2 | v3 | v3 speedup vs FP16 | v3 % of copy BW |",
            "|---|---|---|---|---|---|---|---|---|"]
    for x in b:
        if x["M"] != 1:
            continue
        out.append(f"| {x['shape']} | {x['K']} x {x['N']} | {x['fp16_cublas']['us']:.1f} | {x['dequant_cublas']['us']:.1f} | "
                   f"{x['v1']['us']:.1f} | {x['v2']['us']:.1f} | {x['v3']['us']:.1f} | "
                   f"{x['fp16_cublas']['us'] / x['v3']['us']:.2f}x | {x['v3']['pct_peak']:.0f}% |")
    out += ["", "### Small batch (M = 1, 2, 4, 8): v3 vs FP16 cuBLAS, microseconds", "",
            "| layer | " + " | ".join(f"M={m}" for m in (1, 2, 4, 8)) + " |", "|---|---|---|---|---|"]
    for name in dict.fromkeys(x["shape"] for x in b):
        cells = []
        for m in (1, 2, 4, 8):
            x = next((y for y in b if y["shape"] == name and y["M"] == m), None)
            cells.append(f"{x['v3']['us']:.1f} / {x['fp16_cublas']['us']:.1f}" if x else "-")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    pre = [x for x in b if x["M"] > 8]
    if pre:
        out += ["", "### Prefill: dequant + cuBLAS overhead vs FP16 cuBLAS", "",
                "| layer | " + " | ".join(f"M={m}" for m in sorted({x['M'] for x in pre})) + " |",
                "|---" * (1 + len({x['M'] for x in pre})) + "|"]
        for name in dict.fromkeys(x["shape"] for x in pre):
            row = sorted((x for x in pre if x["shape"] == name), key=lambda x: x["M"])
            out.append(f"| {name} | " + " | ".join(
                f"{x['dequant_cublas']['us'] / x['fp16_cublas']['us']:.2f}x" for x in row) + " |")
    return out


def e2e(rs):
    out = ["## A2: end to end (Hugging Face, batch 1, greedy)", "", f"Environment: {fmt_env(rs[0]['env'])}", "",
           "| model | weights FP16 -> INT4 | decode tok/s FP16 -> INT4 | TTFT ms FP16 -> INT4 | peak mem GB | "
           "WikiText-2 PPL FP16 -> INT4 | identical greedy tokens |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        f, q = r["fp16"], r["int4"]
        out.append(
            f"| {r['model'].split('/')[-1]} | {f['weight_bytes'] / 2**30:.2f} -> {q['weight_bytes'] / 2**30:.2f} GiB | "
            f"{f['decode_tokens_per_s']:.1f} -> {q['decode_tokens_per_s']:.1f} ({r['speedup_decode']:.2f}x) | "
            f"{f['ttft_ms']:.0f} -> {q['ttft_ms']:.0f} | {f['peak_mem_bytes_decode'] / 1e9:.2f} -> "
            f"{q['peak_mem_bytes_decode'] / 1e9:.2f} | {f['wikitext2']['ppl']:.3f} -> {q['wikitext2']['ppl']:.3f} | "
            f"{r['greedy_tokens_identical_prefix']} / {f['new_tokens']} |")
    out += ["", "Where decode GPU time goes (profiler, 32 steps):", "",
            "| model | run | GPU busy | INT4 GEMV | other GEMM | dequant | other |", "|---|---|---|---|---|---|---|"]
    for r in rs:
        for lab in ("fp16", "int4"):
            p = r[lab]["profile"]
            s = p.get("share_pct", {})
            out.append(f"| {r['model'].split('/')[-1]} | {lab} | {p['gpu_busy_pct']:.0f}% | {s.get('int4_gemv', 0):.0f}% | "
                       f"{s.get('gemm', 0):.0f}% | {s.get('dequant', 0):.0f}% | {s.get('other', 0):.0f}% |")
    return out


def main():
    docs = [json.load(open(p)) for p in sys.argv[1:]]
    lines = []
    for d in docs:
        if "bench" in d:
            lines += a1(d) + [""]
    e = [d for d in docs if "speedup_decode" in d]
    if e:
        lines += e2e(e)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
