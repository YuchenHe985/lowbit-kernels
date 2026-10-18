"""A2: the same Hugging Face model in FP16 and with INT4 weight-only linear layers, on a fixed workload.

    python bench/e2e.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/e2e_1.5b.json

Measures TTFT, decode tokens/s at batch 1, peak memory, weight bytes, WikiText-2 perplexity, greedy-token agreement
with FP16, and where decode time goes on the GPU (profiler, by kernel), so an end-to-end change is attributed.
"""
import argparse
import gc
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lowbit import model as lbmodel  # noqa: E402
from gpu_run import env  # noqa: E402

PROMPT = ("Explain, step by step, how a GPU executes a matrix-vector product for one decoding step of a large language "
          "model, and why memory bandwidth rather than arithmetic usually limits it.")


def load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
    except TypeError:                                                         # transformers < 4.56
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    m = m.cuda().eval()
    return tok, m


def weight_bytes(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) + sum(
        mod.wq.nbytes() for mod in m.modules() if isinstance(mod, lbmodel.W4Linear))


@torch.inference_mode()
def speed(tok, m, new_tokens, reps=3):
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt")
    ids = (ids["input_ids"] if not torch.is_tensor(ids) else ids).cuda()     # transformers 5 returns a BatchEncoding
    kw = dict(do_sample=False, pad_token_id=tok.eos_token_id)
    m.generate(ids, max_new_tokens=8, min_new_tokens=8, **kw)                      # warm-up
    ttft, total = [], []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        m.generate(ids, max_new_tokens=1, min_new_tokens=1, **kw)
        torch.cuda.synchronize(); ttft.append(time.perf_counter() - t)
        t = time.perf_counter()
        out = m.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, **kw)
        torch.cuda.synchronize(); total.append(time.perf_counter() - t)
    ttft.sort(); total.sort()
    t1, tt = ttft[len(ttft) // 2], total[len(total) // 2]
    return {"prompt_tokens": ids.shape[1], "new_tokens": new_tokens, "ttft_ms": 1000 * t1,
            "decode_tokens_per_s": (new_tokens - 1) / (tt - t1), "total_s": tt}, out[0, ids.shape[1]:].tolist()


@torch.inference_mode()
def where_time_goes(tok, m, steps=32):
    """GPU time by kernel over `steps` decode steps (profiler); grouped into int4 GEMV / other GEMM / the rest."""
    from torch.profiler import ProfilerActivity, profile
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    kw = dict(do_sample=False, pad_token_id=tok.eos_token_id, max_new_tokens=steps, min_new_tokens=steps)
    m.generate(ids, **kw)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize(); t = time.perf_counter()
        m.generate(ids, **kw)
        torch.cuda.synchronize(); wall = time.perf_counter() - t
    rows = [(e.key, e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total)
            for e in prof.key_averages() if getattr(e, "device_type", None) is not None]
    kern = [(k, us) for k, us in rows if us > 0 and not k.startswith(("aten::", "cuda", "Memcpy", "Memset")) and "::" not in k[:6]]
    gpu = sum(us for _, us in kern)
    groups = {"int4_gemv": 0.0, "gemm": 0.0, "dequant": 0.0, "other": 0.0}
    for k, us in kern:
        g = ("int4_gemv" if "gemv_v" in k else "dequant" if "dequant_kernel" in k else
             "gemm" if any(s in k.lower() for s in ("gemm", "gemv", "cutlass", "turing_", "volta_")) else "other")
        groups[g] += us
    top = sorted(kern, key=lambda r: -r[1])[:12]
    return {"steps": steps, "wall_ms": 1000 * wall, "gpu_kernel_ms": gpu / 1000, "gpu_busy_pct": 100 * gpu / 1000 / (1000 * wall),
            "share_pct": {g: 100 * v / gpu for g, v in groups.items()} if gpu else {},
            "top_kernels": [{"name": k[:90], "ms": us / 1000} for k, us in top]}


@torch.inference_mode()
def perplexity(tok, m, windows, ctx=1024):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    nll, n = 0.0, 0
    for i in range(windows):
        chunk = ids[i * ctx:(i + 1) * ctx].unsqueeze(0).cuda()
        if chunk.shape[1] < ctx:
            break
        out = m(chunk, labels=chunk)
        nll += out.loss.float().item() * (ctx - 1)
        n += ctx - 1
    return {"windows": i + 1 if n else 0, "ctx": ctx, "tokens": n, "ppl": float(torch.exp(torch.tensor(nll / n)))}


def measure(tok, m, label, a):
    torch.cuda.reset_peak_memory_stats()
    s, toks = speed(tok, m, a.new_tokens)
    r = {"label": label, "weight_bytes": weight_bytes(m), **s, "peak_mem_bytes_decode": torch.cuda.max_memory_allocated()}
    r["profile"] = where_time_goes(tok, m)
    r["wikitext2"] = perplexity(tok, m, a.ppl_windows)
    print(label, json.dumps({k: v for k, v in r.items() if k != "profile"}), flush=True)
    print(label, "profile", json.dumps(r["profile"]["share_pct"]), flush=True)
    return r, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--ppl-windows", type=int, default=40)
    a = ap.parse_args()
    tok, m = load(a.model)
    fp16, toks16 = measure(tok, m, "fp16", a)
    t = time.time()
    replaced, w4 = lbmodel.quantize_model(m)
    qsec = time.time() - t
    gc.collect(); torch.cuda.empty_cache()
    int4, toks4 = measure(tok, m, "int4_w4a16_g128", a)
    agree = next((i for i, (x, y) in enumerate(zip(toks16, toks4)) if x != y), min(len(toks16), len(toks4)))
    res = {"env": env(), "model": a.model, "layers_replaced": replaced, "int4_linear_bytes": w4, "quantize_seconds": qsec,
           "fp16": fp16, "int4": int4, "greedy_tokens_identical_prefix": agree,
           "speedup_decode": int4["decode_tokens_per_s"] / fp16["decode_tokens_per_s"],
           "ppl_delta": int4["wikitext2"]["ppl"] - fp16["wikitext2"]["ppl"]}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print("SUMMARY", json.dumps({k: v for k, v in res.items() if k not in ("fp16", "int4", "env")}), flush=True)


if __name__ == "__main__":
    main()
