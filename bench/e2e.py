"""A2: the same Hugging Face model in FP16, as fake-quant (FP16 layers holding the dequantized INT4 weights) and
with the INT4 kernels, on a fixed workload.

    python bench/e2e.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/e2e_1.5b.json [--clip]

Quality: WikiText-2 perplexity, and next-token agreement with FP16 over the same tokens (teacher-forced, so one
early difference does not decide the score). Fake-quant and INT4 multiply by the same weights, so a gap between
them is a kernel error, while their gap to FP16 is the price of 4-bit weights.
Speed at batch 1: TTFT and decode tokens/s, both in eager mode (what plain `generate` gives) and with each decode
step captured as one CUDA graph over a static KV cache (how serving engines remove per-step CPU overhead), plus
where eager decode GPU time goes (profiler, by kernel).
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
CLIP_GRID = tuple(round(1.0 - 0.025 * i, 3) for i in range(11))        # 1.00, 0.975, ..., 0.75


def load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
    except TypeError:                                                         # transformers < 4.56
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    return tok, m.cuda().eval()


def weight_bytes(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) + sum(
        mod.wq.nbytes() for mod in m.modules() if isinstance(mod, lbmodel.W4Linear))


def prompt_ids(tok):
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt")
    return (ids["input_ids"] if not torch.is_tensor(ids) else ids).cuda()     # transformers 5 returns a BatchEncoding


@torch.inference_mode()
def speed_eager(tok, m, new_tokens, reps=3):
    ids = prompt_ids(tok)
    kw = dict(do_sample=False, pad_token_id=tok.eos_token_id)
    m.generate(ids, max_new_tokens=8, min_new_tokens=8, **kw)                      # warm-up
    ttft, total = [], []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        m.generate(ids, max_new_tokens=1, min_new_tokens=1, **kw)
        torch.cuda.synchronize(); ttft.append(time.perf_counter() - t)
        t = time.perf_counter()
        m.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, **kw)
        torch.cuda.synchronize(); total.append(time.perf_counter() - t)
    ttft.sort(); total.sort()
    t1, tt = ttft[len(ttft) // 2], total[len(total) // 2]
    return {"prompt_tokens": ids.shape[1], "new_tokens": new_tokens, "ttft_ms": 1000 * t1,
            "decode_tokens_per_s": (new_tokens - 1) / (tt - t1)}


def static_cache(m, max_len):
    from transformers import StaticCache
    for kw in (dict(config=m.config, max_cache_len=max_len),
               dict(config=m.config, max_batch_size=1, max_cache_len=max_len, device="cuda", dtype=torch.float16)):
        try:
            return StaticCache(**kw)
        except TypeError:
            continue
    raise RuntimeError("could not build a StaticCache with this transformers version")


@torch.inference_mode()
def speed_graph(tok, m, new_tokens, reps=3):
    """Greedy decode with one CUDA graph per step: prefill eagerly into a static KV cache, capture a single-token
    forward (fixed shapes: the token and its cache position live in static tensors), then replay it per token."""
    ids = prompt_ids(tok)
    p = ids.shape[1]
    cache = static_cache(m, p + new_tokens + 8)
    tok_in = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    pos = torch.zeros(1, dtype=torch.long, device="cuda")

    def step():
        return m(input_ids=tok_in, past_key_values=cache, cache_position=pos, use_cache=True).logits[:, -1].argmax(-1)

    def prefill():
        cache.reset()
        out = m(input_ids=ids, past_key_values=cache, cache_position=torch.arange(p, device="cuda"), use_cache=True)
        return out.logits[:, -1].argmax(-1)

    nxt = prefill()
    tok_in.copy_(nxt.view(1, 1)); pos.fill_(p)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    toks = torch.empty(new_tokens, dtype=torch.long, device="cuda")
    ttft, dec = [], []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        nxt = prefill()
        torch.cuda.synchronize(); ttft.append(time.perf_counter() - t)
        toks[0] = nxt[0]
        t = time.perf_counter()
        for i in range(1, new_tokens):
            tok_in.copy_(toks[i - 1].view(1, 1))
            pos.fill_(p + i - 1)
            graph.replay()
            toks[i] = out[0]
        torch.cuda.synchronize(); dec.append(time.perf_counter() - t)
    ttft.sort(); dec.sort()
    return {"prompt_tokens": p, "new_tokens": new_tokens, "ttft_ms": 1000 * ttft[len(ttft) // 2],
            "decode_tokens_per_s": (new_tokens - 1) / dec[len(dec) // 2]}, toks.tolist()


@torch.inference_mode()
def where_time_goes(tok, m, steps=32):
    """GPU time by kernel over `steps` eager decode steps (profiler); grouped into INT4 kernels / GEMM / the rest."""
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
    groups = {"int4_kernels": 0.0, "gemm": 0.0, "dequant": 0.0, "other": 0.0}
    for k, us in kern:
        g = ("int4_kernels" if ("gemv_v" in k or "gemm_v5" in k) else "dequant" if "dequant_kernel" in k else
             "gemm" if any(s in k.lower() for s in ("gemm", "gemv", "cutlass", "turing_", "volta_")) else "other")
        groups[g] += us
    top = sorted(kern, key=lambda r: -r[1])[:12]
    return {"steps": steps, "wall_ms": 1000 * wall, "gpu_kernel_ms": gpu / 1000, "gpu_busy_pct": 100 * gpu / 1000 / (1000 * wall),
            "share_pct": {g: 100 * v / gpu for g, v in groups.items()} if gpu else {},
            "top_kernels": [{"name": k[:90], "ms": us / 1000} for k, us in top]}


def wikitext_ids(tok):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return tok(text, return_tensors="pt").input_ids[0]


@torch.inference_mode()
def quality(ids, m, windows, ref_argmax=None, ctx=1024):
    """Perplexity over `windows` non-overlapping windows, plus the argmax next token at every position; with
    `ref_argmax` (FP16's), also the share of positions where this model picks the same token."""
    nll, n, argmax = 0.0, 0, []
    for i in range(windows):
        chunk = ids[i * ctx:(i + 1) * ctx].unsqueeze(0).cuda()
        if chunk.shape[1] < ctx:
            break
        logits = m(chunk).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits, chunk[0, 1:], reduction="sum").item()
        n += ctx - 1
        argmax.append(logits.argmax(-1).cpu())
    argmax = torch.cat(argmax)
    r = {"windows": len(argmax) // (ctx - 1), "ctx": ctx, "tokens": n, "ppl": float(torch.exp(torch.tensor(nll / n)))}
    if ref_argmax is not None:
        r["next_token_agreement_with_fp16"] = float((argmax == ref_argmax).float().mean())
    return r, argmax


def measure(tok, m, label, a, ref_argmax=None, speed=True):
    torch.cuda.reset_peak_memory_stats()
    r = {"label": label, "weight_bytes": weight_bytes(m)}
    r["wikitext2"], argmax = quality(a.ppl_ids, m, a.ppl_windows, ref_argmax)
    toks = None
    if speed:
        r["eager"] = speed_eager(tok, m, a.new_tokens)
        try:
            r["cuda_graph"], toks = speed_graph(tok, m, a.new_tokens)
        except Exception as e:                                            # report it, keep the eager numbers
            r["cuda_graph"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        r["profile"] = where_time_goes(tok, m)
        r["peak_mem_bytes"] = torch.cuda.max_memory_allocated()
    print(label, json.dumps({k: v for k, v in r.items() if k != "profile"}), flush=True)
    if "profile" in r:
        print(label, "profile", json.dumps(r["profile"]["share_pct"]), flush=True)
    return r, argmax, toks


def first_diff(x, y):
    if x is None or y is None:
        return None
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--ppl-windows", type=int, default=40)
    ap.add_argument("--clip", action="store_true", help="per-group clipping search for the scales (CLIP_GRID)")
    ap.add_argument("--gemv-version", type=int, default=0, help="0 = pick by batch size (ops.linear)")
    a = ap.parse_args()
    tok, m = load(a.model)
    a.ppl_ids = wikitext_ids(tok)                       # first, so a dataset problem fails before any timing
    fp16, ref_argmax, toks16 = measure(tok, m, "fp16", a)

    t = time.time()
    layers = lbmodel.quantize_layers(m, clip_grid=CLIP_GRID if a.clip else None)
    torch.cuda.synchronize()
    qsec = time.time() - t
    lbmodel.apply_fake_quant(layers)
    fake, fake_argmax, _ = measure(tok, m, "fake_quant", a, ref_argmax, speed=False)
    replaced, w4 = lbmodel.apply_int4(layers, gemv_version=a.gemv_version)
    gc.collect(); torch.cuda.empty_cache()
    int4, int4_argmax, toks4 = measure(tok, m, f"int4_w4a16_g128{'_clip' if a.clip else ''}", a, ref_argmax)

    res = {"env": env(), "model": a.model, "clip_search": a.clip, "gemv_version": a.gemv_version,
           "layers_replaced": replaced, "int4_linear_bytes": w4, "quantize_seconds": qsec,
           "fp16": fp16, "fake_quant": fake, "int4": int4,
           "int4_vs_fake_quant_next_token_agreement": float((int4_argmax == fake_argmax).float().mean()),
           "int4_vs_fake_quant_ppl_rel_diff": abs(int4["wikitext2"]["ppl"] / fake["wikitext2"]["ppl"] - 1),
           "greedy_identical_prefix_graph": first_diff(toks16, toks4),
           "ppl_delta": int4["wikitext2"]["ppl"] - fp16["wikitext2"]["ppl"]}
    for mode in ("eager", "cuda_graph"):
        f, q = fp16.get(mode, {}), int4.get(mode, {})
        if "decode_tokens_per_s" in f and "decode_tokens_per_s" in q:
            res[f"speedup_decode_{mode}"] = q["decode_tokens_per_s"] / f["decode_tokens_per_s"]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print("SUMMARY", json.dumps({k: v for k, v in res.items() if k not in ("fp16", "fake_quant", "int4", "env")}), flush=True)


if __name__ == "__main__":
    main()
