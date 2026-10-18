"""Build the single-file Kaggle script (the CLI pushes one code file) that embeds this repo's sources and runs
the CPU tests, a compute-sanitizer memcheck and the A1 GPU run on a Tesla T4.

    python kaggle/build_kernel.py [quick|full|e2e] # writes kernel/ (e2e: A2 on Qwen2.5 0.5B/1.5B/3B)
    kaggle kernels push -p kernel
    kaggle kernels output <user>/lowbit-kernels -p results/kaggle
"""
import json
import os
import pathlib
import subprocess
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
root = pathlib.Path(__file__).resolve().parent.parent
git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
commit = (git("rev-parse", "--short", "HEAD") or "uncommitted") + ("-dirty" if git("status", "--porcelain") else "")
files = {p: (root / p).read_text() for p in
         ["lowbit/__init__.py", "lowbit/quant.py", "lowbit/ops.py", "lowbit/model.py", "csrc/w4_gemv.cu",
          "bench/gpu_run.py", "bench/e2e.py", "tests/test_quant.py"]}
script = f'''import os, pathlib, subprocess, sys
FILES = {files!r}
MODE = {mode!r}
work = pathlib.Path("/kaggle/working/lowbit-kernels")
for rel, text in FILES.items():
    (work / rel).parent.mkdir(parents=True, exist_ok=True)
    (work / rel).write_text(text)
os.chdir(work)
env = dict(os.environ, PYTHONPATH=str(work), LOWBIT_COMMIT={commit!r})

def run(cmd, check=True):
    print("RUN", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env)
    if check and r.returncode:
        sys.exit(f"failed: {{' '.join(cmd)}} -> {{r.returncode}}")
    return r.returncode

subprocess.run(["nvidia-smi"])
run([sys.executable, "-m", "pytest", "-q", "tests"])
# memcheck on a small case of every kernel (out-of-bounds or misaligned access fails the run)
pathlib.Path("memcheck.py").write_text(
    "import torch\\nfrom lowbit import ops\\n"
    "for k, n in ((256, 37), (896, 128)):\\n"
    "    wq = ops.W4Weight.from_float(torch.randn(n, k))\\n"
    "    for m in (1, 3, 8):\\n"
    "        x = torch.randn(m, k).half().cuda()\\n"
    "        [ops.gemv(x, wq, v, r) for v, r in ((1, 0), (2, 0), (3, 0), (4, 1), (4, 2), (4, 4))]\\n"
    "        ops.linear(torch.randn(16, k).half().cuda(), wq)\\n"
    "torch.cuda.synchronize(); print('memcheck workload done')\\n")
run([sys.executable, "-c", "from lowbit import ops; ops.ext()"])   # build once, outside the sanitizer
san = "/usr/local/cuda/bin/compute-sanitizer"
if os.path.exists(san):
    pathlib.Path("results").mkdir(exist_ok=True)
    # pass 1: everything, eager module loading (the lazy loader's first cuKernelGetFunction probe is reported as an
    # API error under the sanitizer); pass 2 only if pass 1 is not clean: memory-access errors alone decide
    def san_run(extra, tag):
        cmd = [san, "--tool", "memcheck", "--error-exitcode", "3"] + extra + [sys.executable, "memcheck.py"]
        r = subprocess.run(cmd, env=dict(env, CUDA_MODULE_LOADING="EAGER"), capture_output=True, text=True)
        pathlib.Path(f"results/memcheck_{{tag}}.txt").write_text(" ".join(cmd) + "\\n" + r.stdout + r.stderr)
        print(tag, "exit", r.returncode, r.stdout[-600:], flush=True)
        return r.returncode
    if san_run([], "all") and san_run(["--report-api-errors", "no"], "memory_only"):
        sys.exit("memcheck found memory-access errors")
else:
    print("compute-sanitizer not found in this image", flush=True)
if MODE == "e2e":
    for size in ("0.5B", "1.5B", "3B"):
        run([sys.executable, "bench/e2e.py", "--model", f"Qwen/Qwen2.5-{{size}}-Instruct",
             "--out", f"results/e2e_{{size.lower()}}.json"], check=False)
else:
    run([sys.executable, "bench/gpu_run.py", "--out", "results/a1.json"] + (["--quick"] if MODE == "quick" else []))
subprocess.run(["cp", "-r", "results", "/kaggle/working/"])
print("DONE", flush=True)
'''
out = root / "kernel"
out.mkdir(exist_ok=True)
(out / "run_kaggle.py").write_text(script)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": os.environ.get("KAGGLE_USERNAME", "yuchenhe555") + "/lowbit-kernels" + ("-e2e" if mode == "e2e" else ""),
    "title": "lowbit kernels" + (" e2e" if mode == "e2e" else ""),
    "code_file": "run_kaggle.py", "language": "python", "kernel_type": "script", "is_private": "true",
    "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
print(f"built kernel/run_kaggle.py ({mode})")
