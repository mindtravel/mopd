"""The resident vLLM OpenAI server on one service GPU. use(job) makes it serve the job's checkpoint:
same architecture and tokenizer/chat template -> weights loaded in place (weight_swap, ~4 s instead
of a ~80 s start); a LoRA adapter on the running base -> added at runtime; anything else -> restart.
counters() reads vLLM's /metrics once, for the per-job utilization (mopd.profiling)."""

import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from mopd import paths
from mopd.runtime import compile_cache

STATE = paths.ROOT / "runs/bench-service"


def serving_signature(model, settings):
    """What a running server must share with a checkpoint for an in-place weight swap: the model
    config (minus bookkeeping fields), tokenizer and chat template (the server keeps its own), the
    serving settings, and for a LoRA its base."""
    model = Path(model)
    if (model / "adapter_config.json").exists():
        a = json.loads((model / "adapter_config.json").read_text())
        return {"lora_base": serving_signature(a["base_model_name_or_path"], settings)}
    c = json.loads((model / "config.json").read_text())
    for k in ("_name_or_path", "transformers_version", "torch_dtype", "dtype", "use_cache"):
        c.pop(k, None)
    h = hashlib.sha256(json.dumps(c, sort_keys=True).encode())
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        if (model / f).exists():
            h.update(f.encode() + (model / f).read_bytes())
    return {"model": h.hexdigest(), "settings": json.dumps(settings, sort_keys=True)}


class Engine:
    """use(job) -> (model name to request, {"engine": started | weights_swapped | lora_added, "seconds"})."""

    def __init__(self, gpu):
        self.gpu, self.port = gpu, 18100 + gpu
        self.proc = self.signature = None
        self.lora = set()

    def call(self, path, body, timeout=600):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout).read().decode()

    def use(self, j):
        model = Path(j["model"])
        sig = serving_signature(model, j["protocol"])
        lora = (model / "adapter_config.json").exists()
        if self.proc and self.proc.poll() is None and sig == self.signature:
            mark = time.time()
            if lora:
                if j["name"] not in self.lora:
                    self.call("/v1/load_lora_adapter", {"lora_name": j["name"], "lora_path": str(model)})
                    self.lora.add(j["name"])
                how, served = "lora_added", j["name"]
            else:
                self.call("/collective_rpc", {"method": "load_checkpoint", "args": [str(model)]})
                self.call("/reset_prefix_cache", {})
                how, served = "weights_swapped", "policy"
            return served, {"engine": how, "seconds": round(time.time() - mark, 1)}
        self.stop()
        mark = time.time()
        cmd = [str(paths.SERVE_PY.parent / "vllm"), "serve"]
        p = j["protocol"]
        if lora:
            a = json.loads((model / "adapter_config.json").read_text())
            cmd += [a["base_model_name_or_path"], "--served-model-name", "policy", "--enable-lora",
                    "--max-lora-rank", str(max(64, a.get("r", 16))), "--max-loras", "4",
                    "--lora-modules", f"{j['name']}={model}"]
        else:
            cmd += [str(model), "--served-model-name", "policy"]
        cmd += ["--host", "127.0.0.1", "--port", str(self.port), "--dtype", "bfloat16",
                "--max-model-len", str(p["max_model_len"]), "--gpu-memory-utilization", "0.9",
                "--max-num-seqs", "256", "--enable-auto-tool-choice", "--tool-call-parser", p["tool_parser"], "--seed", "0",
                # text only: a multimodal checkpoint (Qwen3.5) keeps no memory for images or video
                "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),
                "--worker-extension-cls", "mopd.serving.weight_swap.WeightSwap",
                *(["--reasoning-parser", "qwen3"] if p["thinking"] else [])]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu),
               # the serving env's own tools first: vLLM JIT-builds kernels with its ninja
               "PATH": f"{paths.SERVE_PY.parent}:{os.environ.get('PATH', '')}",
               "VLLM_SERVER_DEV_MODE": "1",  # /collective_rpc, /reset_prefix_cache (server is on 127.0.0.1 only)
               "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
               "PYTHONPATH": str(paths.ROOT / "src"),
               # this GPU's compiler caches (mopd.runtime.compile_cache), reused across restarts
               **compile_cache.isolated(paths.ROOT / "cache/compile" / f"bench-gpu{self.gpu}")}
        self.vlog = STATE / f"vllm-gpu{self.gpu}.log"
        with self.vlog.open("a") as f:
            f.write(f"\n==== {time.strftime('%F %T')} {' '.join(cmd)}\n")
            f.flush()
            self.proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        wait_ready(self.proc, self.port, self.vlog)
        self.signature, self.lora = sig, ({j["name"]} if lora else set())
        return (j["name"] if lora else "policy"), {"engine": "started", "seconds": round(time.time() - mark, 1)}

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc = self.signature = None


def wait_ready(proc, port, log, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited ({proc.returncode}); tail of {log}:\n" + log.read_text()[-3000:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).read()
            return
        except OSError:
            time.sleep(5)
    raise RuntimeError(f"vLLM not ready after {timeout} s")


COUNTERS = {"generated": "generation_tokens_total", "prompt": "prompt_tokens_total",
            "prefix_hits": "prefix_cache_hits_total", "steps": "iteration_tokens_total_count",
            "preemptions": "num_preemptions_total"}


def counters(port):
    """vLLM's cumulative counters (one /metrics read); diff two reads for a job's share."""
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    out = {}
    for k, name in COUNTERS.items():
        m = re.search(rf"^vllm:{name}(?:{{[^}}]*}})? ([0-9.e+-]+)$", text, re.M)
        out[k] = float(m.group(1)) if m else None
    return out
