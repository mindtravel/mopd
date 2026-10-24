"""Resident vLLM engines on several GPUs (mopd.rollout.engine_worker) and the one coordinator that
feeds them: used by benchmark generation (scripts/generate_benchmark_vllm.py) and on-policy rollouts
(mopd.rollout.vllm_processes.VllmRollouts).

run(items, session, policy) streams (key, result, device) in finish order. Rows are handed to whichever
engine asks, but no engine may hold more than its fair share of the unfinished rows, ceil((queued + held
anywhere) / engines): the first engine to start cannot take everything, and as a session drains the
share shrinks so the tail spreads over every GPU. Engines persist across sessions (one per training run).
"""

import collections
import contextlib
import json
import os
import queue
import signal
import subprocess
import threading
from pathlib import Path

from mopd import paths
from mopd.runtime import compile_cache
from mopd.training.distributed import engine_env
from mopd.rollout.engine_worker import TAG


class EnginePool:
    def __init__(self, devices, engine, log_dir, cache_dir, python=None):
        """devices: physical GPU ids; engine: engine_worker config (model, max_model_len, max_seqs,
        max_num_batched_tokens, memory, prefix_caching, max_lora_rank, seed)."""
        self.devices = [int(str(d).split(":")[-1]) for d in devices]
        log_dir, cache_dir = Path(log_dir), Path(cache_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.procs, self.inbox, self.logs, self.session = {}, {}, {}, 0
        try:
            for d in self.devices:
                env = engine_env(
                    **compile_cache.isolated(cache_dir / f"gpu{d}"),
                    CUDA_VISIBLE_DEVICES=str(d),
                    PYTHONPATH=str(paths.ROOT / "src")
                    + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
                )
                self.logs[d] = (log_dir / f"engine-gpu{d}.log").open("a")
                self.procs[d] = subprocess.Popen(
                    [str(python or paths.INFER_PY), "-u", "-m", "mopd.rollout.engine_worker", json.dumps(engine)],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self.logs[d],
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                self.inbox[d] = queue.Queue()
                threading.Thread(target=self._reader, args=(d,), daemon=True).start()
            self.eos = {d: self._next(d)["eos"] for d in self.devices}
        except BaseException:
            self.close()
            raise

    def _reader(self, d):
        for line in self.procs[d].stdout:
            if line.startswith(TAG):
                self.inbox[d].put(json.loads(line[len(TAG) :]))
        self.inbox[d].put({"error": f"engine on GPU {d} exited ({self.procs[d].wait()}); see {self.logs[d].name}"})

    def _next(self, d):
        m = self.inbox[d].get()
        if "error" in m:
            raise RuntimeError(f"GPU {d}: {m['error']}")
        return m

    def _send(self, d, value):
        self.procs[d].stdin.write(json.dumps(value) + "\n")
        self.procs[d].stdin.flush()

    def run(self, items, session, policy=None):
        """items: [(key, prompt_token_ids)]; session: sampling (max_tokens, temperature, top_p, seed,
        min_tokens, logprobs). Yields (key, result, device) as rows finish; afterwards self.stats holds
        per-engine counts and weight-load seconds."""
        self.session += 1
        work = collections.deque(items)
        held, per_gpu, total = collections.Counter(), collections.Counter(), len(items)
        lock, results, failures = threading.Lock(), queue.Queue(), []
        idle = {}

        def serve(d):
            try:
                self._send(d, {"session": {**session, "session": self.session}, "policy": policy})
                while True:
                    m = self._next(d)
                    if "want" in m:
                        with lock:
                            cap = -(-(len(work) + sum(held.values())) // len(self.devices))
                            rows = [work.popleft() for _ in range(min(m["want"], max(0, cap - held[d]), len(work)))]
                            held[d] += len(rows)
                            reply = {"rows": rows} if rows or work else {"eof": True}
                        self._send(d, reply)
                    elif "key" in m:
                        with lock:
                            held[d] -= 1
                            per_gpu[d] += 1
                        results.put((m["key"], m, d))
                    elif m.get("idle"):
                        if m.get("session") != self.session:
                            raise RuntimeError(f"GPU {d}: idle for session {m.get('session')}, expected {self.session}")
                        idle[d] = m
                        return
            except BaseException as e:
                failures.append(e)
            finally:
                results.put(None)

        threads = [threading.Thread(target=serve, args=(d,), daemon=True) for d in self.devices]
        for t in threads:
            t.start()
        finished = done = 0
        while finished < len(threads):
            item = results.get()
            if item is None:
                finished += 1
                if failures:
                    raise failures[0]
                continue
            done += 1
            yield item
        if done != total:
            raise RuntimeError(f"session {self.session}: {done} of {total} rows came back")
        self.stats = {
            "per_gpu": {str(d): per_gpu[d] for d in self.devices},
            "load_seconds": {str(d): round(idle[d].get("load_seconds", 0.0), 3) for d in idle},
            "policy_id": {str(d): idle[d].get("policy_id") for d in idle},
        }

    def close(self):
        for d, proc in self.procs.items():
            with contextlib.suppress(Exception):
                self._send(d, {"stop": True})
        for d, proc in self.procs.items():
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for f in self.logs.values():
            f.close()
        self.procs = {}
