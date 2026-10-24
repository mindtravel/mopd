"""mopd.serving without GPUs: request validation, job state and restart re-queue, /metrics parsing,
per-job utilization."""

import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mopd.serving import engine, protocol
from mopd.serving.jobs import Jobs

METRICS = """# HELP x
vllm:generation_tokens_total{engine="0",model_name="policy"} 1500.0
vllm:prompt_tokens_total{engine="0",model_name="policy"} 9000.0
vllm:prefix_cache_hits_total{engine="0",model_name="policy"} 6000.0
vllm:iteration_tokens_total_count{engine="0",model_name="policy"} 300.0
vllm:iteration_tokens_total_sum{engine="0",model_name="policy"} 4500.0
"""


def model_dir(root):
    d = Path(root) / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "num_attention_heads": 2, "head_dim": 4, "num_key_value_heads": 1}))
    return d


class ServingTest(unittest.TestCase):
    def test_request_validation(self):
        with tempfile.TemporaryDirectory() as t:
            m = model_dir(t)
            r = protocol.request({"model": str(m)})
            self.assertEqual((r["name"], r["benchmarks"], r["auto_domains"]), ("m", list(protocol.BENCHMARKS), "simple"))
            for bad in ({"model": "relative"}, {"model": str(m), "benchmarks": ["mmlu"]},
                        {"model": str(m), "auto_domains": "sales,nope"}, {"model": str(m), "ifbench_limit": 0}):
                with self.assertRaises(ValueError):
                    protocol.request(bad)

    def test_job_state_and_restart_requeue(self):
        with tempfile.TemporaryDirectory() as t:
            m = model_dir(t)
            jobs = Jobs(Path(t))
            j = jobs.submit({"model": str(m), "benchmarks": ["ifbench", "automationbench"]})
            self.assertEqual(j["protocol"]["ifbench"], protocol.IFBENCH)
            jobs.take()
            jobs.update(j, "ifbench", state="complete", scores={"x": 1})
            jobs.update(j, "automationbench", state="running")
            self.assertEqual(j["state"], "running")
            again = Jobs(Path(t))  # service restart: unfinished part re-queued, finished part kept
            k = again.jobs[j["id"]]
            self.assertEqual(again.queue, [j["id"]])
            self.assertEqual((k["parts"]["ifbench"]["state"], k["parts"]["automationbench"]["state"]), ("complete", "queued"))

    def test_counters_parse(self):
        with mock.patch("urllib.request.urlopen") as u:
            u.return_value.read.return_value = METRICS.encode()
            c = engine.counters(18106)
        self.assertEqual((c["generated"], c["prompt"], c["prefix_hits"], c["steps"]), (1500, 9000, 6000, 300))
        self.assertIsNone(c["preemptions"])  # absent metric: not zero

    def test_job_utilization(self):
        from mopd.serving.service import utilization

        with tempfile.TemporaryDirectory() as t:
            m = model_dir(t)
            zero = dict.fromkeys(engine.COUNTERS, 0.0)
            after = {"generated": 1500.0, "prompt": 9000.0, "prefix_hits": 6000.0, "steps": 300.0, "preemptions": 0.0}
            with mock.patch("mopd.profiling.model.cached_peaks", return_value={"bf16_tflops": 200.0, "hbm_gbps": 1600}):
                u = utilization({"model": str(m)}, 6, zero, after, 10.0)
        self.assertEqual(u["tokens_per_step"], (1500 + 3000) / 300)  # computed prefill excludes cache hits
        self.assertEqual(u["prefix_hit"], round(6000 / 9000, 3))
        self.assertIn("mbu_weights", u)


if __name__ == "__main__":
    unittest.main()
