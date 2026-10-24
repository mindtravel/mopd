import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(_ROOT / "monitor"))
import monitor_data as d
class Tests(unittest.TestCase):
 def test_partial_log_and_case_paging(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);run=root/"runs/dual";run.mkdir(parents=True)
   (run/"config.json").write_text(json.dumps({"steps":64}))
   (run/"metrics.jsonl").write_text(json.dumps({"step":1,"domains":{"math":{"count":16,"tokens":40,"truncated":0}}})+"\n")
   row={"id":"x","domain":"math","step":1,"tokens":8192,"truncated":True,"response":"<script>bad</script>","response_ids":[1,2],"prompt":"p"}
   (run/"rollouts.jsonl").write_text(json.dumps(row)+"\n"+'{"incomplete":')
   with patch.object(d,"ROOT",root),patch.object(d,"RUNS",{"dual":{"path":"runs/dual","label":"test"}}),patch.object(d,"gpu_snapshot",return_value={}):
    snap=d.snapshot("dual")
    self.assertEqual(snap["response_summary"]["count"],1)
    self.assertEqual(snap["response_summary"]["max_tokens"],8192)
    cases=d.cases("dual",step=1,domain="math",truncated=True)
    self.assertEqual(cases["total"],1);self.assertNotIn("response_ids",cases["rows"][0])
    self.assertEqual(cases["rows"][0]["response"],row["response"])
    self.assertEqual(d.cases("dual",domain="instruction")["total"],0)
 def test_unknown_run_rejected(self):
  with self.assertRaises(ValueError):d.run_path("../../etc")
 def test_benchmarks_are_scoped_to_model_version(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);run=root/"runs/fairy-run";run.mkdir(parents=True)
   (run/"config.json").write_text(json.dumps({"steps":2}))
   (run/"model-version.json").write_text(json.dumps({"version_id":"fairy-run"}))
   report=run/"benchmarks/ifeval-step-2";report.mkdir(parents=True)
   (report/"generation-config.json").write_text(json.dumps({
    "model_version_id":"fairy-run:step-0002","protocol_id":"ifeval-v1","backend":"hf_qat"}))
   (report/"summary.json").write_text(json.dumps({"total":10,"done":10,"domains":{"instruction":{"accuracy":0.7}}}))
   legacy=root/"runs/full-eval/old";legacy.mkdir(parents=True)
   (legacy/"generation-config.json").write_text(json.dumps({"backend":"vllm"}))
   (legacy/"summary.json").write_text(json.dumps({"total":541,"done":541,"domains":{"instruction":{"accuracy":0.9}}}))
   with patch.object(d,"ROOT",root),patch.object(d,"RUNS",{}):
    self.assertEqual(d.run_path("fairy-run"),run)
    results=d.benchmark_report("fairy-run",run)
    self.assertEqual(len(results),1)
    self.assertEqual(results[0]["model_version_id"],"fairy-run:step-0002")
    self.assertEqual(results[0]["protocol_id"],"ifeval-v1")
 def test_run_registry_discovers_training_versions_only(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);run=root/"runs/fairy-run";run.mkdir(parents=True)
   (run/"config.json").write_text(json.dumps({"steps":2,"student":"fairy2i"}))
   training={"inputs":{"domains":{"math":{"train":{"path":"data/math.jsonl","sha256":"abc123","size_bytes":9}}}}}
   (run/"model-version.json").write_text(json.dumps({"version_id":"fairy-run","training":training}))
   old=root/"runs/full-eval/old";old.mkdir(parents=True)
   (old/"generation-config.json").write_text("{}")
   with patch.object(d,"ROOT",root),patch.object(d,"RUNS",{}):
    versions=d.version_registry()
   self.assertEqual([v["key"] for v in versions],["fairy-run"])
   self.assertEqual(versions[0]["training"],training)
 def test_benchmark_generation_progress_survives_missing_summary(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);run=root/"runs/fairy-run";report=run/"evaluations/base/ifeval-v1"
   report.mkdir(parents=True)
   (report/"generation-config.json").write_text(json.dumps({
    "model_version_id":"fairy-run:base","protocol_id":"ifeval-v1","backend":"hf_qat"}))
   (report/"generation-progress.json").write_text(json.dumps({"done":6,"total":32}))
   with patch.object(d,"ROOT",root):
    results=d.benchmark_report("fairy-run",run)
   self.assertEqual(results[0]["summary"],{})
   self.assertEqual(results[0]["progress"],{"done":6,"total":32})
if __name__=="__main__":unittest.main()
