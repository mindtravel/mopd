import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import json, subprocess, tempfile, time, unittest

from run_queue import filtered_plan, running


class QueueTest(unittest.TestCase):
    def test_filtered_plan_keeps_matches_and_their_dependencies(self):
        tmp = Path(tempfile.mkdtemp())
        plan = {
            "jobs": [
                {"id": "train:a", "deps": []},
                {"id": "eval:a:1", "deps": ["train:a"]},
                {"id": "train:b", "deps": []},
                {"id": "eval:b:1", "deps": ["train:b"]},
            ]
        }
        (tmp / "plan.json").write_text(json.dumps(plan))
        out = json.loads(Path(filtered_plan(tmp / "plan.json", ["eval:a:"], tmp / "out.json")).read_text())
        self.assertEqual(sorted(j["id"] for j in out["jobs"]), ["eval:a:1", "train:a"])

    def test_running_finds_other_processes_but_never_itself(self):
        tag = f"mopd-queue-test-{time.time_ns()}"
        self.assertEqual(running(tag), [])  # our own command line contains the tag only via this call's argument
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)", tag])
        try:
            self.assertEqual(running(tag), [p.pid])
        finally:
            p.kill()


if __name__ == "__main__":
    unittest.main()
