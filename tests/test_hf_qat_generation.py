import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_benchmark_hf_qat import emit_progress


class ProgressOutputTests(unittest.TestCase):
    def test_progress_is_one_json_line_and_flushes(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            emit_progress({"done": 2, "total": 32})

        self.assertEqual(json.loads(output.getvalue()), {"done": 2, "total": 32})
        self.assertTrue(output.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
