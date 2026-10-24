"""LiveCodeBench grading without a GPU: output matching, stdin and functional execution, test decoding."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "scripts")]
import base64, gzip, hashlib, json, pickle, tempfile, unittest, zlib
from pathlib import Path

from mopd.evaluation import livecodebench as lcb


def problem(root, tests, **meta):
    path = Path(root) / "t.json.gz"
    path.write_bytes(gzip.compress(json.dumps(tests).encode()))
    return {"domain": "code", "metadata": {"benchmark": "livecodebench", "tests_path": "t.json.gz",
                                           "tests_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **meta}}


class MatchTest(unittest.TestCase):
    def test_lines_tokens_and_decimals(self):
        self.assertTrue(lcb.outputs_match("1 2\n3\n\n", "1 2\n3"))
        self.assertTrue(lcb.outputs_match("0.50", "0.5"))
        self.assertTrue(lcb.outputs_match("1 2 3", "1\n2 3"))
        self.assertFalse(lcb.outputs_match("1 2", "1 3"))
        self.assertFalse(lcb.outputs_match("Yes", "No"))


class GradeTest(unittest.TestCase):
    def test_stdin(self):
        with tempfile.TemporaryDirectory() as root:
            row = problem(root, [{"input": "2 3\n", "output": "5\n", "testtype": "stdin"},
                                 {"input": "10 -4\n", "output": "6\n", "testtype": "stdin"}], testtype="stdin")
            good = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
            bad = "```python\na, b = map(int, input().split())\nprint(a * b)\n```"
            self.assertTrue(lcb.grade(row, good, root)["correct"])
            self.assertFalse(lcb.grade(row, bad, root)["correct"])

    def test_functional_and_lenient(self):
        with tempfile.TemporaryDirectory() as root:
            row = problem(root, [{"input": "[1,2]\n3", "output": "[1, 2, 3]", "testtype": "functional"}],
                          testtype="functional", func_name="push")
            code = "class Solution:\n    def push(self, xs: List[int], x: int) -> List[int]:\n        return tuple(xs + [x])\n"
            self.assertTrue(lcb.grade(row, f"```python\n{code}```", root)["correct"])
            unfenced = lcb.grade(row, " " + code.replace("\n", "\n "), root)
            self.assertFalse(unfenced["correct"])
            self.assertTrue(unfenced["correct_lenient"])

    def test_timeout_and_digest(self):
        with tempfile.TemporaryDirectory() as root:
            row = problem(root, [{"input": "", "output": "1", "testtype": "stdin"}], testtype="stdin")
            self.assertFalse(lcb.grade(row, "```python\nwhile True: pass\n```", root)["correct"])
            row["metadata"]["tests_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                lcb.grade(row, "```python\nprint(1)\n```", root)


class DecodeTest(unittest.TestCase):
    def test_plain_string_payload_only(self):
        blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps([{"input": "1", "output": "1"}]))))
        self.assertEqual(lcb.decode_private_tests(blob)[0]["output"], "1")
        evil = base64.b64encode(zlib.compress(pickle.dumps(Path("x"))))
        with self.assertRaises(pickle.UnpicklingError):
            lcb.decode_private_tests(evil)


if __name__ == "__main__":
    unittest.main()
