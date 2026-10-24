"""Near-duplicate detection between training text and benchmark problems.

Shared 13-grams alone are mostly boilerplate (number runs, "The input is given from Standard
Input", LeetCode's subsequence wording). A near-duplicate is a single training text that covers a
large share of one benchmark problem's 13-grams. Both directions use this measure:
  - benchmark build: which benchmark problems are near-duplicated by some training prompt;
  - training subset build: drop training rows that near-duplicate any benchmark problem.
"""

import re

N = 13
THRESHOLD = 0.3


def ngrams(text, n=N):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


class BenchmarkIndex:
    """13-gram index over benchmark problems: {problem id: text}."""

    def __init__(self, problems):
        self.grams = {pid: ngrams(text) for pid, text in problems.items()}
        self.where = {}
        for pid, gs in self.grams.items():
            for g in gs:
                self.where.setdefault(g, []).append(pid)

    def coverage(self, text):
        """{problem id: share of its 13-grams found in `text`} for problems sharing any 13-gram."""
        shared = {}
        for g in ngrams(text) & self.where.keys():
            for pid in self.where[g]:
                shared[pid] = shared.get(pid, 0) + 1
        return {pid: c / len(self.grams[pid]) for pid, c in shared.items()}

    def worst(self, text):
        """(problem id, coverage) of the most covered problem, or (None, 0.0)."""
        cov = self.coverage(text)
        return max(cov.items(), key=lambda kv: kv[1]) if cov else (None, 0.0)

    def max_coverage(self, texts):
        """Per problem, the largest coverage by any single text of `texts`."""
        best = dict.fromkeys(self.grams, 0.0)
        for text in texts:
            for pid, c in self.coverage(text).items():
                best[pid] = max(best[pid], c)
        return best
