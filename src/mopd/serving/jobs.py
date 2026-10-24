"""Job records, one JSON file each under runs/bench-service/jobs; a restart re-queues unfinished jobs.
A job has one part per benchmark (state, scores, error) and job-level engine/utilization fields."""

import json
import os
import threading
import time
import uuid

from mopd.serving import protocol


class Jobs:
    def __init__(self, state_dir):
        self.dir = state_dir / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock, self.queue, self.jobs, self.waiting = threading.Lock(), [], {}, {}
        for f in sorted(self.dir.glob("*.json")):
            j = json.loads(f.read_text())
            self.jobs[j["id"]] = j
            if any(p["state"] in ("queued", "running") for p in j.get("parts", {}).values()):
                for p in j["parts"].values():
                    if p["state"] != "complete":
                        p.update(state="queued", note="re-queued after service restart")
                self.queue.append(j["id"])
            self.refresh(j)
            self.save(j)

    def save(self, j):
        tmp = self.dir / f"{j['id']}.json.tmp"
        tmp.write_text(json.dumps(j, indent=1, ensure_ascii=False))
        os.replace(tmp, self.dir / f"{j['id']}.json")

    def submit(self, req):
        fields = protocol.request(req)
        j = {
            "id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            **fields,
            "protocol": {"max_model_len": protocol.MAX_MODEL_LEN, "ifbench": protocol.IFBENCH,
                         "automationbench": protocol.AUTOMATIONBENCH},
            "parts": {b: {"state": "queued"} for b in fields["benchmarks"]},
            "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.lock:
            self.jobs[j["id"]] = j
            self.queue.append(j["id"])
            self.refresh(j)
            self.save(j)
        return j

    def take(self):
        with self.lock:
            return self.jobs[self.queue.pop(0)] if self.queue else None

    @staticmethod
    def refresh(j):
        """Job-level state, scores and errors from its parts."""
        states = {p["state"] for p in j["parts"].values()}
        if states <= {"complete", "failed"}:
            j["state"] = "failed" if "failed" in states else "complete"
        else:
            j["state"] = "running" if states & {"running", "complete"} else "queued"
        j["scores"] = {b: p["scores"] for b, p in j["parts"].items() if p.get("scores")}
        j["errors"] = {b: p["error"] for b, p in j["parts"].items() if p.get("error")} or None

    def update(self, j, bench=None, **kw):
        """Fields of one benchmark's part, or of the job itself when bench is None."""
        with self.lock:
            (j["parts"][bench] if bench else j).update(kw)
            self.refresh(j)
            self.save(j)
