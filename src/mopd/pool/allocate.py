"""The pool's matching decision, a pure function of the GPU state and the jobs (tested without GPUs).

Each tick, ready jobs are considered in priority order (higher first; equal priority by submission):
  - a job runs when its whole shape fits the free GPUs at once (gang): `count` GPUs, or for an
    elastic job (`min`..`max`, e.g. an evaluation that spreads rows over its GPUs) as many as are
    free up to `max`;
  - a job that does not fit may preempt running preemptible jobs of lower priority whose GPUs
    would make it fit; they are stopped and requeued (they resume: evaluations continue from their
    written rows), and the job starts on a later tick;
  - once a non-preemptible job waits, lower-priority jobs may still start on the free GPUs only if
    they are preemptible (backfill): they never delay it, since it can take their GPUs back;
  - lent GPUs (the bench service's while it is idle) go to preemptible jobs only: the service can
    recall them at any time.
"""


def shape(job):
    """(min, max) GPUs of a job's request."""
    g = job.get("gpus", 0)
    if isinstance(g, int):
        return g, g
    if "count" in g:
        return g["count"], g["count"]
    return g.get("min", 1), g.get("max", 10**6)


def decide(free, running, ready, lent=()):
    """free: [gpu ids]; running: {job id: {"gpus": [...], "priority", "preemptible"}};
    ready: [jobs in submission order, each {"id", "gpus", "priority", "preemptible"}];
    lent: free GPUs that only preemptible jobs may take.
    Returns (starts {job id: [gpus]}, preempts [job ids])."""
    free = list(free)
    lent = set(lent)
    starts, preempts = {}, []
    blocked = False  # a non-preemptible job of higher priority is waiting
    order = sorted(ready, key=lambda j: -j.get("priority", 0))  # stable: submission order within a priority
    for job in order:
        lo, hi = shape(job)
        if lo == 0:  # CPU job
            starts[job["id"]] = []
            continue
        if blocked and not job.get("preemptible"):
            continue
        usable = free if job.get("preemptible") else [g for g in free if g not in lent]
        if len(usable) >= lo:
            take = usable[: min(hi, len(usable))]
            starts[job["id"]] = take
            free = [g for g in free if g not in take]
            continue
        # preempt lower-priority preemptible jobs (cheapest first: fewest GPUs) if that makes it fit
        victims, gain = [], 0
        for jid, r in sorted(running.items(), key=lambda kv: (kv[1].get("priority", 0), len(kv[1]["gpus"]))):
            if jid in preempts or not r.get("preemptible") or r.get("priority", 0) >= job.get("priority", 0):
                continue
            victims.append(jid)
            gain += len([g for g in r["gpus"] if job.get("preemptible") or g not in lent])  # lent GPUs do not help it
            if len(usable) + gain >= lo:
                break
        if victims and len(usable) + gain >= lo:
            preempts += victims
        if not job.get("preemptible"):
            blocked = True
    return starts, preempts
