"""The process the pool launches for one job: its commands in order from paths.ROOT, appended to its
log, "{gpus}" expanded to this job's GPU ids as it sees them (0..n-1 under CUDA_VISIBLE_DEVICES).
Exits with the first failing command's code.

  python -m mopd.pool.run <job.json>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from mopd import paths


def expand(argv, n):
    ids = [str(i) for i in range(n)]
    out = []
    for a in argv:
        if a == "{gpus}":
            out += ids
        else:
            out.append(str(a).replace("{ngpus}", str(n)))
    return out


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n = len([g for g in visible.split(",") if g]) if visible else 0
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for argv, extra in job["commands"]:
            argv = expand(argv, n)
            f.write(f"\n$ [{job['id']} on GPUs {visible or '-'}] {' '.join(argv)}\n")
            f.flush()
            rc = subprocess.run(argv, cwd=paths.ROOT, env={**os.environ, **extra}, stdout=f, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                sys.exit(rc)


if __name__ == "__main__":
    main()
