"""Which GPUs are actually idle right now, via nvidia-smi -- not a hardcoded allow/deny list.

This server is shared with other teams (e.g. another group's long-running vllm serve has been
seen holding a GPU with no warning in our own config). A static "exclude GPU N" convention only
protects against that one GPU, on that one day; anything we schedule, including ad-hoc manual
commands, should check real occupancy immediately before claiming a device, the same way
run_experiment.py already does for its own preflight.
"""

import subprocess


def gpu_memory_used(gpus=None):
    """{gpu_id: MiB used}, for every GPU or just `gpus` if given."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    used = {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}
    return {g: used[g] for g in gpus} if gpus is not None else used


def idle_gpus(candidates, threshold_mib=1000):
    """candidates with less than threshold_mib used by anyone (us or another team)."""
    used = gpu_memory_used(candidates)
    return [g for g in candidates if used.get(g, 0) < threshold_mib]
