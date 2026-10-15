"""Rollout backend selection by config name."""
from mopd.rollout.hf_batch import generate
from mopd.rollout.processes import ProcessRollouts
from mopd.rollout.replicas import RolloutReplicas

BACKENDS = ("vllm", "replicas", "processes", "hf")


def build_rollout(config, student):
    """Return a backend with generate(student, rows, version) -> (trajectories, seconds, stats).

    The `hf` backend returns None: the caller generates in-process on the student GPU.
    """
    backend = config.get("rollout_backend", "replicas")
    if backend == "hf":
        return None
    if backend == "vllm":
        from mopd.rollout.vllm_processes import VllmRollouts
        return VllmRollouts(config, student)
    factories = {"replicas": RolloutReplicas, "processes": ProcessRollouts}
    if backend not in factories:
        raise ValueError(f"Unknown rollout_backend {backend!r}; expected one of {BACKENDS}")
    return factories[backend](config, student)


__all__ = ["BACKENDS", "build_rollout", "generate"]
