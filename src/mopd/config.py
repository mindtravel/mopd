"""Experiment configuration: one schema, one place that rejects an invalid run.

The config is the contract between the launcher, the training loop and the
model-version record. Keeping validation here means a new backend, domain or
prompt format is added in one file instead of being re-checked at each call site.
"""

from mopd import paths
import json
from pathlib import Path

from mopd.data.mixture import resolve_domain
from mopd.protocol import STUDENT_PROMPT_MODES, TEACHER_PROMPT_FORMATS

STUDENT_BACKENDS = ("hf_lora", "hf_full", "fairy2i_qat")
ROLLOUT_BACKENDS = ("vllm", "replicas", "processes", "hf")
# rollout backends that need their own GPUs, mapped to their config key.
ROLLOUT_DEVICE_KEYS = {"vllm": "rollout_devices", "replicas": "rollout_devices", "processes": "rollout_devices"}


class ConfigError(ValueError):
    """Raised for a configuration that cannot run, with the offending field named."""


def _require(config, key, kind, where=""):
    value = config.get(key)
    if value is None:
        raise ConfigError(f"{where}missing required field {key!r}")
    if not isinstance(value, kind) or isinstance(value, bool) and kind is int:
        raise ConfigError(f"{where}{key!r} must be {getattr(kind, '__name__', kind)}")
    return value


def _student_devices(config):
    """A sharded (hf_full) student runs one torchrun rank per GPU on cuda:0..student_gpus-1."""
    if config["student_backend"] == "hf_full":
        n = _require(config, "student_gpus", int)
        # fp32 weights + grads + AdamW = 16 bytes/parameter: ~64 GB for 4B, too much for fewer than 4
        # 32 GB GPUs unless it lives in host memory (fsdp.offload)
        if n < 4 and not config.get("fsdp", {}).get("offload"):
            raise ConfigError("hf_full needs student_gpus >= 4, or fsdp.offload (optimizer state in host memory)")
        return {f"cuda:{i}" for i in range(n)}
    return {config["student_device"]}


def _check_devices(config):
    # No GPU is reserved by convention (the server is shared; see AGENTS.md): only collisions
    # inside this run are rejected. vLLM claims a fixed share of its GPU's memory up front, so a
    # rollout GPU cannot also hold the student or a teacher.
    rollout = set(config.get("rollout_devices", []))
    taken = _student_devices(config) | {spec["device"] for spec in config["domains"].values()}
    if rollout & taken:
        raise ConfigError(f"rollout devices {sorted(rollout & taken)} also hold the student or a teacher")
    teachers_on_student = _student_devices(config) & {spec["device"] for spec in config["domains"].values()}
    # with fsdp.offload the student's state lives in host memory, leaving room for teachers on its GPUs
    if config["student_backend"] == "hf_full" and teachers_on_student and not config.get("fsdp", {}).get("offload"):
        raise ConfigError(f"teachers {sorted(teachers_on_student)} share a GPU with a student shard (needs fsdp.offload)")


def _check_prompt_protocol(config, adapter_config):
    prompt_format = config.get("student_prompt_format")
    if prompt_format not in STUDENT_PROMPT_MODES:
        raise ConfigError(
            "student_prompt_format must be one of " + ", ".join(STUDENT_PROMPT_MODES) + f", got {prompt_format!r}"
        )
    adapter = config.get("student_adapter")
    if not adapter:
        return
    if adapter_config is None:
        raise ConfigError(
            f"student_adapter {adapter!r} has no sibling config.json, so the format it was trained "
            "with is unknown; pass student_adapter_prompt_format explicitly"
        )
    # A parent run only records its format once it has been migrated (see protocol.py).
    trained = adapter_config.get("student_prompt_format", config.get("student_adapter_prompt_format"))
    if trained != prompt_format:
        raise ConfigError(
            f"Adapter {adapter!r} was trained with student prompt {trained!r}, "
            f"but this config sets student_prompt_format={prompt_format!r}. "
            "An adapter must be rolled out with the format it was trained on."
        )


def _check_domains(config, project_root):
    domains = _require(config, "domains", dict)
    if not domains:
        raise ConfigError("domains must not be empty")
    weights = 0.0
    for name, spec in domains.items():
        where = f"domains.{name}: "
        for key in ("teacher", "device"):
            _require(spec, key, str, where)
        teacher_format = spec.get("teacher_prompt_format", "chat_template_v1")
        if teacher_format not in TEACHER_PROMPT_FORMATS:
            raise ConfigError(
                f"{where}teacher_prompt_format must be one of {TEACHER_PROMPT_FORMATS}, got {teacher_format!r}"
            )
        if not isinstance(spec.get("count"), int) or spec["count"] < 1:
            raise ConfigError(f"{where}count must be a positive integer")
        try:
            resolve_domain(name, spec, project_root or str(paths.ROOT))
        except ValueError as error:
            raise ConfigError(f"{where}{error}") from error
        for source in spec["sources"]:
            if not Path(source["path"]).is_file():
                raise ConfigError(f"{where}source {source['name']!r} does not exist: {source['path']}")
        for path in spec.get("holdout_files", []):
            if not Path(path).is_file():
                raise ConfigError(f"{where}holdout file does not exist: {path}")
        if not isinstance(spec.get("weight"), (int, float)) or spec["weight"] <= 0:
            raise ConfigError(f"{where}weight must be positive")
        weights += spec["weight"]
    if abs(weights - 1.0) > 1e-6:
        raise ConfigError(f"domain weights must sum to 1.0, got {weights!r}")


def load(path, project_root=str(paths.ROOT)):
    """Read and validate a run config; every later stage may assume it is coherent."""
    config = json.loads(Path(path).read_text())
    root = Path(project_root)

    _require(config, "student", str)
    for key in ("student_device", "output"):
        _require(config, key, str)
    for flag in ("steps", "max_new_tokens", "generation_batch_size", "micro_batch_size", "save_every"):
        value = _require(config, flag, int)
        if value < 1:
            raise ConfigError(f"{flag} must be a positive integer")
    lr = config.get("lr")
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ConfigError("lr must be a positive number")

    backend = config.setdefault("student_backend", "hf_lora")
    if backend not in STUDENT_BACKENDS:
        raise ConfigError(f"student_backend must be one of {STUDENT_BACKENDS}, got {backend!r}")
    rollout = config.setdefault("rollout_backend", "replicas")
    if backend == "hf_full" and rollout != "vllm":
        raise ConfigError("hf_full students need rollout_backend 'vllm' (in-place full-weight updates)")
    if rollout not in ROLLOUT_BACKENDS:
        raise ConfigError(f"rollout_backend must be one of {ROLLOUT_BACKENDS}, got {rollout!r}")
    if rollout in ROLLOUT_DEVICE_KEYS:
        if not config.get(ROLLOUT_DEVICE_KEYS[rollout]):
            raise ConfigError(f"rollout_backend {rollout!r} requires {ROLLOUT_DEVICE_KEYS[rollout]!r}")
    if rollout == "vllm":
        utilization = config.get("vllm_memory_utilization", 0.8)
        if not 0 < utilization <= 1:
            raise ConfigError(f"vllm_memory_utilization must be in (0, 1], got {utilization!r}")

    output = Path(config["output"])
    if not output.is_absolute():
        raise ConfigError("output must be an absolute path")
    if project_root is not None and not output.is_relative_to(root):
        raise ConfigError(f"output must stay inside {root}, got {output}")

    _check_domains(config, project_root)
    _check_devices(config)

    adapter = config.get("student_adapter")
    adapter_config = None
    if adapter:
        path = Path(adapter)
        if not (path / "adapter_config.json").is_file() or not (path / "adapter_model.safetensors").is_file():
            raise ConfigError(f"incomplete student adapter: {path}")
        sibling = path.parent / "config.json"
        adapter_config = json.loads(sibling.read_text()) if sibling.is_file() else None
    _check_prompt_protocol(config, adapter_config)
    return config
