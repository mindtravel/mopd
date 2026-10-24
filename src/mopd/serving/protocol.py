"""The evaluation protocol the service runs: fixed, so every checkpoint's scores are comparable.
A request chooses only which benchmarks, subset sizes and AutomationBench domains. The model's own
chat template decides the parsers and whether it thinks (settings())."""

import json
from pathlib import Path

BENCHMARKS = ("ifbench", "automationbench")
MAX_MODEL_LEN = 32768
IFBENCH = {"max_tokens": 4096, "workers": 64, "temperature": 0}  # 64: the upstream wrapper's ceiling
THINKING_MAX_TOKENS = 16384  # a thinking model's answer comes after its reasoning; 4096 would cut many off
AUTOMATIONBENCH = {"max_steps": 50, "max_concurrent": 8, "toolset": "api"}
DOMAINS = ("simple", "sales", "marketing", "operations", "support", "finance", "hr")


def request(req):
    """Validated job fields from a POST /jobs body:
    {"model": "/abs/path", "name"?, "benchmarks"?, "ifbench_limit"?, "auto_domains"? ("simple" | "all" | "sales,hr"),
     "auto_limit"?}"""
    model = Path(req["model"])
    if not model.is_absolute() or not ((model / "config.json").exists() or (model / "adapter_config.json").exists()):
        raise ValueError(f"{model}: not a model or LoRA adapter directory on this machine")
    benches = list(req.get("benchmarks") or BENCHMARKS)
    if set(benches) - set(BENCHMARKS):
        raise ValueError(f"benchmarks must be a subset of {BENCHMARKS}")
    domains = req.get("auto_domains") or "simple"
    if domains != "all" and set(domains.split(",")) - set(DOMAINS):
        raise ValueError(f"auto_domains: 'all' or a comma list of {DOMAINS}")
    limits = {k: req.get(k) for k in ("ifbench_limit", "auto_limit")}
    if any(v is not None and (not isinstance(v, int) or v < 1) for v in limits.values()):
        raise ValueError("limits must be positive integers")
    return {"name": req.get("name") or model.name, "model": str(model), "benchmarks": benches,
            "auto_domains": domains, **limits}


def chat_template(model):
    """The template the server will apply: the checkpoint's own (chat_template.jinja or
    tokenizer_config.json), or its LoRA base's."""
    model = Path(model)
    if (model / "adapter_config.json").exists() and not (model / "tokenizer_config.json").exists():
        model = Path(json.loads((model / "adapter_config.json").read_text())["base_model_name_or_path"])
    if (model / "chat_template.jinja").exists():
        return (model / "chat_template.jinja").read_text()
    cfg = model / "tokenizer_config.json"
    return json.loads(cfg.read_text()).get("chat_template", "") if cfg.exists() else ""


def settings(model):
    """Effective protocol for a checkpoint: tool-call format (Qwen3.5 XML <function=...> vs Qwen3
    JSON hermes), thinking (template has enable_thinking: reasoning separated from the scored answer,
    longer IFBench budget)."""
    t = chat_template(model)
    thinking = "enable_thinking" in t
    return {"max_model_len": MAX_MODEL_LEN, "thinking": thinking,
            "tool_parser": "qwen3_coder" if "<function=" in t else "hermes",
            "ifbench": {**IFBENCH, **({"max_tokens": THINKING_MAX_TOKENS} if thinking else {})},
            "automationbench": AUTOMATIONBENCH}
