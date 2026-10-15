"""Gates that must pass before a run is allowed to consume GPUs.

The 2026-10-14 incident: a warm-start run rolled out a plain-qa adapter through the
chat template, and 23 of 32 first-step rollouts collapsed into repeated garbage.
The format check below would have refused to start; the rollout gate catches the
class of failure that a format check alone cannot.
"""
import unicodedata

RESERVED_GPUS = ("cuda:1",)

# Scripts whose presence in a response, but not in the prompt, indicates decoding collapse.
_SUSPECT_SCRIPTS = ("HANGUL", "HEBREW", "ARABIC", "THAI", "CYRILLIC", "DEVANAGARI", "EGYPTIAN")
_EXTENDED_LATIN = set("łżźśćńęąşğıẩếệộờữ")


def script_set(text):
    """Which writing systems appear in `text`, so prompt language can be subtracted."""
    found = set()
    for char in text:
        if char in _EXTENDED_LATIN:
            found.add("LATIN-EXT")
        elif ord(char) > 127:
            try:
                name = unicodedata.name(char)
            except ValueError:
                found.add("UNNAMED")
                continue
            found.update(script for script in _SUSPECT_SCRIPTS if name.startswith(script))
    return found


def foreign_script(text, prompt=""):
    return bool(script_set(text) - script_set(prompt))


def check_devices(config, reserved=RESERVED_GPUS):
    devices = {config["student_device"], *config.get("rollout_devices", []),
               *(spec["device"] for spec in config["domains"].values())}
    clash = sorted(set(reserved) & devices)
    if clash:
        raise ValueError("Reserved GPU in config: " + ", ".join(clash))


def check_rollout_gate(rollouts, max_degenerate=0.1, min_rows=8):
    """Reject a run whose sampled answers are mostly collapsed; returns the measured rate.

    Intended to run on the first step, where the student still equals the warm-start
    weights and any collapse is a protocol bug rather than an effect of training.
    """
    if len(rollouts) < min_rows:
        raise ValueError(f"Rollout gate needs at least {min_rows} rows, got {len(rollouts)}")
    degenerate = [r for r in rollouts
                  if r.get("truncated") or foreign_script(r["response"], r.get("prompt", ""))]
    rate = len(degenerate) / len(rollouts)
    if rate > max_degenerate:
        example = degenerate[0]
        raise ValueError(
            f"Rollout gate failed: {len(degenerate)}/{len(rollouts)} ({rate:.0%}) answers are "
            f"truncated or contain foreign script. First: id={example.get('id')} "
            f"response={example['response'][:120]!r}")
    return rate
