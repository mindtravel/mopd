"""Run artifacts written outside the training loop's hot path."""
import hashlib
import json
from pathlib import Path

from mopd.runtime.status import write_status  # noqa: F401


def dump(path, value):
    """Write a JSON artifact atomically enough for readers polling the run directory."""
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def fingerprint(model):
    """Content hash of all parameters; used to prove frozen teachers never moved."""
    import torch

    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
