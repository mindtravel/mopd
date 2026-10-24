"""Exact selected-token log probabilities without materializing sequence x vocabulary logits."""

import torch
from torch.utils.checkpoint import checkpoint


def selected_logps(model, ids, start, chunk_size=128):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state
    parts = []

    def head_loss(h, targets):
        logits = base.lm_head(h).float()
        return -torch.nn.functional.cross_entropy(logits, targets, reduction="none")

    for pos in range(start, ids.shape[1], chunk_size):
        end = min(pos + chunk_size, ids.shape[1])
        h = hidden[0, pos - 1 : end - 1]
        targets = ids[0, pos:end]
        if torch.is_grad_enabled():
            parts.append(checkpoint(head_loss, h, targets, use_reentrant=False))
        else:
            parts.append(head_loss(h, targets))
    return torch.cat(parts)
