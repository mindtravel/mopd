"""Masked batches with identical action IDs and sample-mean losses."""

import torch
from torch.utils.checkpoint import checkpoint
from mopd.protocol import (  # noqa: F401
    generation_eos_ids,
    is_eos_token,
    normalize_eos_token_ids,
    trim_generated,
)


def pack(sequences, starts, pad, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    response = torch.zeros_like(ids, dtype=torch.bool)
    for i, (seq, start) in enumerate(zip(sequences, starts)):
        n = len(seq)
        ids[i, :n] = torch.as_tensor(seq, device=device)
        attention[i, :n] = 1
        response[i, start:n] = True
    return ids, attention, response


# Response tokens per lm_head call. The chunk bounds the fp32 logits kept alive (2048 x 151936 x 4 B
# = 1.2 GB) while keeping the vocabulary matmul large: at 128 the projection ran as dozens of small,
# checkpointed matmuls per micro-batch and showed up as a visible share of step time.
HEAD_CHUNK = 2048


class TokenLogProbs(torch.nn.Module):
    """Per-token log-probs of the response tokens, as one module forward. The chunked lm_head (never
    materializes [batch, seq, vocab] logits) goes through the inner decoder and lm_head directly;
    keeping all of it inside this forward is what lets FSDP wrap this module as its root unit --
    FSDP unshards parameters and schedules gradient reduction from module forward hooks, so code
    that called base.model / base.lm_head from outside any wrapped forward would bypass both."""

    def __init__(self, model):
        super().__init__()
        self.base = model.get_base_model() if hasattr(model, "get_base_model") else model

    def forward(self, ids, attention, response):
        base = self.base
        positions = (attention.cumsum(-1) - 1).clamp_min(0)
        hidden = base.model(
            input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False
        ).last_hidden_state
        mask = response[:, 1:] & attention[:, 1:].bool()
        hs = hidden[:, :-1][mask]
        targets = ids[:, 1:][mask]

        def head(h, y):
            return -torch.nn.functional.cross_entropy(base.lm_head(h).float(), y, reduction="none")

        values = []
        for i in range(0, len(targets), HEAD_CHUNK):
            h = hs[i : i + HEAD_CHUNK]
            y = targets[i : i + HEAD_CHUNK]
            values.append(checkpoint(head, h, y, use_reentrant=False) if torch.is_grad_enabled() else head(h, y))
        return torch.cat(values)


def logps(model, ids, attention, response):
    """model: a (PEFT-wrapped) causal LM, or a TokenLogProbs (e.g. the FSDP-sharded one)."""
    scorer = model if isinstance(model, TokenLogProbs) else TokenLogProbs(model)
    mask = response[:, 1:] & attention[:, 1:].bool()
    return list(scorer(ids, attention, response).split(mask.sum(1).tolist()))


def trim_generated(tokens, eos):
    values = tokens.tolist()
    eos_ids = set(normalize_eos_token_ids(eos))
    for index, token in enumerate(values):
        if token in eos_ids:
            return values[: index + 1]
    return values
