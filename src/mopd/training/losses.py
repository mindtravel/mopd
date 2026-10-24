"""Sampled reverse-KL policy gradient on response tokens."""

import torch


def sampled_reverse_kl(
    student_logits, teacher_response_logits, input_ids, response_mask, response_count, advantage_clip=5.0
):
    mask = response_mask[:, 1:].bool()
    if not mask.any():
        raise ValueError("No response tokens")
    targets = input_ids[:, 1:][mask]
    slog = student_logits[:, :-1][mask].float().log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
    if response_count != int(targets.numel()):
        raise ValueError(f"Teacher/student response length mismatch: {response_count} vs {targets.numel()}")
    with torch.no_grad():
        tlog = (
            teacher_response_logits[0, :response_count].float().log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
        )
        advantage = (tlog - slog.detach()).clamp(-advantage_clip, advantage_clip)
    loss = -(advantage * slog).mean()
    return loss, {
        "tokens": int(mask.sum()),
        "sampled_log_ratio": float((slog.detach() - tlog).mean()),
        "advantage_abs": float(advantage.abs().mean()),
    }
