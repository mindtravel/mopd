"""Deprecated import path; the protocol module is authoritative. Kept for compatibility."""

from mopd.protocol import (  # noqa: F401
    STUDENT_PROMPT_MODES,
    apply_chat_template_ids,
    apply_chat_template_text,
)

__all__ = ["STUDENT_PROMPT_MODES", "apply_chat_template_ids", "apply_chat_template_text"]
