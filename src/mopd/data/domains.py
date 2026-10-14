"""Deprecated import path; the protocol module is authoritative. Kept for compatibility."""
from mopd.protocol import (  # noqa: F401
    PLAIN_QA_HEADERS,
    STUDENT_PROMPT_MODES,
    student_prompt,
    teacher_content,
)

__all__ = ["PLAIN_QA_HEADERS", "STUDENT_PROMPT_MODES", "student_prompt", "teacher_content"]
