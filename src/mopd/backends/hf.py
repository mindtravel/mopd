from transformers import AutoModelForCausalLM, AutoTokenizer
from mopd.backends.student_adapter import attach_student_adapter
from mopd.backends.fairy2i_qat_adapter import load_fairy2i_qat_student
import torch


def load_pair(config):
    teacher_tok = AutoTokenizer.from_pretrained(config["teacher"], local_files_only=True)
    if config.get("student_backend") == "fairy2i_qat":
        student_tok, student = load_fairy2i_qat_student(
            config["student"],
            config["student_device"],
            trainable_scales=True,
            update_path=config.get("student_adapter"),
        )
    elif config.get("student_backend", "hf_lora") in {"hf", "hf_lora"}:
        student_tok = AutoTokenizer.from_pretrained(config["student"], local_files_only=True)
        student = AutoModelForCausalLM.from_pretrained(
            config["student"], torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        ).to(config["student_device"])
        student = attach_student_adapter(student, config)
    else:
        raise ValueError("Unknown student_backend: " + str(config.get("student_backend")))
    if teacher_tok.get_vocab() != student_tok.get_vocab():
        raise ValueError("Teacher/student vocabulary mismatch")
    if teacher_tok.pad_token_id is None:
        teacher_tok.pad_token = teacher_tok.eos_token
    teacher = (
        AutoModelForCausalLM.from_pretrained(
            config["teacher"], torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .to(config["teacher_device"])
        .eval()
    )
    teacher.requires_grad_(False)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return teacher_tok, student, teacher
