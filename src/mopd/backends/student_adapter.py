"""Create a fresh adapter or continue saved student weights (new optimizer stage)."""
from pathlib import Path
from peft import LoraConfig,PeftModel,get_peft_model
def attach_student_adapter(student,config):
    checkpoint=config.get("student_adapter")
    if checkpoint:
        path=Path(checkpoint)
        if not (path/"adapter_config.json").is_file() or not (path/"adapter_model.safetensors").is_file():
            raise ValueError("Incomplete student adapter: "+str(path))
        return PeftModel.from_pretrained(student,str(path),is_trainable=True,local_files_only=True)
    return get_peft_model(student,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM"))
