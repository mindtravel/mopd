import tempfile,unittest,torch
from pathlib import Path
from transformers import LlamaConfig,LlamaForCausalLM
from peft import get_peft_model_state_dict
from mopd.backends.student_adapter import attach_student_adapter
class Tests(unittest.TestCase):
 def base(self):
  return LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
 def test_continue_existing_weights_trainable(self):
  a=attach_student_adapter(self.base(),{})
  with torch.no_grad():
   for n,p in a.named_parameters():
    if "lora_" in n:p.fill_(0.125)
  with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"logs") as tmp:
   a.save_pretrained(tmp)
   b=attach_student_adapter(self.base(),{"student_adapter":tmp})
   expected=get_peft_model_state_dict(a);actual=get_peft_model_state_dict(b)
   self.assertEqual(set(expected),set(actual))
   for k in expected:self.assertTrue(torch.equal(expected[k],actual[k]),k)
   self.assertTrue(all(p.requires_grad for n,p in b.named_parameters() if "lora_" in n))
 def test_missing_checkpoint_fails(self):
  with self.assertRaises((ValueError,OSError)):
   attach_student_adapter(self.base(),{"student_adapter":"/ssd/lzx/mopd/not-a-checkpoint"})
if __name__=="__main__":unittest.main()
