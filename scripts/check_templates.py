import os,json,torch
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
r=Path(os.environ['MOPD_ROOT']);tok=AutoTokenizer.from_pretrained(r/'models/Qwen3-4B');out=[]
question='What is 17 + 24? Give only the number.'
templates={'plain':f'Question: {question}\nAnswer:', 'fewshot':'Question: What is 12 + 15? Give only the number.\nAnswer: 27\n\nQuestion: What is 21 + 13? Give only the number.\nAnswer: 34\n\n'+f'Question: {question}\nAnswer:', 'chat':tok.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True,enable_thinking=False)}
for name in ['Qwen3-4B-Base','Qwen3-4B']:
 model=AutoModelForCausalLM.from_pretrained(r/'models'/name,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 for kind,text in templates.items():
  inp=tok(text,return_tensors='pt').to('cuda')
  with torch.no_grad(): ids=model.generate(**inp,do_sample=False,max_new_tokens=48,pad_token_id=151643,eos_token_id=[151643,151645])
  row={'model':name,'template':kind,'input':text,'input_ids':inp.input_ids.tolist(),'output':tok.decode(ids[0,inp.input_ids.shape[1]:],skip_special_tokens=False)};out.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
 del model;torch.cuda.empty_cache()
(r/'logs/template-diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
