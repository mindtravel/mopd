"""Resumable benchmark generation only; grading runs in the training environment."""
import argparse,fcntl,hashlib,json,os,time
from pathlib import Path
from mopd.evaluation.generation_manifest import build_generation_manifest
from mopd.data.templates import STUDENT_PROMPT_MODES
def main():
 p=argparse.ArgumentParser()
 for name in ("model","data","out"):p.add_argument("--"+name,required=True)
 p.add_argument("--device",type=int,default=3);p.add_argument("--adapter");p.add_argument("--teacher",action="store_true")
 p.add_argument("--model-version-id");p.add_argument("--protocol-id")
 p.add_argument("--max-new-tokens",type=int,default=8192);p.add_argument("--max-seqs",type=int,default=32);p.add_argument("--batch-size",type=int,default=16)
 p.add_argument("--student-prompt-format",choices=sorted(STUDENT_PROMPT_MODES))
 a=p.parse_args()
 if not a.teacher and not a.student_prompt_format:p.error("--student-prompt-format is required for student generation")
 os.environ["CUDA_VISIBLE_DEVICES"]=str(a.device)
 out=Path(a.out).resolve();root=Path("/ssd/lzx/mopd")
 if not out.is_relative_to(root):raise ValueError("Output outside project")
 out.mkdir(parents=True,exist_ok=True)
 lock=(out/"generation.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest()
 model_version_id=a.model_version_id
 if not model_version_id and a.adapter:
  adapter_path=Path(a.adapter).resolve()
  manifest_path=adapter_path.parent/"model-version.json"
  if manifest_path.is_file():
   version=json.loads(manifest_path.read_text())
   match=next((item for item in version.get("checkpoints",[]) if Path(item["path"]).resolve()==adapter_path),None)
   model_version_id=match["version_id"] if match else version.get("version_id")
 if not model_version_id:
  model_version_id=("teacher:" if a.teacher else "base:")+Path(a.model).name
 protocol_id=a.protocol_id or f"mopd-{Path(a.data).stem}-greedy-{a.max_new_tokens}-v1"
 generation_options={"temperature":0,"max_new_tokens":a.max_new_tokens,"max_model_len":16384,"seed":20261013}
 adapter_sha256=hashlib.sha256((Path(a.adapter)/"adapter_model.safetensors").read_bytes()).hexdigest() if a.adapter else None
 config=build_generation_manifest(
  run_id=model_version_id.rsplit(":step-",1)[0] if ":step-" in model_version_id else model_version_id,
  model_version_id=model_version_id,protocol_id=protocol_id,
  prompt_mode="teacher_chat_template_v1" if a.teacher else STUDENT_PROMPT_MODES[a.student_prompt_format],
  generation=generation_options,data_path=Path(a.data).resolve(),data_sha256=data_sha256,
  backend="vllm",weight_representation="base_plus_lora" if a.adapter else "base_weights",
  inference_dtype="bfloat16",model={"base_model":str(Path(a.model).resolve()),
   "checkpoint_path":str(Path(a.adapter).resolve()) if a.adapter else None,"checkpoint_sha256":adapter_sha256})
 config["runtime"]={"batch_size":a.batch_size,"max_seqs":a.max_seqs,"max_num_batched_tokens":2048}
 manifest=out/"generation-config.json"
 if manifest.exists():
  if json.loads(manifest.read_text())!=config:raise ValueError("Resume configuration mismatch")
 else:manifest.write_text(json.dumps(config,indent=2))
 rows=[json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
 keys=[(r["domain"],str(r["id"])) for r in rows]
 if len(set(keys))!=len(keys):raise ValueError("Duplicate benchmark ID")
 records=out/"generated.jsonl";done=set()
 if records.exists():
  with records.open("r+b") as f:
   while True:
    offset=f.tell();line=f.readline()
    if not line:break
    try:r=json.loads(line)
    except json.JSONDecodeError:
     if f.read():raise ValueError("Corruption before final record")
     f.truncate(offset);break
    k=(r["domain"],str(r["id"]))
    if k in done or k not in set(keys):raise ValueError("Duplicate/foreign output")
    done.add(k)
 pending=[r for r in rows if (r["domain"],str(r["id"])) not in done]
 if not pending:return
 from vllm import LLM,SamplingParams
 from vllm.lora.request import LoRARequest
 from mopd.data.domains import student_prompt,teacher_content
 from mopd.data.templates import apply_chat_template_ids
 engine=LLM(model=a.model,dtype="bfloat16",max_model_len=16384,
  gpu_memory_utilization=.85,max_num_seqs=a.max_seqs,max_num_batched_tokens=2048,
  enable_chunked_prefill=True,enable_prefix_caching=True,generation_config="vllm",
  enable_lora=bool(a.adapter),max_lora_rank=16,seed=20261013)
 tok=engine.get_tokenizer()
 adapter=LoRARequest("evaluated-student",1,a.adapter) if a.adapter else None
 params=SamplingParams(temperature=0,max_tokens=a.max_new_tokens,stop_token_ids=[tok.eos_token_id])
 started=time.monotonic()
 with records.open("a",buffering=1) as f:
  for offset in range(0,len(pending),a.batch_size):
   batch=pending[offset:offset+a.batch_size];prompts=[]
   for r in batch:
    ids=apply_chat_template_ids(tok,[{"role":"user","content":teacher_content(r)}],add_generation_prompt=True,enable_thinking=False) if a.teacher else tok.encode(student_prompt(tok,r,a.student_prompt_format))
    if len(ids)+a.max_new_tokens>16384:raise ValueError("Prompt exceeds configured context: "+str(r["id"]))
    prompts.append({"prompt_token_ids":ids})
   outputs=engine.generate(prompts,params,lora_request=adapter,use_tqdm=False)
   if len(outputs)!=len(batch):raise ValueError("Missing outputs")
   for row,result in zip(batch,outputs):
    answer=result.outputs[0]
    record={"id":row["id"],"domain":row["domain"],"prompt":row["prompt"],"response":answer.text,
      "response_ids":list(answer.token_ids),"tokens":len(answer.token_ids),
      "truncated":answer.finish_reason=="length","finish_reason":answer.finish_reason}
    f.write(json.dumps(record,ensure_ascii=False)+"\n");done.add((row["domain"],str(row["id"])))
   f.flush();os.fsync(f.fileno())
   progress={"done":len(done),"total":len(rows),"seconds_this_session":time.monotonic()-started,"backend":"vllm"}
   tmp=out/"generation-progress.partial";tmp.write_text(json.dumps(progress,indent=2));tmp.replace(out/"generation-progress.json")
   print(json.dumps(progress),flush=True)
if __name__=="__main__":main()
