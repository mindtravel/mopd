"""Read-only data adapters for current and historical experiments."""
import json,subprocess,time,threading,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT=Path("/ssd/lzx/mopd")
RUNS={"dual":{"path":"runs/20261014-dual-tulu3-from-base-vllm","label":"新实验 · Base 从头 · 数学 + Tülu 3 指令"},
      "dual_previous":{"path":"runs/20261014-dual-domain-batch32-mgpu-opd-b8","label":"历史实验 · 数学 + Alpaca 指令"},
      "math":{"path":"runs/20261013-numina-8192-batch2-opd","label":"历史实验 · 单领域数学"}}
_cache={};_lock=threading.Lock();_gpu={}
def obj(path):
 try:return json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return {}
def rows(path):
 try:stat=path.stat()
 except OSError:return []
 key=(stat.st_mtime_ns,stat.st_size)
 with _lock:
  hit=_cache.get(str(path))
  if hit and hit[0]==key:return hit[1]
  result=[]
  for line in path.read_text().splitlines():
   try:result.append(json.loads(line))
   except json.JSONDecodeError:continue
  _cache[str(path)]=(key,result)
  return result
def run_path(key):
 if key in RUNS:return ROOT/RUNS[key]["path"]
 if not key or Path(key).name!=key or key in {".",".."}:raise ValueError("Unknown run")
 candidate=(ROOT/"runs"/key).resolve()
 if candidate.parent!=(ROOT/"runs").resolve() or not candidate.is_dir():raise ValueError("Unknown run")
 return candidate
def gpu_snapshot():
 global _gpu
 if time.time()-_gpu.get("time",0)<10:return _gpu
 try:
  text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=4)
  data=[dict(zip(["index","used_mib","total_mib","utilization"],map(int,line.split(",")))) for line in text.strip().splitlines()]
  _gpu={"time":time.time(),"devices":data}
 except (OSError,subprocess.SubprocessError,ValueError):_gpu={"time":time.time(),"devices":[],"error":"GPU 状态暂不可读"}
 return _gpu
def cases(key,step=None,domain=None,truncated=False,offset=0,limit=20):
 run=run_path(key);raw=rows(run/"rollouts.jsonl")
 picked=[r for r in raw if (step is None or r["step"]==step) and (not domain or r.get("domain","math")==domain) and (not truncated or r.get("truncated"))]
 selected=[{k:v for k,v in r.items() if k!="response_ids"} for r in picked[::-1][offset:offset+limit]]
 if key=="math":
  prompts={r["id"]:r for r in rows(ROOT/"data/numina/train.jsonl")}
  for r in selected:r.update(domain="math",prompt=prompts.get(r["id"],{}).get("problem",""))
 return {"rows":selected,"total":len(picked),"offset":offset,"limit":limit}
def evaluations(key,run):
 if key!="math":
  result=[]
  for path in sorted((run/"validation/ability").glob("step-*")):
   summary=obj(path/"summary.json");progress=summary or obj(path/"progress.json")
   result.append({"step":int(path.name.split("-")[1]),"complete":bool(summary),"result":progress})
  return result,obj(run/"validation/ability/status.json")
 result=[]
 base=ROOT/"runs/20261013-math-evaluation-8192"
 for step,name in [(0,"student_before"),(64,"numina_student_after")]:
  summary=obj(base/name/"summary.json");progress=summary or obj(base/name/"progress.json")
  if progress:result.append({"step":step,"complete":bool(summary),"result":{**progress,"domains":{"math":progress}}})
 return result,{}
def partial_benchmark(out):
 summary=obj(out/"summary.json")
 if summary:return summary
 progress=obj(out/"generation-progress.json")
 report=obj(out/"scoring-progress.json")
 if not progress and not report:return {}
 return {**progress,**report,"done":report.get("scored_count",progress.get("done",0)),"partial":True}

def benchmark_report(key,run):
 """Return only reports whose immutable generation manifest names this version."""
 reports=[]
 for manifest_path in sorted(run.rglob("generation-config.json")):
  generation=obj(manifest_path)
  version=generation.get("model_version_id","")
  if not version or not (version==key or version.startswith(key+":")):continue
  output=manifest_path.parent
  summary=obj(output/"summary.json")
  progress={**obj(output/"generation-progress.json"),**obj(output/"scoring-progress.json")}
  if summary:progress=summary
  reports.append({"path":str(output.relative_to(ROOT)),"model_version_id":version,
   "protocol_id":generation.get("protocol_id"),"protocol_sha256":generation.get("protocol_sha256"),
   "backend":generation.get("backend"),"summary":summary,"progress":progress,"complete":bool(summary)})
 return reports

def version_registry():
 items=[]
 for run in sorted((ROOT/"runs").iterdir()):
  if not run.is_dir():continue
  cfg=obj(run/"config.json")
  manifest=obj(run/"model-version.json")
  if not cfg or not manifest:continue
  st=obj(run/"status.json")
  checkpoints=sorted(p.name for p in run.glob("checkpoint-*") if p.is_dir())
  items.append({"schema":manifest.get("schema","ifairy.model_version.v1"),"version_id":manifest.get("version_id",run.name),"components":manifest.get("components",{}),"training":manifest.get("training",{}),"key":run.name,"label":run.name,"path":str(run.relative_to(ROOT)),"state":st.get("state","pending"),"step":st.get("step",st.get("steps",0)),"steps":cfg.get("steps"),"student":cfg.get("student"),"student_adapter":cfg.get("student_adapter"),"final_adapter":st.get("final_adapter"),"checkpoints":checkpoints,"rollout_backend":cfg.get("rollout_backend"),"domains":{d:{"teacher":v.get("teacher"),"train_file":v.get("train_file"),"weight":v.get("weight")} for d,v in cfg.get("domains",{}).items()}})
 return items

def available_runs():
 items={}
 for version in version_registry():items[version["key"]]={"key":version["key"],"label":version["label"]}
 for key,spec in RUNS.items():
  if (ROOT/spec["path"]).is_dir() and key not in items:
   items[key]={"key":key,"label":spec["label"]}
 return sorted(items.values(),key=lambda item:item["key"])
def gpu_roles(cfg,key):
 if obj(run_path(key)/"status.json").get("state")=="complete":return {}
 roles={}
 if cfg.get("student_device"):roles[cfg["student_device"].replace("cuda:","")]= "MOPD 学生更新"
 for domain,spec in cfg.get("domains",{}).items():
  if spec.get("device"):roles[spec["device"].replace("cuda:","")]=domain+" 教师评分"
 for index,device in enumerate(cfg.get("rollout_devices",[])):roles[device.replace("cuda:","")]=f"学生生成 worker {index+1}"
 return roles
def snapshot(key="dual"):
 run=run_path(key);config=obj(run/"config.json");metrics=rows(run/"metrics.jsonl");answers=rows(run/"rollouts.jsonl")
 domain_summary={}
 for r in answers:
  domain=r.get("domain","math")
  s=domain_summary.setdefault(domain,{"count":0,"tokens":0,"truncated":0})
  s["count"]+=1;s["tokens"]+=r.get("tokens",0);s["truncated"]+=int(r.get("truncated",False))
 evals,eval_status=evaluations(key,run)
 p=run/"metrics.jsonl"
 return {"run":key,"runs":available_runs(),
  "status":obj(run/"status.json"),"config":config,"metrics":metrics,
  "response_summary":{"count":len(answers),"domains":domain_summary,"steps":sorted({r["step"] for r in answers}),
    "max_tokens":max((r.get("tokens",0) for r in answers),default=0)},
  "validation":rows(run/"validation/kl.jsonl"),"evaluations":evals,"evaluation_status":eval_status,
  "gpu":gpu_snapshot(),"gpu_roles":gpu_roles(config,key),"versions":version_registry(),
  "benchmarks":benchmark_report(key,run),"now":time.time(),"updated":p.stat().st_mtime if p.exists() else None}
