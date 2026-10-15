"""Per-run evaluation tables and the cross-run index; read-only, no torch imports."""
import math
from pathlib import Path
from monitor_data import ROOT,obj,rows,run_path,partial_benchmark,version_registry

MAIN_PROTOCOL="all669-plainqa-greedy4096-v1"

def _case_map(path):
 out={}
 for r in rows(path):out.setdefault(r.get("domain"),{})[str(r["id"])]=bool(r.get("correct"))
 return out

def paired(before,after):
 """Exact two-sided McNemar test on the questions both versions answered."""
 keys=set(before)&set(after)
 if not keys:return None
 gained=sum(1 for k in keys if after[k] and not before[k]);lost=sum(1 for k in keys if before[k] and not after[k])
 n=gained+lost
 p=min(1.0,2*sum(math.comb(n,i) for i in range(min(gained,lost)+1))/2**n) if n else 1.0
 return {"n":len(keys),"gained":gained,"lost":lost,"p":p}

def _row(out,label,kind,step=None):
 summary=obj(out/"summary.json");progress=summary or partial_benchmark(out)
 domains={}
 if summary:
  for d,v in summary.get("domains",{}).items():
   domains[d]={k:v.get(k) for k in ("accuracy","count","truncated","strict_instruction_accuracy")}
 return {"label":label,"kind":kind,"step":step,"path":str(out.relative_to(ROOT)),"complete":bool(summary),
  "done":progress.get("done"),"total":progress.get("total"),"domains":domains}

def _base_eval(run,cfg,protocol):
 own=run/"evaluations/base"/protocol
 if (own/"summary.json").exists():return own
 for path in sorted((ROOT/"runs").glob(f"*/evaluations/base/{protocol}/summary.json")):
  if obj(path).get("generation",{}).get("model",{}).get("base_model")==cfg.get("student"):return path.parent
 return None

def _init_eval(cfg,protocol):
 adapter=cfg.get("student_adapter")
 if not adapter:return None
 out=Path(adapter).parent/"evaluations"/Path(adapter).name/protocol
 return out if (out/"summary.json").exists() else None

def eval_tables(key):
 run=run_path(key);cfg=obj(run/"config.json")
 protocols=sorted({p.name for p in run.glob("evaluations/checkpoint-*/*") if p.is_dir()},key=lambda p:(p!=MAIN_PROTOCOL,p))
 tables=[]
 for protocol in protocols:
  refs=[]
  base=_base_eval(run,cfg,protocol)
  if base:refs.append(("base",base,_row(base,"原始学生（base）","base",0)))
  init=_init_eval(cfg,protocol)
  if init:refs.append(("init",init,_row(init,"初始化来源（"+init.parents[2].name+" · "+init.parent.name+"）","init",0)))
  ref_cases={kind:_case_map(path/"cases.jsonl") for kind,path,_ in refs}
  rows_=[r for _,_,r in refs]
  for out in sorted(run.glob(f"evaluations/checkpoint-*/{protocol}")):
   step=int(out.parent.name.split("-")[1])
   r=_row(out,f"第 {step} 步","checkpoint",step)
   if r["complete"]:
    cur=_case_map(out/"cases.jsonl")
    for kind,cases in ref_cases.items():
     r["vs_"+kind]={d:paired(cases[d],cur[d]) for d in cur if d in cases}
   rows_.append(r)
  tables.append({"protocol":protocol,"base_from":base.parents[2].name if base else None,"rows":rows_})
 return tables

def _mtime(path):
 try:return path.stat().st_mtime
 except OSError:return None

def run_index():
 items=[]
 for v in version_registry():
  run=ROOT/v["path"];cfg=obj(run/"config.json");st=obj(run/"status.json")
  tables=eval_tables(v["key"])
  main=next((t for t in tables if t["protocol"]==MAIN_PROTOCOL),None)
  ckpts=[r for r in (main or {}).get("rows",[]) if r["kind"]=="checkpoint"]
  done=[r for r in ckpts if r["complete"]]
  latest=done[-1] if done else None
  best=max(done,key=lambda r:r["domains"].get("instruction",{}).get("accuracy") or -1) if done else None
  steps=cfg.get("steps") or 0
  items.append({"key":v["key"],"state":st.get("state","pending"),"step":st.get("step",st.get("steps",0)),"steps":steps,
   "updated":max(filter(None,[_mtime(run/"status.json"),_mtime(run/"metrics.jsonl")]),default=None),
   "student":cfg.get("student"),"init":cfg.get("student_adapter"),"lr":cfg.get("lr"),
   "domains":{d:{"train_file":s.get("train_file"),"mixture":s.get("mixture"),"count":s.get("count"),"weight":s.get("weight")} for d,s in cfg.get("domains",{}).items()},
   "minor":steps<16 or "smoke" in v["key"] or st.get("state") in ("failed","stopped"),
   "evaluated":f"{len(done)}/{len(ckpts)}" if ckpts else None,
   "latest":latest,"best":best,
   "base":next((r for r in (main or {}).get("rows",[]) if r["kind"]=="base" and r["complete"]),None)})
 items.sort(key=lambda i:i["updated"] or 0,reverse=True)
 return items
