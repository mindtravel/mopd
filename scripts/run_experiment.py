"""Execute an experiment's job graph (configs/experiments/<name>/plan.json) on the server.

Before anything starts: the deployed source must be a clean, unmodified commit, the spec
sha256 must match plan.json, every data pin must verify (mopd.runtime.provenance), and the
experiment's GPUs must be idle. Then jobs start as soon as their dependencies finish:
  train_sft / eval_* / merge use one free GPU; train_mopd needs its fixed GPU set, which is
  reserved as soon as it is ready so single-GPU jobs cannot starve it; export runs on CPU.
A job already finished (run complete, summary.json present, .finalized marker) is skipped,
so the same command resumes after an interruption; a half-written training run fails the
job instead of being overwritten. Every job start/finish is logged with free disk space.

  python scripts/run_experiment.py configs/experiments/scale256/plan.json
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mopd.runtime.provenance import _source,_data,sha256
PY=sys.executable
# Whole benchmark in one continuous-batching pass. Verified against the earlier 64-row batches on
# scale256 mix-mi step 256: cross-setting differences (581/1090 identical, 41 verdict flips) match
# same-setting repeats (589/38 and 594/40); generation wall time 482 s -> 155 s.
EVAL_SCHEDULING=["--batch-size","4096","--max-seqs","256","--max-num-batched-tokens","8192"]
def free_gib():
    s=os.statvfs(ROOT);return s.f_bfree*s.f_frsize/2**30
class Runner:
    def __init__(self,plan_path,min_free_gib):
        self.plan=json.loads(Path(plan_path).read_text());self.min_free=min_free_gib
        name=self.plan["experiment"]["name"];self.logdir=ROOT/"logs/experiments"/name;self.logdir.mkdir(parents=True,exist_ok=True)
        self.jobs={j["id"]:j for j in self.plan["jobs"]};self.state={}
        self.lock=threading.Lock();self.busy=set()
    def log(self,event,**fields):
        record={"time":time.strftime("%Y-%m-%dT%H:%M:%S"),"event":event,"ssd_free_gib":round(free_gib(),1),**fields}
        with self.lock,(self.logdir/"events.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
    def save(self):
        with self.lock:(self.logdir/"status.json").write_text(json.dumps(self.state,indent=2))
    def preflight(self):
        exp=self.plan["experiment"]
        if sha256(exp["spec_path"])!=exp["spec_sha256"]:raise SystemExit("spec changed since plan.json was generated")
        spec=json.loads(Path(exp["spec_path"]).read_text())
        source=_source(ROOT);_data({"data_pins":spec["data_pins"]})
        used=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
        busy={int(i):int(m) for i,m in (l.split(",") for l in used.strip().splitlines())}
        for name,ref in self.plan.get("reuse_experts",{}).items():
            src=json.loads((Path(ref["dir"])/"expert-source.json").read_text())["run_config"]
            if (src.get("experiment") or {}).get("name")!=ref["from_experiment"]:
                raise SystemExit(f"reused expert {name}: {ref['dir']} was not trained by experiment {ref['from_experiment']}")
            mapping={"student":"student","seed":"seed","steps":"steps","lr":"lr","lr_schedule":"lr_schedule",
                     "micro_batch_size":"micro_batch_size","sft_max_seq_len":"max_seq_len"}
            diff={k:(self.plan["controls"][k],src.get(v)) for k,v in mapping.items() if self.plan["controls"][k]!=src.get(v)}
            if sum(d["count"] for d in src["domains"].values())!=self.plan["controls"]["samples_per_step"]:diff["samples_per_step"]="differs"
            if diff:raise SystemExit(f"reused expert {name} was trained with different controls: {diff}")
            if not (Path(ref["dir"])/".finalized").exists():raise SystemExit(f"reused expert {name} is not finalized")
        taken=[g for g in self.plan["gpus"] if busy.get(g,0)>1000]
        if taken:raise SystemExit(f"GPUs not idle: {taken}")
        self.log("preflight_ok",source=source["source_version"],pins=len(spec["data_pins"]))
    def done(self,j):
        k=j["kind"]
        if k in("train_sft","train_mopd"):
            status=Path(json.loads((ROOT/j["config"]).read_text())["output"])/"status.json"
            return status.exists() and json.loads(status.read_text()).get("state")=="complete"
        if k=="eval_checkpoint":
            run=Path(j["model_version"]).parent
            return (run/"evaluations"/j["checkpoint"]/self.plan["evaluation"]["protocol_id"]/"summary.json").exists()
        if k in("export_expert","merge"):return (Path(j["out"])/".finalized").exists()
        if k=="eval_model":return (ROOT/j["out"]/"summary.json").exists()
        raise ValueError(k)
    def commands(self,j,gpus):
        ev=self.plan["evaluation"];k=j["kind"];base=self.plan["student"]
        if k=="train_sft":return [([PY,"scripts/train_sft.py","--config",j["config"]],{"CUDA_VISIBLE_DEVICES":str(gpus[0])})]
        if k=="train_mopd":return [([PY,"-u","scripts/train_multidomain_opd.py","--config",j["config"]],{})]
        if k=="eval_checkpoint":
            return [([PY,"scripts/evaluate_model_version.py","--model-version",j["model_version"],"--checkpoint",j["checkpoint"],
                      "--data",ev["data"],"--protocol-id",ev["protocol_id"],"--device",str(gpus[0]),
                      "--max-new-tokens",str(ev["max_new_tokens"])]+EVAL_SCHEDULING,{})]
        if k=="export_expert":
            return [([PY,"scripts/export_lora_expert.py","--base",base,"--adapter",j["adapter"],"--out",j["out"]],{"CUDA_VISIBLE_DEVICES":""}),
                    ([PY,"scripts/finalize_merged_model.py","--base",base,j["out"]],{})]
        if k=="merge":
            return [([str(ROOT/"mopd_merge/bin/mergekit-yaml"),j["yaml"],j["out"],"--cuda","--lazy-unpickle"],{"CUDA_VISIBLE_DEVICES":str(gpus[0])}),
                    ([PY,"scripts/finalize_merged_model.py","--base",base,j["out"]],{})]
        if k=="eval_model":
            out=ROOT/j["out"]
            return [([str(ROOT/"mopd_infer/bin/python"),"scripts/generate_benchmark_vllm.py","--model",j["model"],"--data",ev["data"],
                      "--out",str(out),"--device",str(gpus[0]),"--model-version-id",f"merge:{Path(j['model']).name}",
                      "--protocol-id",ev["protocol_id"],"--student-prompt-format","plain_qa_v1",
                      "--max-new-tokens",str(ev["max_new_tokens"])]+EVAL_SCHEDULING,{}),
                    ([PY,"scripts/grade_benchmark_outputs.py","--data",ev["data"],"--out",str(out)],{})]
        raise ValueError(k)
    def finish_artifact(self,j):
        out=Path(j["out"])
        if j["kind"]=="merge":
            shutil.copy(ROOT/j["yaml"],out/Path(j["yaml"]).name)
            experts=[l.split("model:",1)[1].strip() for l in (ROOT/j["yaml"]).read_text().splitlines() if "model:" in l]
            (out/"merge-source.json").write_text(json.dumps({"yaml":j["yaml"],"yaml_sha256":sha256(ROOT/j["yaml"]),
                "experts":{e:json.loads((Path(e)/"expert-source.json").read_text())["adapter_sha256"] for e in experts}},indent=2))
        (out/".finalized").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    def run_job(self,j,gpus):
        log=self.logdir/(j["id"].replace(":","__")+".log")
        self.state[j["id"]]={"state":"running","gpus":gpus,"started":time.strftime("%H:%M:%S")};self.save()
        self.log("start",job=j["id"],gpus=gpus)
        try:
            if j["kind"] in("train_sft","train_mopd"):
                output=Path(json.loads((ROOT/j["config"]).read_text())["output"])
                if output.exists():raise RuntimeError(f"incomplete run directory exists: {output}; inspect before rerunning")
            for cmd,extra in self.commands(j,gpus):
                with log.open("a") as f:
                    f.write(f"\n$ {' '.join(cmd)}\n");f.flush()
                    subprocess.run(cmd,cwd=ROOT,env={**os.environ,**extra},stdout=f,stderr=subprocess.STDOUT,check=True)
            if j["kind"] in("export_expert","merge"):self.finish_artifact(j)
            self.state[j["id"]]["state"]="complete"
        except Exception as error:
            self.state[j["id"]].update(state="failed",error=str(error)[-500:],log=str(log))
        self.state[j["id"]]["finished"]=time.strftime("%H:%M:%S");self.save()
        self.log("finish",job=j["id"],state=self.state[j["id"]]["state"])
        with self.lock:self.busy-=set(gpus)
    def loop(self):
        for j in self.jobs.values():
            if self.done(j):self.state[j["id"]]={"state":"complete","skipped":True}
        self.save();threads=[]
        order={"train_mopd":0,"train_sft":1,"merge":2,"export_expert":3,"eval_model":4,"eval_checkpoint":5}
        while True:
            pending=[j for j in self.jobs.values() if j["id"] not in self.state]
            if not pending and all(s["state"]!="running" for s in self.state.values()):break
            blocked=[j for j in pending if any(self.state.get(d,{}).get("state")=="failed" for d in j["deps"])]
            for j in blocked:self.state[j["id"]]={"state":"blocked","by":[d for d in j["deps"] if self.state.get(d,{}).get("state")=="failed"]}
            if blocked:self.save();continue
            ready=sorted((j for j in pending if all(self.state.get(d,{}).get("state")=="complete" for d in j["deps"])),key=lambda j:order[j["kind"]])
            with self.lock:
                free=[g for g in self.plan["gpus"] if g not in self.busy]
                reserved={g for j in ready if isinstance(j["gpus"],list) for g in j["gpus"]}
            for j in ready:
                if free_gib()<self.min_free:self.log("low_disk_hold",job=j["id"]);break
                if isinstance(j["gpus"],list):
                    if not set(j["gpus"])<=set(free):continue
                    gpus=j["gpus"]
                elif j["gpus"]==0:gpus=[]
                else:
                    pool=[g for g in free if g not in reserved]
                    if not pool:continue
                    gpus=[pool[0]]
                with self.lock:self.busy|=set(gpus)
                free=[g for g in free if g not in gpus]
                self.state[j["id"]]={"state":"starting"}
                t=threading.Thread(target=self.run_job,args=(j,gpus),daemon=False);t.start();threads.append(t)
            time.sleep(10)
        for t in threads:t.join()
        counts={}
        for s in self.state.values():counts[s["state"]]=counts.get(s["state"],0)+1
        self.log("experiment_done",**counts)
def main():
    p=argparse.ArgumentParser();p.add_argument("plan");p.add_argument("--min-free-gib",type=float,default=100);a=p.parse_args()
    r=Runner(a.plan,a.min_free_gib);r.preflight();r.loop()
if __name__=="__main__":main()
