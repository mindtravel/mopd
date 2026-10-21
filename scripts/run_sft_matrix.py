"""Queue SFT trainings and model evaluations across GPUs, one job per GPU at a time.

jobs file (JSON list), each job one of:
  {"name": ..., "model": "/path/to/hf_model_dir"}            evaluate standalone weights (e.g. merged)
  {"name": ..., "train": "configs/x.json"}                 train, then evaluate final checkpoint
  {"name": ..., "model_version": ".../model-version.json", "checkpoint": "checkpoint-0064" | "base"}
Every evaluation uses the same --data/--protocol-id so all results are one measurement.

Disk: /ssd is shared and nearly full. Before every job the free space root can still
use (statvfs f_bfree; shutil.disk_usage reports f_bavail, which is 0 for non-root once
the reserved blocks are reached) is logged to logs/disk-usage.jsonl, and no new job
starts below --min-free-gib. A running job is not interrupted.
"""
import argparse,json,os,queue,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESERVED_GPUS={1}
LOCK=threading.Lock()
def free_gib(path):
    s=os.statvfs(path);return s.f_bfree*s.f_frsize/2**30
def log_disk(event):
    record={"time":time.strftime("%Y-%m-%dT%H:%M:%S"),"event":event,"ssd_free_gib":round(free_gib(ROOT),1)}
    with LOCK,(ROOT/"logs/disk-usage.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
    return record["ssd_free_gib"]
def update(status_path,name,**fields):
    with LOCK:
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        status[name]={**status.get(name,{}),**fields,"updated":time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp=status_path.with_suffix(".tmp");tmp.write_text(json.dumps(status,indent=2));tmp.replace(status_path)
def run(cmd,log,env):
    with open(log,"a") as f:
        f.write(f"\n$ {' '.join(map(str,cmd))}\n");f.flush()
        subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def job_commands(job,gpu,a):
    if "model" in job:  # standalone weights (e.g. a mergekit output): generate + grade directly
        out=ROOT/"results/merge-evals"/job["name"]/a.protocol_id
        return [([ROOT/"mopd_infer/bin/python","scripts/generate_benchmark_vllm.py","--model",job["model"],"--data",a.data,
                  "--out",out,"--device",gpu,"--model-version-id",f"merge:{job['name']}","--protocol-id",a.protocol_id,
                  "--student-prompt-format","plain_qa_v1","--max-new-tokens",a.max_new_tokens,"--max-seqs",64,"--batch-size",64],{}),
                ([sys.executable,"scripts/grade_benchmark_outputs.py","--data",a.data,"--out",out],{})]
    commands=[];manifest=job.get("model_version");checkpoint=job.get("checkpoint")
    if "train" in job:
        config=json.loads((ROOT/job["train"]).read_text())
        commands.append(([sys.executable,"scripts/train_sft.py","--config",job["train"]],{"CUDA_VISIBLE_DEVICES":str(gpu)}))
        manifest=Path(config["output"])/"model-version.json";checkpoint=f"checkpoint-{config['steps']:04d}"
    commands.append(([sys.executable,"scripts/evaluate_model_version.py","--model-version",manifest,"--checkpoint",checkpoint,
        "--data",a.data,"--protocol-id",a.protocol_id,"--device",gpu,"--max-new-tokens",a.max_new_tokens,
        "--batch-size",64,"--max-seqs",64],{}))
    return commands
def worker(gpu,jobs,a,status_path):
    while True:
        try:job=jobs.get_nowait()
        except queue.Empty:return
        name=job["name"];free=log_disk(f"before:{name}:gpu{gpu}")
        if free<a.min_free_gib:
            update(status_path,name,state="skipped_low_disk",ssd_free_gib=free);continue
        update(status_path,name,state="running",gpu=gpu,ssd_free_gib_before=free)
        log=ROOT/"logs"/f"sft-matrix-{name}.log"
        try:
            for cmd,extra in job_commands(job,gpu,a):run(cmd,log,{**os.environ,**extra})
            update(status_path,name,state="complete",ssd_free_gib_after=log_disk(f"after:{name}"))
        except subprocess.CalledProcessError as error:
            update(status_path,name,state="failed",returncode=error.returncode,log=str(log))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jobs",required=True);p.add_argument("--gpus",required=True,help="comma-separated physical GPU ids")
    p.add_argument("--data",required=True);p.add_argument("--protocol-id",required=True)
    p.add_argument("--max-new-tokens",type=int,default=4096);p.add_argument("--min-free-gib",type=float,default=40)
    p.add_argument("--status",default="logs/sft-matrix-status.json");a=p.parse_args()
    gpus=[int(x) for x in a.gpus.split(",")]
    if RESERVED_GPUS&set(gpus):raise ValueError(f"GPUs {sorted(RESERVED_GPUS)} are reserved")
    jobs=queue.Queue()
    for job in json.loads((ROOT/a.jobs).read_text()):jobs.put(job)
    status_path=ROOT/a.status
    threads=[threading.Thread(target=worker,args=(g,jobs,a,status_path)) for g in gpus]
    for t in threads:t.start()
    for t in threads:t.join()
    log_disk("matrix_done")
if __name__=="__main__":main()
