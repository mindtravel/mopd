"""Keep one GPU evaluating: fixed targets first, then checkpoints of watched runs as they appear."""
import argparse,json,subprocess,sys,time
from pathlib import Path

ROOT=Path("/ssd/lzx/mopd")
FINAL={"complete","failed","stopped"}

def target_dir(manifest,checkpoint):
    return Path(manifest).parent/"evaluations"/(Path(checkpoint).name if checkpoint!="base" else "base")

def pending(a):
    items=[]
    for spec in a.fixed:
        manifest,checkpoint,fmt=spec.split("::")
        items.append((manifest,checkpoint,fmt))
    for run in a.watch:
        manifest=Path(run)/"model-version.json"
        if not manifest.is_file():continue
        version=json.loads(manifest.read_text())
        items+=[(str(manifest),c["path"],None) for c in version.get("checkpoints",[]) if c.get("complete")]
    return [i for i in items if not (target_dir(i[0],i[1])/a.protocol_id/"summary.json").exists()]

def watched_final(a):
    for run in a.watch:
        status=Path(run)/"status.json"
        if not status.is_file() or json.loads(status.read_text()).get("state") not in FINAL:return False
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",type=int,required=True);p.add_argument("--data",required=True)
    p.add_argument("--protocol-id",required=True);p.add_argument("--max-new-tokens",type=int,default=4096)
    p.add_argument("--max-seqs",type=int,default=64);p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--fixed",action="append",default=[],help="MANIFEST::CHECKPOINT::FORMAT (FORMAT may be empty)")
    p.add_argument("--watch",action="append",default=[],help="run directory whose checkpoints are evaluated")
    p.add_argument("--poll",type=int,default=60)
    a=p.parse_args()
    state_path=ROOT/"logs"/f"eval-queue-{a.protocol_id}-gpu{a.device}.json"
    failed=set()
    while True:
        todo=[i for i in pending(a) if i not in failed]
        state_path.write_text(json.dumps({"pending":todo,"failed":sorted(failed),"updated":time.time()},indent=1))
        if not todo:
            if watched_final(a):return
            time.sleep(a.poll);continue
        manifest,checkpoint,fmt=todo[0]
        command=[sys.executable,str(ROOT/"scripts/evaluate_model_version.py"),"--model-version",manifest,
            "--checkpoint",checkpoint,"--data",a.data,"--protocol-id",a.protocol_id,"--device",str(a.device),
            "--max-new-tokens",str(a.max_new_tokens),"--max-seqs",str(a.max_seqs),"--batch-size",str(a.batch_size)]
        if fmt:command+=["--student-prompt-format",fmt]
        print(json.dumps({"start":todo[0],"time":time.time()}),flush=True)
        if subprocess.run(command,cwd=ROOT).returncode!=0:
            failed.add(todo[0]);print(json.dumps({"failed":todo[0]}),flush=True)

if __name__=="__main__":main()
