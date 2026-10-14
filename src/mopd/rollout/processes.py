"""Independent Python process per GPU, synchronous versioned adapter transfer."""
from concurrent.futures import ThreadPoolExecutor
import atexit,json,subprocess,sys,time
from pathlib import Path
import torch
from mopd.rollout.replicas import assign_shards
class ProcessRollouts:
    def __init__(self,c,student):
        self.c=c;self.workers=[]
        self.root=Path(c["output"])/"rollout_runtime";self.root.mkdir(parents=True,exist_ok=True)
        atexit.register(self.close)
        try:
            for device in c["rollout_devices"]:
                err=(self.root/(device.replace(":","-")+".log")).open("a")
                p=subprocess.Popen([sys.executable,"-u","-m","mopd.rollout.worker","--model",c["student"],"--device",device],
                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
                self.workers.append((device,p,err))
            for device,p,_ in self.workers:
                ready=json.loads(p.stdout.readline() or "{}")
                if not ready.get("ready") or ready["device"]!=device:raise RuntimeError("Worker failed to load: "+device)
        except BaseException:self.close();raise
    def close(self):
        for _,p,err in self.workers:
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            err.close()
        self.workers=[]
    def generate(self,student,rows,version,*,sample=True,min_new_tokens=0):
        start=time.perf_counter()
        state={n:p.detach().cpu() for n,p in student.named_parameters() if "lora_" in n}
        part=self.root/"adapter.partial";dest=self.root/"adapter.pt"
        torch.save(state,part);part.replace(dest)
        sync_seconds=time.perf_counter()-start
        shards=assign_shards(list(enumerate(rows)),len(self.workers))
        def run(worker,shard,index):
            device,p,_=worker
            if not shard:return [],{}
            request={"state":str(dest),"rows":[r for _,r in shard],"version":version,
              "max_new_tokens":self.c["max_new_tokens"],"batch_size":self.c["generation_batch_size"],"prompt_format":self.c["student_prompt_format"],
              "sample":sample,"min_new_tokens":min_new_tokens,"seed":self.c["seed"]+version*len(self.workers)+index}
            p.stdin.write(json.dumps(request)+"\n");p.stdin.flush()
            result=json.loads(p.stdout.readline() or "{}")
            if "error" in result:raise RuntimeError(result["error"])
            if result.get("version")!=version or len(result.get("output",[]))!=len(shard):raise RuntimeError("Missing/stale rollout")
            output=result.pop("output")
            for item,(idx,_) in zip(output,shard):
                item.update(index=idx,rollout_device=device,student_version=version)
            result["tokens"]=sum(len(r["answer"]) for r in output);result["count"]=len(output)
            return output,result
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            futures=[pool.submit(run,w,s,i) for i,(w,s) in enumerate(zip(self.workers,shards))]
            results=[f.result() for f in futures]
        output=sorted([r for items,_ in results for r in items],key=lambda r:r["index"])
        if len(output)!=len(rows):raise ValueError("Lost rollout")
        return output,time.perf_counter()-start,{"sync_seconds":sync_seconds,"workers":[s for _,s in results],"backend":"processes"}
