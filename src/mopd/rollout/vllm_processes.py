"""Isolated vLLM workers; all generation finishes before the next student update."""
import atexit,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from mopd.data.domains import student_prompt
from mopd.rollout.vllm_protocol import validate_reply
from mopd.rollout.dispatch import length_bucket_dispatch
class VllmRollouts:
    def __init__(self,c,student):
        self.c=c;self.workers=[]
        if not c.get("rollout_devices"):raise ValueError("No rollout GPUs configured")
        self.root=Path(c["output"])/"vllm_runtime";self.root.mkdir(parents=True,exist_ok=True)
        self.tokenizer=AutoTokenizer.from_pretrained(c["student"],local_files_only=True)
        python=c.get("vllm_python","/ssd/lzx/mopd/mopd_infer/bin/python")
        atexit.register(self.close)
        try:
            for device in c["rollout_devices"]:
                if device not in ("cuda:2","cuda:3","cuda:5","cuda:6","cuda:7"):raise ValueError("Unapproved rollout GPU")
                log=(self.root/(device.replace(":","-")+".log")).open("a")
                args=[python,"-u","-m","mopd.rollout.vllm_worker","--model",c["student"],
                    "--device",device,"--max-seqs",str(c["generation_batch_size"]),
                    "--memory",str(c.get("vllm_memory_utilization",.85))]
                env=os.environ.copy()
                # Compiler caches are mutable: isolate workers and environments.
                cache=self.root/"compile_cache"/device.replace(":","-")
                for key,folder in (("TORCHINDUCTOR_CACHE_DIR","torch"),("TRITON_CACHE_DIR","triton"),("VLLM_CACHE_ROOT","vllm")):
                    dest=cache/folder;dest.mkdir(parents=True,exist_ok=True);env[key]=str(dest)
                proc=subprocess.Popen(args,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                    stderr=log,text=True,bufsize=1,start_new_session=True)
                self.workers.append((device,proc,log))
            for device,proc,_ in self.workers:
                ready=self.read(proc)
                if not ready.get("ready") or ready.get("device")!=device or ready.get("eos")!=self.tokenizer.eos_token_id:
                    raise RuntimeError("Worker/tokenizer readiness mismatch")
        except BaseException:self.close();raise
    @staticmethod
    def read(proc):
        for line in proc.stdout:
            if line.startswith("MOPD_JSON:"):
                value=json.loads(line[len("MOPD_JSON:"):])
                if "error" in value:raise RuntimeError(value["error"])
                return value
        raise RuntimeError("vLLM worker exited before replying")
    def close(self):
        for _,proc,log in self.workers:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
            log.close()
        self.workers=[]
    def generate(self,student,rows,version,*,sample=True,min_new_tokens=0):
        started=time.perf_counter()
        adapter=self.root/f"adapter-{version:06d}"
        if adapter.exists():raise ValueError("Adapter versions must be immutable and unique")
        student.save_pretrained(adapter,safe_serialization=True)
        self.tokenizer.save_pretrained(adapter)
        digest=hashlib.sha256((adapter/"adapter_model.safetensors").read_bytes()).hexdigest()
        inputs=[{"index":i,"prefix":self.tokenizer.encode(student_prompt(self.tokenizer,r,self.c["student_prompt_format"])),"row":r} for i,r in enumerate(rows)]
        n=len(self.workers)
        sync_seconds=time.perf_counter()-started
        # Small length-sorted batches plus work stealing prevent one long tail
        # request from idling the other worker for the rest of the step.
        batch_size=max(1, int(self.c.get("generation_batch_size", 8)))
        def run(worker_id,batch):
            device,proc,_=self.workers[worker_id]
            request={"adapter":str(adapter),"adapter_sha256":digest,"inputs":batch,"version":version,
                "max_new_tokens":self.c["max_new_tokens"],"sample":sample,"min_new_tokens":min_new_tokens,
                "seed":self.c["seed"]+version*n+worker_id,"debug_logprobs":self.c.get("debug_logprobs")}
            proc.stdin.write(json.dumps(request)+"\n");proc.stdin.flush()
            reply=self.read(proc)
            output=validate_reply(reply,batch,version,digest,self.c["max_new_tokens"],self.tokenizer.eos_token_id)
            for r in output:r["rollout_device"]=device
            stats={k:v for k,v in reply.items() if k!="output"}
            return output,stats
        started=time.perf_counter()
        def row_length(x): return len(x["prefix"])
        replies,dispatch=length_bucket_dispatch(inputs,workers=n,batch_size=batch_size,
            length=row_length,run=run)
        output=sorted([r for batch,_ in replies for r in batch],key=lambda r:r["index"])
        return output,time.perf_counter()-started,{"backend":"vllm-dynamic",
            "sync_seconds":sync_seconds,"dispatch":dispatch.__dict__,
            "workers":[s for _,s in replies]}
