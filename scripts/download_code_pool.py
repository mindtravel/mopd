"""Download one open-source code instruction shard plus HumanEval/MBPP for decontamination.

Mirrors scripts/prepare_math.py's download+verify pattern. A single
filtered_infinity_instruct shard (~34k rows after local filtering) of
OpenCoder-LLM/opc-sft-stage1 is enough for a small first code SFT pool; the other
21 shards of that dataset are left undownloaded until more data is actually needed.
HumanEval and MBPP (sanitized, all splits) are downloaded only to build the
decontamination n-gram set used by scripts/prepare_code_pool.py, the same way the
instruction pools are screened against IFEval.
"""
import hashlib,json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
BASE="https://hf-mirror.com"
TARGETS=[
    {"repo":"OpenCoder-LLM/opc-sft-stage1","revision":"1bcab575f5e2d1c1fd6652720418524c27b3d58b",
     "path":"data/filtered_infinity_instruct-00000-of-00004.parquet",
     "out":ROOT/"data/code/upstream/opc_sft_stage1/filtered_infinity_instruct-00000-of-00004.parquet",
     "license":"apache-2.0"},
    {"repo":"openai/openai_humaneval","revision":"7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
     "path":"openai_humaneval/test-00000-of-00001.parquet",
     "out":ROOT/"data/code/upstream/humaneval/test-00000-of-00001.parquet",
     "license":"MIT"},
] + [
    {"repo":"google-research-datasets/mbpp","revision":"4bb6404fdc6cacfda99d4ac4205087b89d32030c",
     "path":f"sanitized/{split}-00000-of-00001.parquet",
     "out":ROOT/f"data/code/upstream/mbpp/sanitized-{split}-00000-of-00001.parquet",
     "license":"cc-by-4.0"}
    for split in ("prompt","test","train","validation")
]
def get(url):
    for attempt in range(4):
        try:
            r=requests.get(url,timeout=90);r.raise_for_status();return r
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2**attempt)
def main():
    manifest=[]
    for t in TARGETS:
        info=get(f"{BASE}/api/datasets/{t['repo']}/tree/{t['revision']}?recursive=true").json()
        entry=next(f for f in info if f["path"]==t["path"])
        data=get(f"{BASE}/datasets/{t['repo']}/resolve/{t['revision']}/{t['path']}?download=true").content
        sha=hashlib.sha256(data).hexdigest()
        expected=entry.get("lfs",{}).get("oid",entry["oid"])
        assert sha==expected,f"{t['repo']}:{t['path']} sha256 mismatch"
        t["out"].parent.mkdir(parents=True,exist_ok=True)
        t["out"].write_bytes(data)
        manifest.append({"repo":t["repo"],"revision":t["revision"],"path":t["path"],
                          "license":t["license"],"sha256":sha,"bytes":len(data),"out":str(t["out"])})
        print("downloaded",t["repo"],t["path"],len(data),flush=True)
    (ROOT/"data/code/upstream/download_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
