"""Concurrent ranges from ModelScope, pinned Hugging Face hashes as authority."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
import requests
from download_instruction_teacher import ROOT,META,download,digest
CHUNK=16*1024*1024
def fetch(job):
    name,start,end,total,dst=job
    if dst.exists() and dst.stat().st_size==end-start+1:return
    for attempt in range(5):
        try:
            url=f"https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507/resolve/master/{name}?offset={start}"
            r=requests.get(url,headers={"Range":f"bytes={start}-{end}"},timeout=(15,90))
            r.raise_for_status()
            assert r.status_code==206 and r.headers.get("Content-Range")==f"bytes {start}-{end}/{total}"
            assert len(r.content)==end-start+1
            dst.write_bytes(r.content);return
        except Exception as e:
            print("RETRY",name,start,attempt,str(e),flush=True)
            if attempt==4:raise
            time.sleep(2)
def main():
    jobs=[];large=[]
    for item in META["siblings"]:
        if item["rfilename"]==".gitattributes":continue
        if not item.get("lfs") or item["lfs"]["size"]<128*1024*1024:
            download(item);continue
        name=item["rfilename"];size=item["lfs"]["size"];dest=ROOT/name
        if dest.exists() and digest(dest,"sha256")==item["lfs"]["sha256"]:continue
        parts=ROOT/"ranges"/name;parts.mkdir(parents=True,exist_ok=True)
        old=ROOT/(name+".part")
        if old.exists():
            with old.open("rb") as f:
                for start in range(0,old.stat().st_size,CHUNK):
                    end=min(size-1,start+CHUNK-1)
                    if end>=old.stat().st_size:break
                    dst=parts/str(start)
                    if not dst.exists():dst.write_bytes(f.read(end-start+1))
                    else:f.seek(end-start+1,1)
        for start in range(0,size,CHUNK):
            end=min(size-1,start+CHUNK-1)
            jobs.append((name,start,end,size,parts/str(start)))
        large.append(item)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for n,_ in enumerate(pool.map(fetch,jobs),1):
            if n%8==0:print(json.dumps({"completed_ranges":n,"total_ranges":len(jobs)}),flush=True)
    for item in large:
        name=item["rfilename"];size=item["lfs"]["size"];part=ROOT/(name+".assembled")
        with part.open("wb") as out:
            for start in range(0,size,CHUNK):
                out.write((ROOT/"ranges"/name/str(start)).read_bytes())
        assert part.stat().st_size==size and digest(part,"sha256")==item["lfs"]["sha256"]
        part.replace(ROOT/name)
    results=[download(i) for i in META["siblings"]]
    (ROOT/"verified.json").write_text(json.dumps({"revision":META["sha"],"files":results},indent=2))
    print("ALL_VERIFIED",flush=True)
if __name__=="__main__":main()
