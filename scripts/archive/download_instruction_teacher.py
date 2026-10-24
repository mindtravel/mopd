"""Download official mirror bytes; verify every file against pinned HF metadata."""
import concurrent.futures, hashlib, json, os, time
from pathlib import Path
import requests
ROOT=Path("/ssd/lzx/mopd/models/Qwen3-4B-Instruct-2507")
META=json.loads((ROOT/"upstream_metadata.json").read_text())
REV=META["sha"]
def digest(path,kind):
    h=hashlib.new(kind)
    if kind=="sha1":h.update(("blob "+str(path.stat().st_size)+"\0").encode())
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def download(item):
    name=item["rfilename"]
    if name==".gitattributes":return
    dst=ROOT/name;part=ROOT/(name+".part")
    lfs=item.get("lfs");expected=lfs["sha256"] if lfs else item.get("blobId")
    kind="sha256" if lfs else "sha1"
    if not expected:raise ValueError(("missing hash",name,item))
    if dst.exists() and digest(dst,kind)==expected:return {"file":name,"verified":True,"hash":expected}
    if lfs:
        size=lfs["size"];part.touch(exist_ok=True)
        while part.stat().st_size<size:
            start=part.stat().st_size;end=min(size-1,start+32*1024*1024-1)
            for retry in range(5):
                try:
                    u=f"https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507/resolve/master/{name}?offset={start}"
                    r=requests.get(u,headers={"Range":f"bytes={start}-{end}"},timeout=(20,90))
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{size}" or len(r.content)!=end-start+1:raise ValueError("Invalid range response")
                    with part.open("ab") as f:f.write(r.content)
                    break
                except Exception as e:
                    print("RETRY",name,start,retry,repr(e),flush=True)
                    if retry==4:raise
                    time.sleep(2)
            print(json.dumps({"file":name,"bytes":part.stat().st_size,"total":size}),flush=True)
    else:
        u=f"https://hf-mirror.com/Qwen/Qwen3-4B-Instruct-2507/resolve/{REV}/{name}"
        r=requests.get(u,timeout=(20,60));r.raise_for_status();part.write_bytes(r.content)
    if digest(part,kind)!=expected:raise ValueError(("checksum mismatch",name))
    part.replace(dst)
    return {"file":name,"verified":True,"hash":expected}
if __name__=="__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(download,META["siblings"]))
    (ROOT/"verified.json").write_text(json.dumps({"revision":REV,"files":results},indent=2))
    print("ALL_VERIFIED",flush=True)
