import requests,hashlib,json
from pathlib import Path
root=Path("/ssd/lzx/mopd/data/numina");root.mkdir(parents=True,exist_ok=True)
rev="1e997dc9e012c163ba38548a953591cc0f63bea5"
url="https://modelscope.cn/api/v1/datasets/AI-MO/NuminaMath-CoT/repo"
params={"Revision":rev,"FilePath":"data/train-00000-of-00005.parquet"}
r=requests.get(url,params=params,stream=True,timeout=(20,60));r.raise_for_status()
p=root/"train-00000.parquet.part";h=hashlib.sha256();n=0
with p.open("wb") as f:
 for b in r.iter_content(4*1024*1024):
  f.write(b);h.update(b);n+=len(b);print(n,flush=True)
assert h.hexdigest()=="25c0fa2ca96dd8078470ad1bbc439d0d211fd8b7d22e04abf7f61e9be9bf9ec3"
p.rename(root/"train-00000.parquet")
(root/"download.json").write_text(json.dumps({"revision":rev,"sha256":h.hexdigest(),"bytes":n,"source":url,"params":params},indent=2))
print("VERIFIED",flush=True)
