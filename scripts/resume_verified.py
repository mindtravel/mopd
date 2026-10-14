"""Resume verified HF LFS blobs from official ModelScope; publish only after SHA256."""
import os,json,time,hashlib,concurrent.futures
from pathlib import Path
import requests
root=Path(os.environ['MOPD_ROOT'])
models={'Qwen3-4B':'1cfa9a7208912126459214e8b04321603b3df60c','Qwen3-4B-Base':'906bfd4b4dc7f14ee4320094d8b41684abff8539'}
def blob(name,item):
 folder=root/'models'/name; filename=item['rfilename'];info=item['lfs'];sha=info['sha256'];size=info['size'];dest=folder/filename
 if dest.exists():partial=dest
 else:
  candidates=list((folder/'.cache/huggingface/download').glob('*.'+sha+'.incomplete'))
  if len(candidates)!=1:raise ValueError(f'Expected one partial for {name}/{filename}: {candidates}')
  partial=candidates[0]
 attempts=0
 while partial.stat().st_size<size:
  start=partial.stat().st_size;end=min(size-1,start+16*1024*1024-1)
  try:
   url=f'https://modelscope.cn/models/Qwen/{name}/resolve/master/{filename}?offset={start}'
   with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
    r.raise_for_status()
    assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}',(r.status_code,r.headers.get('Content-Range'))
    payload=r.content
    assert len(payload)==end-start+1
   with partial.open('ab') as f:f.write(payload)
   print(json.dumps({'model':name,'file':filename,'bytes':partial.stat().st_size,'total':size}),flush=True)
   attempts=0
  except Exception as e:
   attempts+=1;print('RETRY',name,filename,start,repr(e),flush=True)
   if attempts>=5:raise
   time.sleep(3)
 h=hashlib.sha256()
 with partial.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 assert partial.stat().st_size==size and h.hexdigest()==sha,(name,filename,'checksum mismatch')
 if partial!=dest:partial.replace(dest)
 return {'model':name,'file':filename,'size':size,'sha256':sha,'verified':True}
jobs=[]
for name,rev in models.items():
 r=requests.get(f'https://hf-mirror.com/api/models/Qwen/{name}/revision/{rev}?blobs=true',timeout=30);r.raise_for_status();j=r.json()
 for item in j['siblings']:
  if item['rfilename'].endswith('.safetensors'):jobs.append((name,item))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(lambda args:blob(*args),jobs))
(root/'models/checksums.json').write_text(json.dumps(results,indent=2))
(root/'models/revisions.json').write_text(json.dumps(models,indent=2))
print('ALL_MODELS_VERIFIED',flush=True)
