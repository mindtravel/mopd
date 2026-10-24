"""Read only English tokenizer files from the official ZIP using HTTP ranges."""
import io,json,zipfile
from pathlib import Path
import requests
URL="https://fastly.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/tokenizers/punkt_tab.zip"
SIZE=4319076
class RemoteZip(io.RawIOBase):
 def __init__(self):self.pos=0
 def seekable(self):return True
 def seek(self,n,whence=0):
  self.pos=n if whence==0 else self.pos+n if whence==1 else SIZE+n
  return self.pos
 def tell(self):return self.pos
 def read(self,n=-1):
  n=SIZE-self.pos if n<0 else min(n,SIZE-self.pos)
  if n<=0:return b""
  start=self.pos;end=start+n-1
  r=requests.get(URL,headers={"Range":f"bytes={start}-{end}"},timeout=(15,45))
  r.raise_for_status()
  if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZE}":raise ValueError((r.status_code,r.headers.get("Content-Range")))
  if len(r.content)!=n:raise ValueError("Length mismatch")
  self.pos+=n;return r.content
root=Path("/ssd/lzx/mopd/cache/nltk/tokenizers")
with zipfile.ZipFile(RemoteZip()) as z:
 for info in z.infolist():
  if info.filename.startswith("punkt_tab/english/") and not info.is_dir():
   dest=root/info.filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(info))
   print(info.filename,info.file_size,flush=True)
print("ENGLISH_READY",flush=True)
