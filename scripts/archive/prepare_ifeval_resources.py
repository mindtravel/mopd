"""Fetch official NLTK data with a total transfer deadline, project-local only."""
import hashlib,io,json,subprocess,zipfile
from pathlib import Path
root=Path("/ssd/lzx/mopd/cache/nltk");root.mkdir(parents=True,exist_ok=True)
zip_path=root/"punkt_tab.zip"
url="https://fastly.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/tokenizers/punkt_tab.zip"
subprocess.run(["curl","-fL","--connect-timeout","15","--max-time","90","--retry","1","-o",str(zip_path),url],check=True)
target=root/"tokenizers";target.mkdir(exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
 for name in z.namelist():
  if not (target/name).resolve().is_relative_to(target.resolve()):raise ValueError("Unsafe path")
 z.extractall(target)
(root/"source.json").write_text(json.dumps({"url":url,"sha256":hashlib.sha256(zip_path.read_bytes()).hexdigest()},indent=2))
print("READY",flush=True)
