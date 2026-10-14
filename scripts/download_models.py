import os,json
from pathlib import Path
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
root=Path(os.environ['MOPD_ROOT'])
models={'Qwen3-4B':'1cfa9a7208912126459214e8b04321603b3df60c','Qwen3-4B-Base':'906bfd4b4dc7f14ee4320094d8b41684abff8539'}
for name,rev in models.items():
 print('DOWNLOAD',name,rev,flush=True)
 path=snapshot_download('Qwen/'+name,revision=rev,local_dir=root/'models'/name,max_workers=3,allow_patterns=['*.json','*.safetensors','*.txt','README.md','LICENSE'])
 print('COMPLETE',path,flush=True)
(root/'models'/'revisions.json').write_text(json.dumps(models,indent=2))
