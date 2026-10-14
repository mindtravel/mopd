"""Wait for the already-running download; never restart it implicitly."""
import json,os,time,subprocess
from pathlib import Path
root=Path(os.environ['MOPD_ROOT'])
status=root/'logs'/'opd-status.json'
def report(state,**kw):
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'time':time.time(),**kw},indent=2));tmp.replace(status)
download_pid=int(os.environ['MOPD_DOWNLOAD_PID'])
report('waiting_for_models',download_pid=download_pid)
while not (root/'models'/'revisions.json').exists():
    try:
        os.kill(download_pid,0)
        cmd=Path(f'/proc/{download_pid}/cmdline').read_bytes()
        if b'resume_verified.py' not in cmd and b'download_models.py' not in cmd: raise ProcessLookupError()
    except (ProcessLookupError,FileNotFoundError):
        report('download_stopped_without_completion');raise SystemExit(1)
    time.sleep(15)
report('training')
with (root/'logs'/'opd-train.log').open('a',buffering=1) as log:
    result=subprocess.run([str(root/'mopd_train/bin/python'),'-u',str(root/'scripts/train_opd.py'),'--config',str(root/'configs/single_teacher.json')],stdout=log,stderr=subprocess.STDOUT)
summary=root/'runs/20261013-qwen3-4b-lora-opd/summary.json'
report('passed' if result.returncode==0 and summary.exists() else 'failed',returncode=result.returncode,summary=str(summary))
raise SystemExit(result.returncode)
