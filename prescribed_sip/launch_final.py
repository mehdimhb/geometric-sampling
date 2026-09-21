"""Detach the local final-study scheduler from the chat/tool process lifetime."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

root=Path('/home/bardia/Desktop/GFS_Prescribed_SIP_Final_Results')
if root.exists():
    raise SystemExit('Output directory already exists; use its tested resume.sh rather than overwrite it.')
root.mkdir(parents=True)
repo=Path(__file__).resolve().parents[1]
environment=os.environ.copy()
environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   MPLCONFIGDIR='/tmp/gfs-final-mpl',PYTHONPATH=str(repo))
with (root/'execution.log').open('a',buffering=1) as log:
    process=subprocess.Popen([sys.executable,'-u','-m','prescribed_sip.final_study',
                              '--output',str(root)],cwd=repo,env=environment,
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True,close_fds=True)
data={'PID':process.pid,'started_UTC':datetime.now(timezone.utc).isoformat(),
      'output':str(root),'resume_command':'sh '+str(root/'resume.sh'),
      'local_computation_requires_chat_tokens':False}
(root/'launcher.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
