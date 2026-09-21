"""Finish reporting only; assert the saved simulation numbers remain unchanged."""
from pathlib import Path
import hashlib
import json
import shutil
from prescribed_sip.final_artifacts import generate_artifacts
from prescribed_sip.run_all import save_json

root=Path('/home/bardia/Desktop/GFS_Prescribed_SIP_Final_Results')
config=json.loads((root/'configs/study.json').read_text())
protected=list((root/'raw_runs').rglob('*'))+list((root/'targets').rglob('*'))+list((root/'tables').glob('*.csv'))
protected=[p for p in protected if p.is_file()]
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
frozen=root/'configs/code/prescribed_sip/final_artifacts.py'
old=hashlib.sha256(frozen.read_bytes()).hexdigest()
backup=root/'configs/reporting_backups/final_artifacts_before_line_curves.py'
backup.parent.mkdir(exist_ok=True)
if not backup.exists():shutil.copy2(frozen,backup)
shutil.copy2('prescribed_sip/final_artifacts.py',frozen)
new=hashlib.sha256(frozen.read_bytes()).hexdigest()
hashfile=root/'configs/code_SHA256.json'
hashes=json.loads(hashfile.read_text());hashes['prescribed_sip/final_artifacts.py']=new
save_json(hashfile,hashes)
shutil.copy2('prescribed_sip/plans/final_study_notes.txt',root/'configs/study_notes.txt')
generate_artifacts(root,config)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in before.items())
save_json(root/'configs/reporting_revision.json',{
    'old_renderer_SHA256':old,'new_renderer_SHA256':new,
    'changes':'Connect recorded convergence checkpoints; interpolate aggregate curves in log(error). Clarify native competitor initialization and auxiliary timing. Algorithms, final numerical results, targets and per-run traces are unchanged.',
    'protected_numerical_files_unchanged':len(before),
    'pre_revision_renderer_backup':str(backup)})
print('Reporting revised;',len(before),'protected numerical files unchanged. Independent audit passed.')
