"""Verify preserved research files and exact scope of this integration."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ALLOWED = ROOT/'graphical_sampling/clustering/fip_balanced_nmeans.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    before = json.loads((HERE/'protected_before.json').read_text())
    changed, missing = [], []
    for name, digest in before.items():
        path = Path(name)
        if not path.is_file():
            missing.append(name)
        elif sha(path.read_bytes()) != digest:
            changed.append(name)
    unexpected = sorted(set(changed)-{str(ALLOWED)})
    manifest = json.loads((HERE/'backup_manifest.json').read_text())
    archive = HERE/manifest['archive']
    backup_ok = sha(archive.read_bytes()) == manifest['archive_SHA256']
    with tarfile.open(archive) as tar:
        for name, digest in manifest['files'].items():
            backup_ok &= sha(tar.extractfile(name).read()) == digest
        dirty_name = 'astronomy_worktree_uncommitted/test_constrained_gfs_notebook.py'
        backup_ok &= sha(tar.extractfile(dirty_name).read()) == manifest['astronomy_uncommitted_test_SHA256']
    worktree_test = ROOT/'.worktrees/constrained-gfs-astronomy/tests/test_constrained_gfs_notebook.py'
    worktree_unchanged = sha(worktree_test.read_bytes()) == manifest['astronomy_uncommitted_test_SHA256']
    old_fip = subprocess.check_output(['git','show',manifest['HEAD']+':'+str(ALLOWED.relative_to(ROOT))],cwd=ROOT)
    exact_patch = ALLOWED.read_bytes() == old_fip.replace(b'np.argsort(key)]',b'np.argsort(key, kind="stable")]')
    old_main = 'dc2881af84a41e4f7658b6d21401e58470871b55'
    main_refs = {name:subprocess.check_output(['git','rev-parse',name],cwd=ROOT,text=True).strip()
                 for name in ('main','origin/main','origin/backup/main-before-di-20261016')}
    result = dict(passed=not (unexpected or missing) and backup_ok and worktree_unchanged
                         and exact_patch and all(value==old_main for value in main_refs.values()),
                  protected_count=len(before), unchanged_count=len(before)-len(changed)-len(missing),
                  authorized_changed_files=changed, unexpected_changes=unexpected, missing_files=missing,
                  backup_valid=bool(backup_ok), source_worktree_test_unchanged=worktree_unchanged,
                  exact_one_line_production_patch=exact_patch, main_refs=main_refs)
    (HERE/'preservation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['passed'], result


if __name__ == '__main__':
    main()
