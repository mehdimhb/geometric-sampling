"""Check the combined package in an isolated export; preserve prior results."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ASTRONOMY = [
    'graphical_sampling/constrained.py',
    'simulations/jupyters/constrained_gfs_astronomy_demo.ipynb',
    'tests/test_constrained_gfs.py',
    'tests/test_constrained_gfs_notebook.py',
]
CHANGES = ASTRONOMY + [
    'graphical_sampling/clustering/fip_balanced_nmeans.py',
    'tests/test_fip_stable_ordering.py',
]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    path = HERE/'combined_verification.json'
    if path.exists():
        raise FileExistsError('Preserving prior verification; use a new evidence directory')
    for name in ASTRONOMY:
        committed = git('show', f'codex/constrained-gfs-astronomy:{name}')
        assert (ROOT/name).read_bytes() == committed, name
    fip_path = 'graphical_sampling/clustering/fip_balanced_nmeans.py'
    before = git('show', f'HEAD:{fip_path}')
    expected = before.replace(b'np.argsort(key)]', b'np.argsort(key, kind="stable")]')
    assert expected != before and (ROOT/fip_path).read_bytes() == expected
    snapshot = Path(tempfile.mkdtemp(prefix='gfs-combined-validation-'))/'snapshot'
    snapshot.mkdir()
    archive = subprocess.Popen(['git', 'archive', 'HEAD', 'graphical_sampling', 'tests',
        'prescribed_sip', 'simulations/jupyters/17_graphical_maxent_die.ipynb', 'pyproject.toml'],
        cwd=ROOT, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=archive.stdout, mode='r|') as tar:
        tar.extractall(snapshot, filter='data')
    assert archive.wait() == 0
    for name in CHANGES:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    env = os.environ.copy()
    env.update({key:'1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                   'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')})
    env['PYTHONPATH'] = str(snapshot)
    env['MPLCONFIGDIR'] = str(snapshot.parent/'matplotlib')
    python = str(ROOT/'.venv/bin/python')
    imported = subprocess.check_output([python, '-c',
        'import graphical_sampling; print(graphical_sampling.__file__)'],
        cwd=snapshot, env=env, text=True).strip()
    assert Path(imported).is_relative_to(snapshot)
    record = dict(snapshot=str(snapshot), imported_package=imported,
        base_HEAD=git('rev-parse','HEAD').decode().strip(),
        astronomy_HEAD=git('rev-parse','codex/constrained-gfs-astronomy').decode().strip(),
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CHANGES},
        command=[python,'-m','pytest','tests','-q'], stage='running',
        note='Committed package tests plus requested astronomy/stable additions. Untracked unrelated tests are excluded. Existing notebook contracts use reduced settings in the isolated snapshot; no original Monte Carlo or completed480 DI evaluation is run.')
    path.write_text(json.dumps(record, indent=2)+'\n')
    with (HERE/'combined_tests.log').open('w') as output:
        result = subprocess.run(record['command'], cwd=snapshot, env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    record.update(stage='complete',exit_code=result.returncode)
    path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key:record[key] for key in ('snapshot','exit_code')}, indent=2))
    print('\n'.join((HERE/'combined_tests.log').read_text().splitlines()[-12:]))


if __name__ == '__main__':
    main()
