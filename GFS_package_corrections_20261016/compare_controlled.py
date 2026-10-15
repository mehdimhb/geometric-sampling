"""Four fixed DI cases and two initialization controls; no simulation study.

Extract the hash-verified pre-fix archive into a temporary directory and import
old/new sources in separate processes. Unseeded API compatibility is compared
under common, explicitly controlled Python and Numba legacy RNG states. The
new opt-in RNG is deliberately not conflated with that comparison.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-gfs-corrections')


def worker(source, input_path, output_path):
    sys.path.insert(0, str(source))
    import numpy as np
    from numba import njit
    import graphical_sampling
    from graphical_sampling import Population
    from graphical_sampling.clustering import FIPBalancedNMeans
    from graphical_sampling.index import DensityDisparity

    assert Path(graphical_sampling.__file__).resolve().is_relative_to(source.resolve())

    @njit
    def seed_numba(seed):
        np.random.seed(seed)

    def reset():
        np.random.seed(317)
        seed_numba(317)

    inputs = json.loads(input_path.read_text())
    cases = []
    for case in inputs['cases']:
        x, pi = np.array(case['coords']), np.array(case['pi'])
        sample = np.array(case['sample'])
        pop = Population(x, pi)
        reset()
        clustering = FIPBalancedNMeans(pop.n, init_clust_method='weighted')
        clustering.fit(pop, init_centroids=pop.coords[sample])
        reset()
        di = float(DensityDisparity(pop, n_jobs=1, representative='nmedoids').score(sample[None])[0])
        cases.append(dict(name=case['name'], requested_n=len(sample), inferred_n=pop.n,
                          K=clustering.K, sample_centers_accepted=clustering.K == len(sample),
                          path=clustering.path_order.tolist(), labels=clustering.labels.tolist(),
                          centroids=clustering.centroids.tolist(), DI=di))

    square = np.array(inputs['cases'][0]['coords'])
    controls = []
    for method in ('ot', 'expanded'):
        for name, centers in [('horizontal', [[0., .5], [1., .5]]),
                              ('vertical', [[.5, 0.], [.5, 1.]])]:
            reset()
            solver = FIPBalancedNMeans(2, init_clust_method=method, split_size=.25)
            labels, centroids = solver._get_labels_centroids(square, np.full(4, .5), np.array(centers))
            controls.append(dict(method=method, initialization=name, labels=labels.tolist(),
                                 centroids=centroids.tolist()))
    output_path.write_text(json.dumps(dict(imported_from=str(graphical_sampling.__file__),
                                           cases=cases, initialization_controls=controls), indent=2)+'\n')


def main():
    import numpy as np
    manifest = json.loads((HERE/'backup_manifest.json').read_text())
    archive = HERE/manifest['backup']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['backup_SHA256']
    meuse_path = ROOT/'simulations/populations/real/meuse.csv'
    meuse = np.genfromtxt(meuse_path, delimiter=',', names=True)
    rng = np.random.default_rng(20261016)
    cluster_centers = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    clustered = np.repeat(cluster_centers, 6, axis=0) + rng.normal(0, .04, (24, 2))
    roundoff = np.repeat(cluster_centers, 36, axis=0) + rng.normal(0, .06, (144, 2))
    pi = 4*np.ones(144)/144
    pi *= 4/pi.sum()
    cases = [dict(name='square_compatibility', coords=cluster_centers.tolist(), pi=[.5]*4, sample=[0, 3]),
             dict(name='clustered24_compatibility', coords=clustered.tolist(), pi=[1/6]*24, sample=[1, 7, 13, 19]),
             dict(name='clustered144_roundoff', coords=roundoff.tolist(), pi=pi.tolist(), sample=[5, 41, 77, 113]),
             dict(name='Meuse_first24_compatibility', coords=np.column_stack((meuse['x'][:24], meuse['y'][:24])).tolist(),
                  pi=[.25]*24, sample=[0, 4, 8, 12, 16, 20])]
    input_path = HERE/'controlled_inputs.json'
    input_path.write_text(json.dumps(dict(input_seed=20261016, common_legacy_rng_seed=317,
        meuse_source=str(meuse_path.relative_to(ROOT)), meuse_SHA256=hashlib.sha256(meuse_path.read_bytes()).hexdigest(),
        cases=cases), indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='gfs-historical-') as tmp:
        oldroot = Path(tmp)
        with tarfile.open(archive) as tar:
            tar.extractall(oldroot, filter='data')
        for rel, digest in manifest['files'].items():
            assert hashlib.sha256((oldroot/rel).read_bytes()).hexdigest() == digest, rel
        for name, source in [('historical', oldroot), ('corrected', ROOT)]:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(source),
                            str(input_path), str(HERE/f'controlled_{name}.json')], cwd=oldroot,
                           check=True, env=os.environ.copy())
    old = json.loads((HERE/'controlled_historical.json').read_text())
    new = json.loads((HERE/'controlled_corrected.json').read_text())
    rows = []
    for a,b in zip(old['cases'], new['cases']):
        row = dict(name=a['name'], historical_n=a['inferred_n'], corrected_n=b['inferred_n'],
                   historical_DI=a['DI'], corrected_DI=b['DI'], DI_difference=b['DI']-a['DI'],
                   same_labels=a['labels']==b['labels'], same_path=a['path']==b['path'])
        if 'compatibility' in a['name']:
            assert a['labels']==b['labels'] and a['path']==b['path'] and a['DI']==b['DI'], row
        rows.append(row)
    assert rows[2]['historical_n']==3 and rows[2]['corrected_n']==4
    (HERE/'controlled_comparison.json').write_text(json.dumps(dict(
        historical_backup_SHA256=manifest['backup_SHA256'], cases=rows,
        interpretation='Fixed cases only, not Monte Carlo estimates or evidence about historical design rankings. Common legacy RNG states isolate compatibility/target-n effects; the new optional seed uses a separate local Generator.',
        DI_evaluations=8), indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        worker(*(Path(arg) for arg in sys.argv[2:]))
    else:
        main()
