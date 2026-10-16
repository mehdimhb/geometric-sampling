# Astronomy integration and targeted reproducibility correction

The existing DI-correction branch now incorporates all seven committed astronomy changes from `codex/constrained-gfs-astronomy` (`789d29aa9`) through merge commit `5ff9ed4a6`. It retains the earlier DI corrections and all existing main commits. This consolidates those two development lines; it does not claim to contain every historical/experimental branch or untracked study directory. Main and the completed 480-run DI study remain unchanged. No package release or original simulation rerun occurred.

## Exact production scope

The requested reproducibility patch modifies **one line in one existing production function**:

- `graphical_sampling/clustering/fip_balanced_nmeans.py`, `FIPBalancedNMeans._generate_exact_clusters`: `np.argsort(key)` becomes `np.argsort(key, kind="stable")`.

Within each cluster, positions originate in `np.flatnonzero(labels == lab)`. Equal computed scores now retain that input order, even when global IDs are scrambled. Unequal scores retain their previous numerical ordering. No rounding or tolerance is introduced to merge near ties; no distance, clustering equation, sample-selection rule or quota/hardening equation changes.

`OpenTSPSolver` requires **no further modification in this patch**. Commit902ed8ef4 already provides the optional keyword-only `random_state` API in OpenTSPSolver and FIPBalancedNMeans. The constructor validates the seed; `_solve_numba_accelerated` constructs a per-call local Generator; `_run_multi_start_numba` uses it for start nodes and `_greedy_jitter_numba` uses it for nearest-neighbor jitter. The default None path continues using the existing unseeded Numba random stream. The greedy heuristic, jitter range, distance criterion, `_two_opt_open_numba`, default 15 restarts and large-input fallback remain unchanged.

Usage:

```python
from graphical_sampling.clustering.fip_balanced_nmeans import (
    FIPBalancedNMeans, OpenTSPSolver,
)

path = OpenTSPSolver(random_state=42).solve(points)
clustering = FIPBalancedNMeans(n=sample_size, random_state=42)
clustering.fit(population, init_centroids=sample_centers)
```

Replay assumes identical input ordering/settings and the same numerical environment. None does not request deterministic random paths. The intentional default-output change is the order of units with equal within-cluster scores; that stable ordering applies with or without a seed.

## Astronomy files integrated

These four files are byte-identical to the committed astronomy branch:

- `graphical_sampling/constrained.py`: constrained Madow/GFS construction, diagnostics, repair/search and independent reference solvers.
- `simulations/jupyters/constrained_gfs_astronomy_demo.ipynb`: existing astronomy demonstration notebook and saved contents.
- `tests/test_constrained_gfs.py`: prototype regression coverage.
- `tests/test_constrained_gfs_notebook.py`: notebook structure and reduced-execution contract.

No existing astronomy function was redesigned for this integration. The source worktree has an uncommitted edit replacing its notebook test file with a blank. That edit was backed up and left untouched there; the integrated branch retains the committed tests. The notebook is a new addition to this branch, not an edit of the original JSSAM manuscript or saved DI artifacts. The existing package root exports remain unchanged; prototype callers import `graphical_sampling.constrained` directly.

## Focused tests and independent review

New `tests/test_fip_stable_ordering.py` contains six focused cases: mixed exact ties across quicksort partition boundaries; preserved strict-score ordering; all-zero single-cluster scores; seeded replay with one and 15 restarts; and equivalence of implicit versus explicit 15 restarts. The tied fixture uses scrambled identifiers to distinguish input order from ID sorting.

Before the source edit, the suite produced **one expected failure / five passes**: the default sort displaced 20 of 33 positions in the mixed-tie fixture. Afterward, the combined focused ordering, initialization, TSP and DI replay suites passed **49 tests**. The astronomy module and reduced notebook checks passed **46 tests**. Logs are stable_red.log, reproducibility_tests.log and astronomy_tests.log.

**The full clean combined package passed 157 tests, zero failures (20 plotting warnings).** Its exact source hashes are recorded in combined_verification.json and combined_tests.log. Its snapshot includes committed package tests plus the requested astronomy/stable additions, and restores the locally deleted MaxEnt notebook only inside a temporary export. It excludes unrelated untracked notebook tests. No original file was restored in the user's checkout. Existing notebook tests run reduced validation settings with output writing disabled; no full astronomy study, original Monte Carlo or completed 480 DI evaluation was run. Libraries are single-threaded, with at most two DI test workers, preserving the approximately 55% CPU policy.

A separate read-only reviewer found no actionable integration issue. It checked astronomy compatibility with corrected Population, exact source integration, stable input-order semantics, test scope, and unchanged seeded/unseeded TSP behavior. It did not certify historical results, general convergence or scientific optimality. See CODE_REVIEW.md.

## Backup, preservation and numerical consequences

Before edits, pre_integration_backup.tar.gz captured the affected existing FIP source, existing initialization/replay tests and the source worktree's uncommitted notebook-test contents. backup_manifest.json records SHA256 values and both branch tips; astronomy_worktree_diff.patch preserves that worktree edit. Local backup branch `backup/corrections-before-astronomy-20261016` preserves 902ed8ef4. The older historical package archive remains unchanged under GFS_package_corrections_20261016/.

preservation.json verifies the 5,287 previously protected files against this task's pre-edit snapshot. The only permitted change among them is the exact one-line stable sort; astronomy additions are verified against their source commit. Existing manuscript, result, dataset and figure hashes remain unchanged. The source astronomy worktree edit is also unchanged. Main and its previously pushed backup remain at dc2881af8.

Stable ties can change balanced partitions, medoids, translations and therefore future DI values when tied units cross a balance boundary. Strictly ordered scores retain their order. This correction does not establish that old numerical results are unaffected or invalid, and no historical simulations are claimed to have used fixed seeds. Existing historical records and saved results are preserved rather than regenerated.

## Inspect or verify

```bash
cd /home/bardia/projects/graphical-sampling
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_fip_stable_ordering.py \
  tests/test_clustering_initialization_replay.py tests/test_density_disparity_replay.py -q
.venv/bin/python GFS_astronomy_reproducibility_20261016/verify_preservation.py
```

The combined package test log is authoritative for the clean candidate. A bare test run in the original dirty checkout still observes the unrelated deleted MaxEnt notebook and extra untracked tests. No main integration or package release is authorized by this branch consolidation.
