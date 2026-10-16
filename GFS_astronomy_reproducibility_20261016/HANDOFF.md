# Astronomy and reproducibility integration handoff

Implementation and verification complete on `codex/gfs-numerical-corrections-20261016`. Astronomy merge commit: `5ff9ed4a6`. The following stable-sort commit changes one production line in FIPBalancedNMeans._generate_exact_clusters; tests/evidence are retained in this directory. Read REPORT.md for exact scope, API usage and numerical consequences.

The branch contains all current main commits, earlier DI corrections and all seven committed astronomy changes from 789d29aa9. Other historical/experimental branches and untracked study outputs are not automatically included. Main remains at dc2881af8; no main merge, package release, original Monte Carlo or completed 480 DI evaluation was performed.

Tests: 157 passed / zero failures in the clean combined export; 46 focused astronomy tests and 49 reproducibility/DI replay tests also passed. The original tied-score defect first produced 1 failure/5 passes. Independent read-only review found no actionable integration issue. Existing optional random_state reaches Numba starts/jitter; this patch adds no duplicate seed API or heuristic change.

Preservation: 5,286 of 5,287 protected files unchanged, with only the exact one-line stable argsort modification permitted. Astronomy additions match their committed source. All existing manuscripts, saved results/datasets/figures and historical backups remain unchanged. The astronomy source worktree's uncommitted blank notebook-test edit is backed up and left intact there; the integrated branch retains the committed test file.

No implementation work remains. Source hashes and the clean snapshot path are in combined_verification.json. Inspect commit/upstream state with:
```bash
cd /home/bardia/projects/graphical-sampling
git log -2 --oneline codex/gfs-numerical-corrections-20261016
git branch -vv
.venv/bin/python GFS_astronomy_reproducibility_20261016/verify_preservation.py
```
Focused verification command:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_fip_stable_ordering.py tests/test_clustering_initialization_replay.py tests/test_density_disparity_replay.py -q
```

Do not restore the locally deleted unrelated notebook17, stage unrelated dirty files, overwrite completed studies, remove either worktree, merge into main or release a package without a new instruction. The completed DI study's original HANDOFF.md is unchanged. Full tests used a temporary clean export with reduced notebook contracts and disabled output writing; a bare run in the original dirty checkout observes its unrelated missing notebook and extra untracked tests.
