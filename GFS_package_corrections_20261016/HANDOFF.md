# GFS package corrections handoff

Approved corrections are implemented, tested and independently reviewed on `codex/gfs-numerical-corrections-20261016`. The original branch `gfs-yves` remains at `7e20e9ccbf4eb09e13464899f89e681e0bb1b864`. No merge, push, release, publication or version change occurred.

Read REPORT.md for the final scientific/software boundaries, API impact and historical reproduction instructions. The source corrections restore near-integer target n, validate/honor supplied centers for all three preliminary methods, and add optional reproducible local random_state streams. DI formula/objectives/kernel/ties and effective OT convergence settings remain unchanged.

Final verification: all 68 added regression/compatibility cases pass; full package tests are 114 passed / the same 3 pre-existing missing-notebook17 failures. Independent review found no blocking issue. Three fixed old/new compatibility cases match exactly; the roundoff toy restores n=4 from3 with changed path/partition but DI=0 in both. These controls do not certify historical scientific conclusions. Controlled comparisons use only eight DI scores; the small synthetic regression tests also compute scores. No historical Monte Carlo or completed480 evaluation was rerun.

Preservation: final_preservation.json checks5,287 original files:5,284 unchanged, exactly3 approved source changes. All49 historical archived files retain their hashes. The backup archive and its manifest/environment remain in this directory; the archive SHA256 is4f45e7293c6e8a788b8d8a4360677b4b1d40795f92b13c5d2e4d9d091850ec3b. Manuscripts, prior figures/results, the480 study and unrelated working-tree changes are untouched.

No implementation work remains. Release is not authorized; the existing three test failures and cross-environment release validation remain separate readiness limitations. Existing OT convergence/hardening limitations remain explicitly documented. Do not modify the missing notebook merely to make this patch green, integrate the patch into another branch, or rerun studies without a new instruction.

Inspect/retest from the project root (no experimental resume needed):
```bash
cd /home/bardia/projects/graphical-sampling
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -m pytest tests -q
.venv/bin/python GFS_package_corrections_20261016/verify_preservation.py
git log -1 --oneline codex/gfs-numerical-corrections-20261016
```

Evidence: regression_red.log, regression_green.log, final_tests.log, CODE_REVIEW.md, controlled_inputs.json, controlled_historical.json, controlled_corrected.json, controlled_comparison.json, final_preservation.json and corrected_source_manifest.json. Completed study HANDOFF.md remains unchanged.
