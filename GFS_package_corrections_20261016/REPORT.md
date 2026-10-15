# GFS package correction report

Approved corrections implemented on `codex/gfs-numerical-corrections-20261016`, based on `7e20e9ccbf4eb09e13464899f89e681e0bb1b864` (`gfs-yves`). No merge, push, publication, version change or release was performed. The initial proposal remains preserved in PROPOSED_CORRECTIONS.md; its approval gate was satisfied by the user's subsequent explicit approval.

## Confirmed defects and corrections

| Defect | Correction | Compatibility consequence |
| --- | --- | --- |
| `int(sum(pi))` truncates totals just below an integer. The historical N=144 EP construction inferred 3, 7, 15 and 31 for intended n=4, 8, 16 and 32. | Compensated summation, nearest-integer validation within `max(1e-10, 16*float64_eps*max(N, abs(total), 1))`. Already supplied probabilities are not renormalized. | Corrected n may be one larger than before; downstream references and DI can change. Genuinely fractional totals now raise ValueError. |
| Supplied centers were ignored by the public OT and expanded dispatchers. The weighted branch silently fell back to random restarts when their count differed from K. | Validate finite `(K, 2)` supplied centers and honor them for all three methods. Reject incompatible arrays rather than discard them. | Explicitly sample-initialized OT/expanded calls can change numerical results. No-centers initialization options remain available. |
| Invalid n/r, nonfinite coordinates and invalid weights were insufficiently validated. | Require positive integral n/r, n<=N, divisibility, finite coordinates and positive finite inputs. Verify fitted inclusion mass matches K*r, including recursive fractional-zone targets. | Invalid calls now fail clearly. Integral floats and booleans are rejected as targets; NumPy integer scalars are supported. Explicit-n weight normalization is retained. |
| No consistent public seed covered preliminary clustering, Numba TSP and DI workers. | Add optional keyword-only `random_state` to FIPBalancedNMeans, OpenTSPSolver and DensityDisparity. Use local streams and per-sample seeds independent of workers/batch positions. | Existing positional calls and unseeded defaults remain. Opt-in seeded replay is supported in the same software environment without consuming caller NumPy or Numba global RNG state. |

Only three production files changed: `graphical_sampling/population.py`, `graphical_sampling/clustering/fip_balanced_nmeans.py`, and `graphical_sampling/index/density_disparity.py`. New regression tests and task evidence are separate additions.

## Algorithm and scientific boundaries

DI retains its historical weighted preliminary method, aggregation/sign formula, Euclidean medoid and matching objectives, radial top-hat KDE with scalar Scott bandwidth, refinement and hardening. This patch does not substitute the separate manuscript-oriented OT/box-KDE implementation. No manuscript file or saved result/figure was edited, and no original Monte Carlo or completed 480-run evaluation was rerun.

TSP retains the same randomized greedy construction, jitter, two-opt candidates, first strictly better route selection, 15 restarts and large-input KD-tree fallback. Exact-distance argmin behavior, refinement argsort, medoid near-tie rules and matching are unchanged. Optional seeded calls use a local Generator rather than attempting to reproduce the old global RNG sequence. Reordering population rows or the indices within a sample can change the numerical path; only identical input ordering/settings is promised. Serial/parallel and changed *batch* order replay are tested. Custom subclasses overriding `_process_one_sample` keep their signature and remain responsible for their own seed propagation.

Convergence settings are deliberately unchanged. The public OT helper still uses epsilon=.01, max_iter=100, sinkhorn_iter=100, tol=1e-4; FIP's tol/max_iter continue to govern weighted clustering rather than that helper. This patch does not certify OT feasibility or convergence quality, fix near-tie hardening ambiguity, or establish suitability of unchanged OT internals for every grouped r>1 configuration. Grouped weighted sampling and recursive fractional zones are regression-tested. Changing numerical acceptance/hardening or OT mass conventions would require a separately scoped decision.

Corrections can materially change future DI when the old n or initialization defect was triggered. They neither invalidate nor certify the historical scientific analyses. The uncertainties in DI_implementation_audit.md about original per-run n, provenance, replication counts and manuscript correspondence remain. Preserving an old artifact does not establish that it evaluated the intended method.

## Tests and controlled comparisons

Before production edits, the new defect suite produced **45 failures and 17 passes** (`regression_red.log`). The first corrected run passed all **62** targeted cases (`regression_green.log`). Added guards cover Numba-global RNG isolation, exact-distance TSP ties, recursive fractional zones, historical subclass/positional APIs and medoid tie behavior. **All 68 correction/compatibility regression cases pass. The final full package run is 114 passed / 3 pre-existing failures** (`final_tests.log` and progress.json). The broader `tests/` suite retains exactly the three baseline failures listed below; no unrelated missing notebook was restored or rewritten.

Three failures already existed before this task, all caused by missing `simulations/jupyters/17_graphical_maxent_die.ipynb`:

- `test_wide_two_part_exchange_moves_six_percent_and_shades_both_regions`
- `test_mean_only_microscopic_start_approaches_exact_target_without_entropy`
- `test_notebook_executes_reduced_study_and_exposes_expected_results`

The baseline was 46 passed / 3 failed. Test execution was scoped to the package's `tests/` directory, rather than recursively launching unrelated historical study suites. Libraries were limited to one thread, with at most two DI test workers; no experiment runner was launched. See CODE_REVIEW.md for independent review and additional small compatibility probes.

Four fixed cases were compared against a freshly extracted, hash-verified historical archive in separate Python processes. Both variants used the existing unseeded API with explicitly controlled common Python/Numba legacy streams (seed 317). This separates the approved n/initialization fixes from the new optional RNG API. Inputs and source provenance are saved in controlled_inputs.json; full paths, labels, centers and DI are saved in controlled_historical.json and controlled_corrected.json.

| Fixed case | Old/new n | Historical DI | Corrected DI | Path/partition |
| --- | --- | ---: | ---: | --- |
| Square, N=4 | 2 / 2 | 0 | 0 | Identical |
| Clustered, N=24 | 4 / 4 | 0 | 0 | Identical |
| Clustered roundoff, N=144 | 3 / 4 | 0 | 0 | Changed; intended cardinality restored |
| First 24 Meuse units | 6 / 6 | 0.09856258954992361 | 0.09856258954992361 | Identical |

Two geometrically distinct supplied initializations on a symmetric square also verify the dispatcher defect: historical OT/expanded ignore the choice and return the same geometry, while corrected fits honor horizontal versus vertical centers. These are small fixed controls, not Monte Carlo estimates. Zero DI in the roundoff toy does not imply that the defect is harmless or quantify its historical impact. No confidence intervals or design-ranking claims are warranted from these controls. There were eight controlled old/new DI evaluations in total; regression tests additionally evaluate small synthetic scores.

## Preservation and historical reproduction

`final_preservation.json` verifies all 5,287 protected files: **5,284 unchanged and exactly the three authorized source changes**, with no unexpected change or missing file. All 49 archived historical files match their pre-fix hashes. The archive SHA256 is `4f45e7293c6e8a788b8d8a4360677b4b1d40795f92b13c5d2e4d9d091850ec3b`. Unrelated working-tree changes recorded in git_status_before.txt remain outside this patch. The old branch and historical commits are retained. Current software versions are recorded in environment.json; cross-version/platform bitwise equality is not promised.

Inspect/retest the corrected branch without running historical studies:

```bash
cd /home/bardia/projects/graphical-sampling
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q
.venv/bin/python GFS_package_corrections_20261016/verify_preservation.py
```

To use historical source without overwriting the corrected checkout, extract the backup into a **fresh temporary directory**:

```bash
historical_dir=$(mktemp -d /tmp/gfs-historical.XXXXXX)
tar -xzf /home/bardia/projects/graphical-sampling/GFS_package_corrections_20261016/historical_package_backup.tar.gz -C "$historical_dir"
cd "$historical_dir"
/home/bardia/projects/graphical-sampling/.venv/bin/python -c 'import graphical_sampling; print(graphical_sampling.__file__)'
```

The printed import must point inside that temporary directory. Use the recorded historical environment for historical reproduction; no in-package legacy switch or original-history rewrite is needed.

## API and release assessment

No required signature migration for valid existing positional calls. Optional seeds accept integers in [0, 2**32) or None. Validation is intentionally stricter, and numerical backward compatibility is not guaranteed where a defect was active. Users requiring earlier numbers must use the historical snapshot and matching environment, rather than the corrected code.

**This is a reviewed local correction candidate, not a release-ready certification.** Scoped regressions and controlled compatibility checks pass; the repository-wide package test gate still has three pre-existing failures, and cross-environment release validation has not been performed. Resolve that gate separately, document the numerical-impact changes in release notes, and obtain explicit approval before any release, publication or merge.
