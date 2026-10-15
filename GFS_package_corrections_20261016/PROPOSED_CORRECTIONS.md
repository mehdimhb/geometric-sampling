# Proposed bounded package corrections — approval required

Status: defects reproduced; historical backup saved; no production code modified. No release/version change. The user's final instruction requires approval before a correction that could materially change DI. This proposal identifies those consequences before implementation.

## Confirmed defects

1. Population.__init__ infers n using int(np.sum(pi)). Arithmetic-only EP probes with N=144 infer 3,7,15,31 for intended 4,8,16,32. Compensated summation alone is insufficient: its results can also remain slightly below the integer. See confirmed_defects.json.
2. FIPBalancedNMeans._get_labels_centroids ignores supplied centers in the OT dispatcher and forwards its grid. A recording subclass confirms that input [[0,1],[1,0]] becomes [[0,0],[1,0]], without running OT. The expanded branch also uses this grid even when centers were supplied.
3. The weighted branch silently rejects supplied centers whose count differs from K and falls back to random restarts. This masks the target-n defect and can destroy intended sample initialization.
4. Public FIPBalancedNMeans/DensityDisparity/OpenTSPSolver do not expose a consistently propagated seed. TSP uses a Numba RNG, independent of seeding NumPy's Python RNG. Unseeded weighted restarts and internal OT RNGs also need explicit handling for reproducible seeded calls.

## Proposed changes

- In graphical_sampling/population.py: compute an accurate total, accept an inferred integer only within a documented float64 tolerance (absolute floor 1e-10, enlarged only by an explicit floating-error bound 16*eps*max(N,abs(total),1)); reject genuinely fractional totals. Validate explicit n as a positive integer no larger than population size and require finite valid inputs. Preserve explicit-n weight normalization and do not silently renormalize already supplied inclusion probabilities when n is inferred.
- In graphical_sampling/clustering/fip_balanced_nmeans.py: validate supplied center shape/finiteness and use supplied centers rather than the grid when present. Apply this consistently to weighted, OT and expanded branches, while keeping each algorithm and its existing no-centers initialization option. Reject an incompatible supplied center count with ValueError instead of silently substituting random initialization. Validate intended K=n/r and test the actual constructed cluster count. The requirement K=n applies to the default r=1; preserve supported r>1 grouped sampling.
- Add an optional trailing keyword random_state (integer or None) to clustering/TSP/DI entry points and propagate reproducible per-sample/algorithm seeds. Keep None as the existing unseeded default. Seeded DI results should replay across serial/parallel execution in the same software environment. Preserve the current weighted/expanded/OT choices, TSP candidate algorithm and number of restarts. Do not globally reseed the caller's Python/NumPy RNG as an undocumented side effect.
- Review exact tie and convergence behavior with regression checks. Do not change the DI formula, kernel/bandwidth, medoid/matching distance objectives, near-tie tolerances, OT entropy coefficient or effective default stopping thresholds in this patch. Current OT helper settings are hardcoded (tol=1e-4,max_iter=100,sinkhorn_iter=100); report that limitation rather than silently tightening convergence. Any additional change to tie hardening or OT numerical acceptance that could move DI substantially needs a separately explained decision.

## Numerical and compatibility consequences

The intended n may increase from n−1 to n. The package can then accept sample initialization that it previously discarded. OT (and explicitly supplied expanded initialization) will start from the actual supplied centers instead of its former grid. These corrections can change partitions, medoids, matching, translations and DI substantially. They do not change the DI aggregation formula or establish that old manuscript conclusions are invalid. Re-running historical analyses with the corrected implementation is not guaranteed to reproduce their old numbers.

Existing valid integer-total calls and calls without supplied centers retain their algorithm choices. New optional keyword arguments are additive. Previously accepted genuinely fractional totals, invalid n or incompatible center arrays will raise clear errors; this is an intentional validation tightening. No switch will silently change weighted clustering to OT. No tolerance-based merging of near-equal mathematical alternatives is proposed. Cross-version/platform bitwise equality is not promised.

Historical reproduction remains available through historical_package_backup.tar.gz plus backup_manifest.json, recorded environment.json and the existing Git commits/tag. The backup captures the actual current files before correction; restoring history does not require rewriting Git or overwriting the current working tree.

## Regression and controlled validation after approval

Add tests before production changes and demonstrate the old failures:

- Near-integer totals from below/above, true fractional totals, NaN/Inf, invalid explicit n, and unchanged supplied pi for inferred n.
- Valid supplied centers reaching OT/expanded rather than the grid, malformed shapes rejected, weighted mismatch rejected, and existing no-centers options retained.
- Default r=1 produces n constructed clusters and accepts n starting centers; supported r>1 keeps K=n/r.
- Explicit seeds replay TSP/clustering/DI, including n_jobs=1 versus a bounded n_jobs=2, without leaking Python NumPy global RNG changes.
- Existing score formula and representative/kernel options remain unchanged.

Compare historical backup and corrected code on a few small fixed toy/clustered cases and a small Meuse control, with input vectors, samples, seeds, paths/partitions and DI differences saved. Do not rerun Monte Carlo studies or any of the 480 completed evaluations. Limit BLAS/OpenMP/Numba threads to one and workers to two, remaining below approximately55% of available capacity.

The current tests/ baseline is 46 passed, 3 failed. All three failures are in tests/test_maxent_notebook.py because simulations/jupyters/17_graphical_maxent_die.ipynb is already missing in the user's working tree. Its deletion existed before this task. Do not restore, replace or change that unrelated notebook automatically. Preserve and report these baseline failures separately from new regressions; a fully clean release gate is not currently satisfied.

## Approval requested

Approve the corrections above with the stated possibility of changed future DI values, while preserving every existing manuscript/result/figure and the historical backup. Until approval, only inspection, backups, diagnostics and baseline tests are completed; production changes and old-versus-corrected DI comparisons remain pending.
