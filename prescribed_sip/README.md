# Prescribed-SIP simulation study

## Educational notebook

For a shorter target-generation cell, open
[GFS_prescribed_SIP_demo_simple.ipynb](../simulations/jupyters/GFS_prescribed_SIP_demo_simple.ipynb).
It keeps the target walk's checks and diagnostics in a collapsed notebook helper,
leaving a 12-line target cell with the same downstream variables. The detailed
original remains in `simulations/jupyters/GFS_prescribed_SIP_demo.ipynb`.
The simple copy retains the user's population/budget settings and uses the
serial `n_jobs=1` tracking configuration for Original GFS; Directed repetitions use their own worker setting. Equal/unequal small-case checks compare
its target probabilities exactly with the detailed walk and execute later cells.

Open [GFS_prescribed_SIP_demo.ipynb](notebooks/GFS_prescribed_SIP_demo.ipynb)
with the repository's `.venv` Python kernel. Edit **code cell 1 — USER SETTINGS**,
then restart the kernel and run all cells. It shows the population/FIP formula,
original package design builder, random `Design.exchange` target construction,
support-based SIP calculations, and a direct `GreedyBestFirstSearch.run` call.
Target construction now starts from a **separate standard GFS design on a
reproducibly permuted order**, rather than `p0`. `target_permutation[position]`
is the original zero-based unit ID; the package order retains those original IDs,
so each prescribed inclusion probability stays attached to its unit. The frozen
SIP is read in permutation order and restored with the inverse permutation before
reconstruction. Original GFS and Directed both start from standard original-order
`p0`. The notebook prints the permutation, original-initial-to-target RMSE/NRMSE,
FIP error and exact sample-size confirmation. The permutation is included in
saved settings and frozen problem matrices; target walks and existing budgets
are otherwise unchanged. Pre-change notebooks/outputs are backed up under
`notebooks/backups/before_target_permutation_*/`.
The optional `RUN_GFS_DIRECTED` comparison uses the existing additional directed
exchange search on the same target and initial design. Its evaluation ceiling is
`DIRECTED_MAX_EVALUATIONS=100000`; it fits exchange
mass by the exact squared-error line minimum. Original GFS remains the primary
package optimizer. The notebook compares their convergence and final metrics.
Directed's earlier diagnostic median was lower, but this is not a universal ranking.

Directed now defaults to `N_REPETITIONS=30`, `DIRECTED_BATCH_SIZE=50` and
`DIRECTED_N_JOBS=6`. Every repetition starts from the same `p0` and frozen target,
with its own reproducible seed and fresh search state. Processes run independent
repetitions; Original GFS tracking remains serial. Set `DIRECTED_N_JOBS=1` for
serial execution. Counts measure scored exchange directions, not accepted moves.
Each run has its own maximum budget and stops early when its **raw SIP RMSE
is strictly below `1e-10`** (`DIRECTED_RMSE_STOP`, defaulting to `GFS_RMSE_STOP`).
Already-recovered Directed initial designs stop before any candidate scores.
Runs may also stop at no valid candidates or the existing time cap; all stopping reasons and valid final values
are retained. The Directed summary row uses means over all runs and sample SD
(`ddof=1`); it does not select the best run. These SDs describe optimizer
randomness on one fixed target, rather than population-to-population uncertainty.

All method tables and reconstruction traces retain raw SIP RMSE, calculated
exactly as before over all `i<j` pairs. Main scale-free reporting now uses
`rho_ij=(pi_ij-pi_i*pi_j)/sqrt(pi_i*(1-pi_i)*pi_j*(1-pi_j))`,
`RMSE_rho=sqrt(mean_valid_pairs((rho_hat-rho_star)**2))` and
`Recovery_rho=100*(1-RMSE_rho_final/RMSE_rho_initial)`.
`RHO_BOUNDARY_TOL=1e-12` excludes every pair involving a unit with prescribed
FIP at or below this tolerance or at or above `1-tolerance`. Included/excluded
pair counts are recorded in tables, per-run results and traces. No valid pairs
gives RMSE_rho=NA; zero/undefined initial error gives Recovery_rho=NA.
Old NRMSE and Recovery_N are removed from main output. Reporting uses the
same fixed FIP and pair mask for all methods/runs, and does **not** change the
existing raw-loss optimizer or stopping rules. Dependence error therefore need
not decrease at every raw-loss improvement. Unequal-FIP dependence errors
are computed from the actual pair residuals, not converted from a scalar RMSE.

Original and Directed comparison plots each show raw RMSE, RMSE_rho,
Recovery_rho, maximum SIP error, runtime, evaluations, support size and entropy.
Directed means/SD include all valid repetitions; each trace refers to the
incumbent selected by raw SIP error. Random target-walk diagnostics report
raw/dependence distance **from p0**, not reconstruction recovery toward Pi_star.

`DIRECTED_SAVE_RESULTS=True` saves seeds/settings, the frozen moment matrices,
per-run values, dependence traces, final supports and mean/SD statistics after each completed
repetition in a new `results/notebook_directed/N.../` folder. This directory is
separate from the completed paper study. Library threads are restricted to one
per worker process; reported run runtimes exclude queue time and the notebook
also reports total wall time. Previous notebooks/outputs are backed up under
`notebooks/backups/before_repetitions_*/`; the prior NRMSE notebook/outputs are
also retained under `notebooks/backups/before_rho_metrics_*/`. After updating, reload the notebook,
restart its kernel, edit USER SETTINGS and run all cells; changed definitions do
not affect a cell already running in an old kernel. No 30 × 100,000 experiment
was automatically launched while implementing this update.
Notebook reconstruction now defaults to **Original GFS + restarts**:
`GFS_AUTO_RESTART=True` rebuilds the package search frontier after exhaustion or
observer stagnation. `GFS_RESTART_EXCHANGES` controls valid random package exchanges
used to perturb the retained best design. All stages share the candidate/time
budget, newly scored restart seeds count once, and repeated stages producing no
neighbor scores stop explicitly. Progress and returned results report restart
counts and actual stopping reasons. Set `GFS_AUTO_RESTART=False` for the original
one-stage stopping protocol. This notebook extension does not change the completed
paper experiment or its frozen results; pre-change notebook outputs are backed up
under `notebooks/backups/before_restarts_*/`.
Defaults are one synthetic N=12,n=4 example, 1,000 accepted nontrivial target
exchanges, and at most 5,000 reconstruction candidate scores. Its target differs
from the frozen paper-study targets below; it does not overwrite study results.

For exactly N=284,n=40, keep `POPULATION="synthetic"` and edit `N` and `n`.
With `POPULATION="MU284"`, the repository CSV supplies 281 actual rows; the
notebook reports and uses that count. Its editable default size column is `P75`.
Large populations disable enumerated competitors, benchmark before GFS search,
and require `ALLOW_LONG_RUN=True` for protected runtime estimates over 30 minutes.
Target construction also has a large-population pilot and 30-minute time limit
unless explicitly allowed. No large search is run to create this notebook.
Support bookkeeping reads the package heap's stored fragments without creating
a temporary `_Sample` for every item. Target construction reuses the previous
aggregated distribution until a move is accepted; sample probabilities still
use `fsum`, and package exchange/merging/RNG behavior is unchanged. Profiles and
exact old/new target comparisons are in `notebooks/bookkeeping_smoke/`.
`KeyboardInterrupt` indicates an interrupted cell, not an invariant failure;
a target-construction cell that did not finish has not frozen a new target.

**Current scope:** the broad paper study is paused. The authorized experiment is
only two fixed N=12,n=4 problems, with 20 repetitions per stochastic method.
Outputs are in `results/N12_n4/`; final reports are
`reports/N12_n4_prescribed_SIP_report.md` and
`reports/N12_n4_paper_ready_results.md`. No N>12 run is launched by this command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/prescribed-sip-matplotlib \
  .venv/bin/python -u -m prescribed_sip.n12_proof
```

The command freezes two reachable random-exchange targets, benchmarks both,
prints a runtime estimate, and refuses full execution when the estimate exceeds
two hours. The high-precision sensitivity reuses the package search bytecode in
an isolated namespace with its score rounding removed. Main scores are D2.
DSD fits unit ordering within the existing projection construction. CPU/RAM and
actual library thread settings are recorded in `results/N12_n4/machine.json`.
Existing broad-study data and earlier pilot results are retained.

The primary optimized GFS method invokes the existing package's
`GreedyBestFirstSearch.run`. Additional exchange searches are comparisons and
ablations. Dedicated line populations (`geometry: 1d`) accompany general settings.
The original research files and existing user edits are preserved.

## Environment and reproduction

From the repository root, use Python 3.12 and the existing project virtual
environment. Install the repository and its dependencies, then the study dependencies:

```bash
python -m pip install -e .
python -m pip install -r prescribed_sip/requirements.txt
```

For deterministic serial evaluation-budget runs, restrict BLAS threading:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest prescribed_sip/tests -q
python -m prescribed_sip.run_all --config prescribed_sip/configs/smoke_test.yaml
```

With the existing environment, substitute `.venv/bin/python` for `python`.
The smoke test executes all artifact types and includes small, moderate, large,
sparse and ablation examples. It uses tiny budgets and is not scientific evidence
for comparative superiority.

Staged validation and full reproduction:

```bash
python -m prescribed_sip.run_all --config prescribed_sip/configs/small_validation.yaml
python -m prescribed_sip.run_all --config prescribed_sip/configs/small_complete.yaml
python -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml --stage small
python -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml --stage moderate
python -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml --stage large
python -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml --stage sparse
python -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml --stage ablation
```

The requested single full-run command is:

```bash
python -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml
```

The paper configuration is intentionally expensive. Run smoke and small validation
first. Review validation and recovery before later stages. The complete small
configuration has 50 replications per population/target/geometry with an explicit
short budget; it does not replace the broader paper budget study.

Every config uses its own output directory. Completed per-run JSON files are
resumed without overwriting. Resume refuses changed source/config fingerprints.
Copy/edit the config with a new `output` for a changed experiment. Interruptions
lose only the current run; completed runs remain inspectable. Regenerate artifacts:

```bash
python -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml --artifacts-only
```

## Output layout

Each configured output contains:

- `results/runs.csv`: all run variables, including unavailable/failure rows.
- `results/runs/*.json`: per-run metadata and convergence checkpoints.
- `results/targets/*.npz` and `.json`: target moments, coordinates and provenance.
- `results/designs/*.json`: initial/final supports; only moments enter reconstruction.
- `results/designs/*_pairs.npz`: representative scatter/residual data.
- `results/checkpoints.csv`, `summary.csv`, `paired_comparisons.csv`, `failures.csv`.
- `results/manifest.json`: config, master seed, environment and source fingerprints.
- `tables/table_1`–`table_4` in CSV and LaTeX, and dedicated 1D summaries.
- `figures/figure_1`–`figure_9` in PDF and 300-dpi PNG.
- `reports/full_simulation_report.md`: executed coverage, conclusions and 15 questions.

A figure with no executed data explicitly says so. An existing figure filename is
not evidence that its scientific experiment has been completed.

## Scientific interpretation

Float64 support masses are read directly; the package's float32 normalization is
not used for scientific calculations. The adapter retains mass above 1e-15 rather
than the original 1e-9 push cutoff; all constraints are checked at 1e-10. Search
policies and nine-decimal score comparisons are preserved. This precision limit can
matter and is reported.

Full objectives use each off-diagonal pair once. Sparse objectives store only
selected pairs. `target_RMSE2` uses the targeted-pair denominator; `weighted_RMSE2`
is the zero-weight full-pair normalization. `RMSE2` independently measures all pairs
after optimization, with its diagnostic time recorded separately. Sparse recovery
means recovery of the prescribed subset only.

Package accepted_moves counts nine-decimal incumbent improvements; additional GFS
variants count applied exchanges. CP2 uses enumerated dual calibration, distinct
from first-order conditional-Poisson target generation. It reports mandatory FIP
finalization separately. Enumeration limits CP2 and LP/QP to N<=14. Sinha is the
specified nonnegative pair-removal mixture and reports incompatible targets. DSD is
an ordered projection baseline, not an arbitrary-SIP optimizer.

The QP certificate is an independent linear-oracle bound: reported objective minus
true optimum lies between zero and certificate_gap when primal constraints pass.
An uncertified solver iterate is never described as globally optimal. All paired
comparisons use the same target within replication; intervals describe replication
variability and do not establish universal exact recovery.

See `reports/study_design.md`, `reports/implementation_plan.md` and `CHANGELOG.md`.

Sparse runs report independently reconstructed full-pair RMSE and maximum error, in bounded row workspaces after the timed optimization. Their diagnostic cost is recorded separately. `weighted_RMSE2` retains the zero-weight objective denominator; `target_RMSE2` measures targeted pairs only. Exact QP gaps use a roundoff-corrected LP dual lower bound.

Current staged outputs: [small-N validation report](results/small_validation/reports/full_simulation_report.md), [raw rows](results/small_validation/results/runs.csv), [summaries](results/small_validation/results/summary.csv), and [staged validation record](reports/staged_validation.md). The pilot has two replications; the configured 50-replication study is a separate experiment.

Seeds are exported as exact decimal strings. When reading CSV with pandas, preserve
optional seed columns as strings to avoid conversion through float64:

```python
runs = pandas.read_csv(path, dtype={'seed': 'string', 'target_seed': 'string', 'initial_seed': 'string'})
```

Per-run JSON also preserves seeds as exact integers. Historical pre-review CSV
exports may contain the earlier seed-coercion defect; reproduce from current code.

The exact current package sources, including existing uncommitted research edits,
are preserved as read-only text snapshots under
`reports/source_snapshots/current_original_package/`, with SHA256 hashes and
restoration instructions. This supports reproduction beyond the clean Git commit.

Longer diagnostic results: [recovery report](results/recovery_validation/reports/full_simulation_report.md). It has three replications and distinct budgets; it is not pooled with the pilot.

## Four-process execution

The active paper run now uses **six evaluation workers**, requested after the
four-worker transition. Use `--workers 6` for subsequent restarts. Timing runs
remain serial. The visible paper execution log filters per-method messages and
prints evaluation progress every 100 completed replications; the JSON progress
file continues updating every ten seconds. Every method result remains in the
per-run JSON files. The prior verbose log is preserved as
`results/paper_execution_before_six_workers.log`, and checksums of the 6,062
completed rows at this restart are in `reports/pre_six_worker_results.json`.

Independent evaluation-budget replications can use four spawned processes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/prescribed-sip-matplotlib \
  .venv/bin/python -u -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml --workers 4
```

The existing paper output is resumed rather than replaced. The first transition
from the saved serial executable requires the explicitly approved migration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/prescribed-sip-matplotlib \
  .venv/bin/python -u -m prescribed_sip.run_all --config prescribed_sip/configs/paper.yaml \
  --workers 4 --resume-execution-change
```

Use the ordinary command for subsequent restarts with identical sources. Source
migration verifies frozen prior execution sources and matching mathematical-method
and original-package hashes. The original manifest environment is retained;
execution-only changes are recorded in its execution history. Completed per-run
JSON files are not rewritten. The pre-transition results archive is
`results/paper_before_parallel.zip`; checksums are in `reports/pre_parallel_results.json`.

An exclusive execution lock prevents two controllers writing to the same study.
The additional archive `results/paper_serial_overlap_archive.zip` preserves the
later serial results. Its 913 timing rows are retained separately under
`results/paper/results/overlap_archive`, with an audit in
`results/paper/reports/overlap_audit.json`, for isolated remeasurement.
The source snapshot for that scheduling correction is
`reports/source_snapshots/pre_overlap_cleanup`; migration is already recorded
in the paper manifest. The parallel implementation passed 65 tests.

Within each stage, remaining evaluation jobs run on up to four workers. That pool
fully shuts down before wall-time runs execute serially, with no simultaneous
evaluation work. Stage completion produces updated figures, tables and reports.
The wall-time phase therefore still uses approximately one core; the whole study
will not accelerate fourfold. Evaluation runs retain their safety time caps, so
actual candidate counts/timeouts and configured worker limits are recorded.
Summaries and paired comparisons keep worker-limit strata separate.

The progress file records stage, phase, worker PIDs, completed/pending replication
counts and update time. Evaluation progress refreshes at least every ten seconds;
timed progress updates at method boundaries. The execution log reports every new
method result. Watch progress with system tools:

```bash
watch -n 5 'cat prescribed_sip/results/paper/results/progress.json'
```

The four-worker smoke is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/prescribed-sip-matplotlib \
  .venv/bin/python -m prescribed_sip.run_all --config prescribed_sip/configs/parallel_smoke.yaml --workers 4
```

## Dependence reporting of saved paper results

The existing N12 n4 final results also have a separate dependence reporting export:
[dependence_results.md](results/N12_n4/dependence_reporting/dependence_results.md),
with per-run and mean/sample-SD CSVs, all eight requested final metrics and PDF/PNG
comparison figures. These are recomputed from the saved SIP matrices/designs;
original raw values and files are preserved. This is a reporting update of the
original paper targets, not a rerun under the notebook's newer permuted-target
protocol. Historical unequal-FIP intermediate traces lack pair residuals/design
snapshots, so complete dependence/support/entropy curves cannot be backfilled
exactly; the report documents that limitation, and future notebook runs record
them directly.
