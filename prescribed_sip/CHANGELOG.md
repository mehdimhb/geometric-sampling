# Changelog

## Educational prescribed-SIP notebook

- Made Directed raw SIP early stopping an explicit `DIRECTED_RMSE_STOP` setting defaulting to Original GFS's `1e-10` threshold. Apply strict less-than checks and stop already-recovered Directed initial designs before candidate scores; retain independent repetitions, maximum budgets, final validation and recorded actual stop reasons.

- Reduced repeated target-support bookkeeping: avoid heap-iterator `_Sample` allocations, aggregate immutable ID sets before sorting, and reuse unchanged before-move probabilities. Package exchanges, merging, RNG and requested accepted-move counts are preserved; verified exact support probabilities and bitwise SIP equality against pre-change targets.

- Replaced main NRMSE/Recovery_N reporting with pairwise dependence RMSE_rho/Recovery_rho; raw SIP RMSE and search decisions remain unchanged. Added explicit near-boundary pair exclusion/tolerance/counts, no-valid-pair/zero-baseline NA behavior, actual pair-residual diagnostics for Directed checkpoints, incumbent support/entropy, all eight requested plot panels and incremental final-support exports.

- Target walks now start from standard GFS on a reproducibly permuted population order with original IDs/FIP attached. Frozen target SIP is restored to original-ID order; both optimizers retain the standard original-order initial design. Added printed permutation/FIP/fixed-size confirmations, saved permutation provenance and tests against an independently reordered-records builder, including certainty units.

- Added 30 independent Directed repetitions (editable 100,000-direction caps and batch 50), six-process scheduling with separate seeds/state, per-run status/counts, mean/sample SD, automatic incremental CSV/settings exports and frozen problem matrices. Added off-diagonal target-normalized NRMSE and Recovery_N to method tables, raw/normalized traces and comparison plots. Package search/exchange code and frozen paper results remain preserved.

- Fixed notebook early frontier exhaustion with an explicit Original GFS + restarts protocol: calls the same package optimizer on fresh frontiers, perturbs retained-best seeds through valid package exchanges, enforces a shared score/time budget, counts new restart seeds, retains the global best, guards against repeated empty stages, and reports restart history/stop reasons. Prior notebooks and outputs are backed up; package source and completed paper-study results are preserved.
- Added optional GFS-Directed with an editable candidate budget, the same frozen notebook target/initial design, independent validation, large-run pilot/guard, and convergence/summary comparisons with Original GFS.
- Added a standalone, editable one-run notebook showing the actual package builder, segment exchanges, original GBFS call, scoring, convergence, and optional existing CP(2)/DSD/oracle implementations.
- Added large-population enumeration/runtime safeguards, explicit MU284 row-count/size-column reporting, and notebook execution tests. Existing package source files and frozen study results are preserved.

## Fixed N12 n4 proof-of-concept

- Paused the broad study and added only two frozen equal/unequal-FIP problems with standard graphical initial designs and objective-blind random mass-exchange targets.
- Added exact saved size/FIP vectors, random-walk entropy/dependence trajectories and separation-based target selection.
- Instrumented the original package optimizer using D2 and added a same-bytecode high-precision comparison diagnostic; package files and user edits remain unchanged.
- Added a benchmark and two-hour execution approval gate, 20 paired GFS seeds per case/criterion, 20 projection-order DSD fitting seeds, deterministic CP2/Sinha/oracle checks, independent support/moment audits, complete threshold summaries and six PDF/300-dpi PNG figure families.
- Added separate complete and paper-ready reports, main and supplementary CSV/Markdown/LaTeX tables, source snapshots, seeds and raw per-run supports/traces under `results/N12_n4/`.

## 2026-09-18

- Added the isolated `prescribed_sip` study, retaining the existing package GBFS as the primary optimizer and adding four exchange-search comparisons.
- Added serial float64 precision adapters, seeded random contexts, raw probability exports, strict invariant checks and certainty-unit conditioning. Original research code and existing user edits remain unchanged.
- Implemented small-N exact LP feasibility and convex QP with corrected LP-dual suboptimality bounds; exact-enumerated CP(2) calibration; specified Sinha mixtures; existing projection-factor DSD baseline.
- Added independent target families, dedicated ordered-line populations, sparse objectives, bounded-workspace full-pair diagnostics, candidate/time regimes, epsilon/batch/multistart ablations and profiling.
- Added fingerprinted resumable run files, explicit applicability/failure statuses, uncertainty summaries, paired comparisons, four CSV/LaTeX tables, nine PDF/300-dpi PNG figures, reports and reproduction configurations.
- Added and executed regression tests for exact moments, numerical invariants, package delegation/reproducibility, known competitor examples, dual certificates, sparse diagnostics, seed export and report grouping. Final suite: 55 passed.
- Preserved preliminary runs in explicitly named `*_pre_certainty_fix` and `*_pre_review` directories. These are historical diagnostics, not current scientific evidence.

- Completed the longer 162-row recovery diagnostic (27 cases) and preserved its separate reports/results; all final numerical invariants passed.
- Added read-only exact package-source snapshots, checksums and hardware details for reproduction.

## Parallel execution transition

- Archived 1,301 completed paper rows and all saved target/design artifacts in `results/paper_before_parallel.zip`, froze the previous executable sources/manifest, and recorded per-run checksums.
- Added four-process independent evaluation scheduling, complete pool shutdown before isolated timing, sequential study-stage barriers, and immutable resume.
- Added explicit audited execution-source migration with original provenance retained, concurrency metadata/stratification, per-method log output and a visible progress file.
- Verified numerical agreement with the frozen serial runner, process isolation, parent-only timing, source-change rejection and unchanged saved JSON on resume. Full test suite: 63 passed.

- Corrected the reviewed exact-only legacy resume edge and verified it with an integration regression. The final four-worker smoke saved 106 rows, used four distinct evaluation-worker PIDs and one timing PID, and produced all figure/table types.

- Added an exclusive execution lock and orderly termination handling. Legacy evaluation completion now defers timing work until after the evaluation pool shuts down. Full suite: 65 passed.
- Preserved 913 later timing rows separately in `results/paper/results/overlap_archive` and the complete `results/paper_serial_overlap_archive.zip`; these timing rows will be measured again in isolation. All 1,301 original saved rows passed their checksum checks.
- Verified the resumed paper study has one live controller and four distinct evaluation processes. The main GFS method still invokes the existing package optimizer; original tracked user edits remain unchanged.
- Increased the active evaluation pool to six processes on request, using the existing worker option without changing numerical code. Filtered the visible log to periodic progress summaries, archived the verbose log, and verified all 6,062 completed rows remain unchanged.
