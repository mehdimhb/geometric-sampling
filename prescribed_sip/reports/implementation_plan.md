# Prescribed-SIP study implementation plan

Goal: implement and execute a reproducible study with package GBFS as the primary GFS optimizer and explicit line populations.
Architecture: serial float64 adapters invoke original Design and GreedyBestFirstSearch; additional mass-exchange variants share an independently checked objective. Enumerated convex benchmarks and specification-based competitors expose applicability and numerical limits.
Tech stack: existing Python 3.12 environment, NumPy, SciPy, matplotlib, pandas, PyYAML, pytest.
Spec: reports/study_design.md plus approved requirements: primary package search and dedicated 1D populations.

Global constraints: preserve original files and existing uncommitted changes; float64 scientific computations; n fixed; invariant tolerance 1e-10; master seed 20260918; no parallel execution until serial validated; stop small-N stage for review before moderate/large if recovery/solver validation reveals a methodological bug.

- [x] Core: tests for exact exchange/SIP direction, probability/FIP/size, destination coincidence, epsilon grid, sparse/full agreement. Run failing tests, implement support/objective modules, rerun.
- [x] Package adapter: failing tests for seeded reproducibility, float64 raw export, inherited GBFS.run delegation, line population, numerical drift rejection. Implement serial random context and precision-only Design subclass. Preserve search rounding, pruning and neighbor policies.
- [x] Competitors: test enumerated LP feasibility, QP certificate and infeasible optimum, CP(2) known exponential example, Sinha-quadratic known example and rejection of negative probabilities; projection DSD moment identities. Implement only specified design families; label calibration implementation precisely.
- [x] Search ablations: test target-independent random generation, local direction, best-incumbent retention, total multistart budget. Implement Random, Greedy, Directed and Advanced, random/fixed/optimal epsilon.
- [x] Targets/configs: independent random/sparse/diffuse/SRS/CP/stratified/projection families, row-balanced infeasible perturbations; line and planar coordinates; deterministic seeds and paired starts. Test positivity and independent moments.
- [x] Runner and artifacts: test smoke pipeline produces raw/checkpoints/support/target files, four CSV/LaTeX summaries, nine PDF/PNG figures, report, explicit statuses and resume fingerprint. Implement CLI and stage selection, candidate/time budgets, uncertainty intervals and paired summaries.
- [ ] Validate tests and smoke, inspect exact small-N results, run complete small-N configured replications. Profile serial pipeline. Only then run moderate/large stages if scientific checks pass; record negative recovery results honestly.
- [ ] Review raw/summaries and figures, verify all original-change fingerprints, document stage coverage and exact reproduction commands. Never claim a configured but unexecuted experiment is complete.

Completed: 55 tests, full-pipeline smoke, corrected 1,716-row small-N pilot, 192 certified exact benchmarks, serial profiling, report-grouping and dual-certificate review corrections. Larger replicated stages remain unexecuted and are not marked complete.
