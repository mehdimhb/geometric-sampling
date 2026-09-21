# Prescribed-SIP simulation study: proposed design

Status: repository reconnaissance complete; implementation and experiments have not started.

## Repository findings

- `graphical_sampling/design.py`: `Design._build` constructs graphical fixed-size support; `exchange` and `_switch` implement four-sample mass splits restricted by zones/fixed IDs. `all_samples_and_probs` aggregates samples, converts masses to float32 and normalizes. `fip`, `indicator_matrix`, and `sip` derive inclusion probabilities from that export.
- `graphical_sampling/search/gbfs.py`: GreedyBestFirstSearch includes order/exchange neighbors, probability-aware support keys, a best-design reservoir and shock/backtracking logic. It uses Python and NumPy random sources; copies create unseeded generators.
- `graphical_sampling/criteria/criteria.py`: existing uncommitted SIPMatchingCriteria evaluates off-diagonal RMSE.
- `graphical_sampling/population.py`: bounded inclusion-probability normalization with iterative certainty-unit handling; validate edge cases before reuse.
- `simulations/jupyters/16_inverse_gfs_sip_matching.ipynb`: inverse-SIP example, with a target generated from a graphical design; useful integration example, insufficient as an independent target family.
- `simulations/jupyters/15_simple_gbfs_vs_dsd.ipynb` and ABC notebooks: real/complex projection-DSD constructions and sampling; mathematical provenance and numerical validation remain required.
- Existing tests cover GBFS probability-aware keys and output modes. These are user-owned untracked files.
- No CP(2), Sinha, rejective or cube implementation identified in first-party source/notebook source cells. Inspect installed `package-sampling` before deciding applicability.
- Existing user modifications: criteria init/implementation, GBFS; untracked notebooks 14–16 and tests. Preserve all.
- Notebook searches examine source cells rather than embedded images/output. Binary result artifacts are inventoried as existing artifacts, not treated as algorithm specifications.

## Recommended architecture

Create an isolated `prescribed_sip` package with core, methods, experiments, configs, results, figures, tables, reports and tests. Preserve existing research files. Reuse the existing graphical builder through a seeded adapter and extract raw double-precision heap masses; independently validate those masses before use. Do not use the float32 normalized export for scientific invariant checks. Implement a separate active-support mass-exchange engine because unrestricted SIP optimization, exact epsilon and sparse objectives are absent from the current zone-restricted search.

Alternative 1: extend Design/GBFS directly. This introduces changes into user-owned research code, repeated full-matrix computation, and existing precision/randomness issues; not recommended.
Alternative 2: use only enumerated convex optimization. This supplies a useful small-N benchmark but cannot satisfy the large-N requirement.

## Mathematics and validation

For residual r=current-target and unit exchange direction c, compute A=sum(w*r*c), B=sum(w*c*c), and epsilon=clip(-A/B,0,min(p_a,p_b)). Explicitly handle B=0, identical/reversed support destinations and exchanges whose SIP direction is zero. Aggregate coincident support changes before applying. Validate analytic directions against independent inclusion calculations and analytic epsilon against a numerical grid.

After every accepted move check total mass, FIP, size and nonnegativity. Track deletion mass so zero pruning cannot silently violate invariants. Periodically recompute SIP independently. Stop and save materially failed runs. Validate initial support before optimization; never repair failed designs silently by normalization.

Dense objectives store unique upper-triangle pairs; sparse objectives store only targeted pairs and their residuals. Report full-pair RMSE and targeted-pair RMSE separately; weights of zero must not dilute targeted accuracy. Full diagnostic computation, if requested for sparse runs, occurs outside timed search and is separately accounted for.

Small-N LP enumerates all samples and checks FIP/SIP equality feasibility. Convex QP minimizes weighted residuals with fixed FIP. Record solver status, primal feasibility and a defensible optimality certificate/bound; do not call an uncertified solver iterate the global optimum. Necessary pair bounds and row sums are diagnostics, not feasibility proofs.

## Search and fairness

Random generates moves without target residual information; its primary undirected trajectory accepts valid random/fixed steps, and its optimized-epsilon ablation is explicitly labeled objective-informed in step choice. Greedy screens random candidates and accepts the best exact improving step. Directed chooses residual pairs and verifies the actual SIP direction. Advanced shares a total evaluation/time budget across seeded starts, directed descent and occasional shocks, preserving a best-so-far incumbent. Save working and incumbent trajectories separately where shocks increase loss.

Count every exact candidate-objective evaluation, including screening, shocks and restarts. Record generation attempts and overhead separately. Use common target and starting-design seeds for paired comparisons. Never initialize from hidden target sample probabilities. Initialization depends on FIP only. Use stable seed derivation from master seed plus scenario, replication and method identifiers; avoid Python's randomized hash.

## Literature and targets

Retrieve original or reliable specifications before including named CP(2), Sinha or DSD constructions. Document exact algorithms/equations and small known-example checks. If unavailable, report: “Method not included because a reliable implementation could not be established from the available specification.” Conditional-Poisson target generation must not be labeled CP(2). DSD is a structured benchmark/target family, not a general arbitrary-SIP solver.

Independent targets: arbitrary random enumerated designs, sparse support, diffuse designs, SRS, unequal-FIP designs and perturbed targets for small N; SRS, stratified, independent finite-support designs and verified conditional-Poisson/projection targets where practical for larger N. Bounded PPS FIP generation and target generation must be mathematically compatible; record design-induced FIP where target families cannot realize a separately prescribed PPS vector.

## Staged execution and artifacts

1. Mathematical core and invariant/epsilon/solver tests.
2. Quick full-pipeline smoke run.
3. Complete small-N study, at least 50 paired replications per specified scenario/target; examine recovery and certified gaps before proceeding.
4. Moderate-N study, 30–50 replications per configured scenario.
5. Profile serial implementation and tuple/bit-mask storage; optimize only measured bottlenecks.
6. Large-N study through N=1000, optionally 2000; all requested densities and random/structured targeting. Save explicit failures/timeouts and per-run limits.

Configurable independent evaluation and wall-time regimes, logarithmic checkpoints, focused epsilon/start/candidate-count ablations. Results are append-only with provenance and resumable configuration fingerprints. Store targets, initial/final support, seeds, raw run rows, checkpoint rows and failures. Include every user-requested field and distinguish exact target feasibility, convergence and recovery at each tolerance.

Generate summaries, paired intervals, four CSV/LaTeX tables, nine PDF/300-dpi PNG figures from saved data only. Choose representative examples by a prespecified scenario/replication, not outcome. Record dependency versions, Git state, source fingerprints and hardware. Produce full report with evidence-linked findings and explicit answers to all 15 scientific questions. Unexecuted scenarios remain marked unexecuted; no invented conclusions.

Commands: `python -m prescribed_sip.run_all --config prescribed_sip/configs/smoke_test.yaml` and corresponding `paper.yaml`. Paper command supports validated serial execution, resume and explicit stage selection; full execution completes all stages when no stage restriction is supplied.

## Current environment inventory

Existing virtual environment: NumPy 2.4.4; SciPy 1.17.1; matplotlib 3.10.8; pandas 3.0.2; pytest 9.0.2; package-sampling 0.3.0; PyYAML 6.0.3. cvxpy and osqp unavailable. Solver selection and dependencies require validation during implementation.

## Approved amendments (2026-09-18)

The existing package `GreedyBestFirstSearch.run` is the PRIMARY optimized GFS implementation. New Random/Greedy/Directed/Advanced optimizers are additional comparisons and ablations. Numerical adapters retain float64 masses and deterministic serial RNGs without editing package files. Include dedicated `geometry=1d` populations positioned and initially ordered on a line alongside general planar settings; include 1D-specific summaries and comparisons with every applicable competitor.
