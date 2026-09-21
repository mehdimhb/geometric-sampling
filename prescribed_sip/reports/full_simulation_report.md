# Prescribed second-order inclusion probability simulation report

This is the completed **small-N validation pilot**, not the full replicated paper study. Raw and summarized data, tables and figures are under `../results/small_validation/`. The full 50-replication and subsequent moderate/large configurations are supplied but have not been executed. See [staged validation](staged_validation.md) and [README](../README.md).

**Execution status:** This report describes saved executed runs only. Configured experiments without raw rows are unexecuted.

Raw rows: 1716. Configured stages: ['ablation', 'small']. Executed stages: ['ablation', 'small'].
No claim that a stage is complete is made solely from its presence in this list; coverage is tabulated below.

## Computational environment

Python: `3.12.13 (main, May 10 2026, 19:30:01) [Clang 22.1.3 ]`. Platform: `Linux-7.0.0-31-generic-x86_64-with-glibc2.43`. Logical CPUs: 16.

Dependency versions: numpy=2.4.4, scipy=1.17.1, matplotlib=3.10.8, pandas=3.0.2, pytest=9.0.2, osqp=1.1.3, PyYAML=6.0.3, graphical-sampling=0.1.0, package-sampling=0.3.0.
Git commit: `dc2881af84a41e4f7658b6d21401e58470871b55`. Full Git state, dependency versions and SHA256 source fingerprints are in `results/manifest.json`.

## Implementation and mathematics

The primary method is the existing `graphical_sampling.search.GreedyBestFirstSearch.run`, invoked serially without replacing its search strategy. Its order neighbors, sample exchanges, frontier, duplicate detection, reservoir, shocks and nine-decimal score comparisons are retained. Adapter-only differences are deterministic random streams, raw float64 probabilities without normalization, and a 1e-15 push cutoff instead of 1e-9. Exact certainty units are conditioned out before invoking the original graphical builder and appended to every sample, fixing a reproduced fractional-wrap event defect in the adapter. One global zone permits unrestricted exchanges; `backtrack_depth=0` avoids invalid random bounds for small frontiers. Scores are target-pair RMSE. Batch and frontier settings are saved per run.

Additional GFS variants are comparisons and ablations. Random accepts undirected valid moves; Greedy screens target-independent candidates; Directed guides pair selection by signed residuals and computes the exact complete change; Advanced shares its budget among directed starts and shocks. Every accepted exchange checks size, positivity, total probability and FIP at 1e-10. SIP is independently recomputed every 25 accepted exchanges and on finalization. Primary GBFS has no incrementally cached SIP in this adapter: every candidate is computed from raw support, so its incremental-error field is zero by construction, not a separate speed claim.

For residual $r_{ij}=\pi_{ij}-\pi^*_{ij}$ and direction $c$, $A=\sum wrc$, $B=\sum wc^2$, $\epsilon=\mathrm{clip}(-A/B,0,\min(p_a,p_b))$ when $B>0$. Pair changes are checked against independently reconstructed indicator moments. Sparse objectives store only targeted pairs. `RMSE2` is independently evaluated over every pair, using bounded row workspaces after sparse optimization; `weighted_RMSE2` uses the full-pair denominator with zero weights on omitted pairs; `target_RMSE2` divides by the number of targeted pairs and drives sparse comparisons.

LP enumerates all fixed-size samples for N<=14. Feasibility uses complete first-/second-order equalities. Infeasible targets use an OSQP convex QP with explicit residual variables; an independent LP linear-minimization oracle provides a Frank–Wolfe upper bound on suboptimality. QP rows are called certified only when primal error <=1e-10 and gap <=1e-7. This gap is an absolute D2 bound; do not infer finer precision from a small objective alone.

## Literature-based competitors and applicability

- **CP2-enumerated:** Bondesson CP(2) design, exact exponential-family partition function for small N. The accessible author specification is Section 3(a) of the BANOCOSS 2011 presentation attached to [Bondesson’s paper](https://www.researchgate.net/publication/260138813_On_Sampling_with_Prescribed_Second-order_Inclusion_Probabilities); version of record: [Bondesson (2012), SJS 39,813–829](https://doi.org/10.1111/j.1467-9469.2012.00808.x). The numerical implementation uses our log-partition dual L-BFGS calibration followed by a Newton first-order tilt. It is not claimed to reproduce the paper’s inaccessible iterative/Gibbs implementation. Exact partition functions are limited to N<=14. Mandatory final FIP calibration and its runtime/evaluations are recorded separately; intermediate CP2 calibration iterates do not preserve FIP. Thus equal GFS candidate budgets are not a claim of equal CP2 internal arithmetic.
- **Sinha:** nonnegative pair-removal mixture, [Bondesson, Grafström & Traat (2014)](https://umu.diva-portal.org/smash/get/diva2:787764/FULLTEXT01), Section4 type(b), pp183–185 and formula before Lemma1. Direct inversion uses $b_{ij}=(N-2)(N-3)\pi_{ij}/[n(n-1)]-(N-2)(\pi_i+\pi_j)/n+1$. Negative coefficients give an explicit not-applicable status. The original 1973 paper was not available; this reliable later specification is used. Sparse prescriptions do not identify the full coefficient vector.
- **DSD-structured:** factor construction reused from notebook15, independently verified against [Loonis (2023), Algorithm3.1 and Eq(3.2)](https://www150.statcan.gc.ca/n1/pub/12-001-x/2023002/article/00008-eng.pdf). Projection SIP is $\pi_i\pi_j-|K_{ij}|^2$. Fixed original population ordering is used; this is a structured baseline, not an arbitrary-SIP solver. For N>14 the distribution is implicit; entropy and enumerated support are unavailable.

A broader large-N implementation of Bondesson’s published calibration/sampling routine is **not included because a reliable implementation could not be established from the available specification**. Conditional-Poisson target generation is distinct from CP(2). No improvised method is presented under a published name.

## Validation and experiment design

Tests cover independent exchange moments, exact invariants, valid support sizes, coincident destinations, optimized epsilon versus a grid, sparse denominators/storage, seeded package repeatability, line coordinates, known CP(2)/Sinha/projection examples, LP feasibility and QP gap bounds. `reports/validation.txt` stores the executed test output when supplied by the staged workflow. Tests do not guarantee global reachability of every feasible SIP under nonnegative two-sample exchanges.

Targets are generated independently of the optimizers; hidden probabilities are discarded at the API boundary. Independent random designs, sparse designs, diffuse designs, SRS, stratified sampling, exact log-DP conditional-Poisson moments and structured projection targets are used as configured. Random sparse designs receive coverage augmentation rather than rejecting unfavorable seeds. Bounded PPS is used for projection diagonals; other unequal families use their exact design-induced FIP. Target/design randomness is separated from coordinates and pair masks. Same underlying target is used across densities/geometries for a replication. Four-cycle perturbations preserve SIP row sums; actual feasibility is established by LP.

**Dedicated 1D experiment:** units lie at equally spaced coordinates on a line (second coordinate zero); initial GFS order follows the line. All applicable competitors run on those targets. See `tables/one_dimensional_summary.csv` and `.tex`. Primary single-zone exchange/ordering search does not use distances in the criterion; geometry effects should not be attributed to spatial balance.

Master seed: 20260918. Independent seeds are SHA256-derived from stable scenario/replication/method identifiers and saved per run. Raw runs include failures. Summaries report means, medians, SD, IQR, extrema, bootstrap intervals and Wilson recovery intervals. Paired differences are competitor minus primary target-pair RMSE; positive values favor primary. No significance-test grid is used.

Evaluation-budget and time-budget regimes are separately labeled. Candidate budgets count exact scored exchanges for new variants and unique scored designs for package GBFS; generation/deduplication overhead is included in runtime. Primary accepted_moves counts best-evaluated incumbent improvements, whereas new variants count applied exchanges; these counts are not interchangeable. Wall-time runs can vary with machine load and cannot be bitwise reproduced. Final validation/mandatory calibration overhead can exceed the nominal timeout and is not hidden. Process RSS is a process-lifetime high-water measurement, not an isolated method peak.

## Executed coverage

| Experiment | N | Geometry | Targets | Raw rows |
|---|---:|---|---:|---:|
| ablation | 8 | 1d | 3 | 48 |
| ablation | 12 | 1d | 3 | 48 |
| small | 8 | 1d | 9 | 162 |
| small | 8 | planar | 9 | 162 |
| small | 10 | 1d | 9 | 324 |
| small | 10 | planar | 9 | 324 |
| small | 12 | 1d | 9 | 162 |
| small | 12 | planar | 9 | 162 |
| small | 14 | 1d | 9 | 162 |
| small | 14 | planar | 9 | 162 |

## Results and evidence

Tables1–4 contain stage-specific summaries; `results/summary.csv` contains the full stratification and uncertainty intervals. Figure2 pools distributions for display only; it does not replace stratified comparisons. Figures3–4 pool median outcomes by N and should be interpreted with changing configured scenarios in mind.

Representative scenario: `small_N10_n3_arbitrary_1d_d1_random`, replication0, selected by prespecified target class/lexicographic scenario order rather than outcome. Figures1,5,6,7 and9 use this example.

| Method | Valid rows | Median target RMSE | Median runtime (s) |
|---|---:|---:|---:|
| CP2-enumerated | 160 | 1.19375e-09 | 0.009764 |
| DSD-structured | 180 | 0.0428903 | 0.00176 |
| Exact-LP-QP | 192 | 1.58353e-16 | 0.004122 |
| GFS-Advanced | 192 | 0.0452294 | 0.01167 |
| GFS-Directed | 216 | 0.0251947 | 0.01105 |
| GFS-Greedy | 216 | 0.0319134 | 0.01145 |
| GFS-Random | 180 | 0.0771649 | 0.06257 |
| Package-GFS | 180 | 0.0254851 | 0.08808 |
| Sinha | 20 | 2.63678e-16 | 0.001123 |

## Figures

- [Figure1: Small-target convergence](../results/small_validation/figures/figure_1.pdf), [300-dpi PNG](../results/small_validation/figures/figure_1.png).
- [Figure2: Small/moderate accuracy distributions](../results/small_validation/figures/figure_2.pdf), [300-dpi PNG](../results/small_validation/figures/figure_2.png).
- [Figure3: Runtime versus population size](../results/small_validation/figures/figure_3.pdf), [300-dpi PNG](../results/small_validation/figures/figure_3.png).
- [Figure4: Accuracy versus population size](../results/small_validation/figures/figure_4.pdf), [300-dpi PNG](../results/small_validation/figures/figure_4.png).
- [Figure5: Accuracy against computing time](../results/small_validation/figures/figure_5.pdf), [300-dpi PNG](../results/small_validation/figures/figure_5.png).
- [Figure6: Target versus reconstructed SIP](../results/small_validation/figures/figure_6.pdf), [300-dpi PNG](../results/small_validation/figures/figure_6.png).
- [Figure7: Initial/final residual matrices](../results/small_validation/figures/figure_7.pdf), [300-dpi PNG](../results/small_validation/figures/figure_7.png).
- [Figure8: Sparse-target scalability](../results/small_validation/figures/figure_8.pdf), [300-dpi PNG](../results/small_validation/figures/figure_8.png).
- [Figure9: Active support and support growth](../results/small_validation/figures/figure_9.pdf), [300-dpi PNG](../results/small_validation/figures/figure_9.png).

## Failures, timeouts and unexpected findings

| Method | Status | Runs |
|---|---|---:|
| CP2-enumerated | budget_exhausted | 32 |
| CP2-enumerated | completed | 128 |
| CP2-enumerated | not_applicable | 20 |
| DSD-structured | completed | 180 |
| Exact-LP-QP | certified_feasible | 172 |
| Exact-LP-QP | certified_qp | 20 |
| GFS-Advanced | completed | 192 |
| GFS-Directed | completed | 216 |
| GFS-Greedy | completed | 216 |
| GFS-Random | completed | 180 |
| Package-GFS | budget_exhausted | 180 |
| Sinha | completed | 20 |
| Sinha | not_applicable | 160 |

Detailed failure reasons are saved in `results/failures.csv`; numerical failures are never silently normalized away. Non-recovery of a known-feasible target is an optimizer result, not evidence of infeasibility. Candidate exhaustion alone cannot distinguish a local minimum from slow convergence. The package’s nine-decimal score comparisons impose an implementation precision limit; this is retained and disclosed rather than silently changed.

## Scientific questions

1. Numerical recovery: 0/160 primary known-feasible runs attained target-pair RMSE<1e-8; 0 attained <1e-10. See Table1 and Figure1; a finite experiment cannot prove universal recovery.
2. Recovery frequency: 0/160 primary known-feasible runs attained target-pair RMSE<1e-8; 0 attained <1e-10. These are finite-budget recovery rates; unrecovered runs are not certified local minima.
3. Global optimum proximity: median primary excess D2=0.0324626, maximum=0.353207 across 180 certified benchmark comparisons. See Table1.
4. Directed versus random: see paired stratified outcomes and Figure1. No material-improvement claim is made without sufficient paired replications.
5. Optimized epsilon: see Table4. If the ablation stage has no rows, its effect is unmeasured.
6. Multi-start/perturbation: see Table4 and saved start/shock counts. Non-recovery alone does not establish local minima.
7. Competitors: CP2 and Sinha are included under the documented limits above; DSD is structured; Exact-LP-QP is the small-N gold standard. Status counts report missing/failed cases.
8. N and fraction: Tables2–3 and Figures3–4 provide executed comparisons. Unexecuted N/fraction settings support no conclusion.
9. Unequal FIP: stratify Table1/2 and summary.csv by FIP_type/target_type; do not confound family differences with inequality alone.
10. Entropy: hidden target entropy is saved for random-design families. Table1/2 target classes separate sparse and diffuse cases; effects need replicated comparisons.
11. Support growth: primary median=5.2, maximum=16.67. See Figure9.
12. Largest executed primary population with a valid final design: N=14; this is an executed limit, not an extrapolated practical ceiling. See Table3.
13. Sparse scalability: see Table3 and Figure8; no claim is made when sparse experiments are unexecuted.
14. Weaknesses: inspect recovery failures, excess D2, support growth and statuses by target type/fraction in Tables1–4. Distinguish algorithmic non-recovery from numerical/implementation failures.
15. Justified claims: limited to the executed budgets, target families, certified benchmarks and replications reported here. No universal exact recovery or competitor superiority claim is justified by smoke results.


## Longer recovery validation

This separate diagnostic study used 27 independent target/replication cases: N=8,n=3; N=10,n=4; N=12,n=4; arbitrary, sparse-support and SRS targets; three replications; ordered line populations. Budgets were at most 10,000 candidates and 15 seconds. These rows must not be pooled with the 200-candidate pilot. All exact LP targets were feasible. No numerical or implementation failures occurred. Thirteen primary runs reached the time cap; actual candidate counts and their valid incumbent errors remain recorded. This is an evaluation ceiling with a safety timeout, not a claim of identical achieved evaluation counts.

| Method | Cases | RMSE < 1e-8 | Median target RMSE | Median runtime (s) |
|---|---:|---:|---:|---:|
| CP2-enumerated | 27 | 26 | 5.99461e-12 | 0.009036 |
| Exact-LP-QP | 27 | 27 | 9.45773e-17 | 0.003769 |
| GFS-Advanced | 27 | 2 | 0.000345069 | 0.4521 |
| GFS-Directed | 27 | 7 | 3.61732e-07 | 0.4714 |
| GFS-Greedy | 27 | 7 | 2.41079e-06 | 0.4667 |
| Package-GFS | 27 | 0 | 0.000833537 | 14.53 |

Additional Greedy and Directed searches numerically recovered some feasible targets, whereas the primary package optimizer recovered none at this budget. This does not establish a local minimum, global convergence, a reachability barrier or universal method superiority. Advanced did not show a recovery advantage in this diagnostic grid. The distinct implementation/evaluation costs and timeout rates require separate budget regimes in a replicated comparison.

[Full recovery report](../results/recovery_validation/reports/full_simulation_report.md), [raw rows](../results/recovery_validation/results/runs.csv), [stratified summaries](../results/recovery_validation/results/summary.csv), [paired comparisons](../results/recovery_validation/results/paired_comparisons.csv).

Maximum invariant errors: FIP 1.40e-12; probability sum 5.66e-14; independently reconstructed incremental SIP 4.45e-16. Every reported support passed the prescribed 1e-10 tolerance.

## Limitations

Two-sample nonnegative exchange reachability and absence of nonglobal minima are not proved. A convex loss in sample probabilities does not make the restricted exchange search globally convergent. CP2 exact enumeration cannot scale, and its boundary calibration may converge slowly. Sinha mixtures are restricted. The DSD baseline is not optimized over kernels. Implicit large-N benchmarks have unavailable entropy/support statistics. Configurations and tested stages can be much broader than executed data; consult coverage rather than configuration files for evidence. Pilot/smoke results cannot support journal-level comparative claims.

## MAIN FINDINGS FOR THE PAPER

- The primary package optimizer was invoked for 180 saved runs; the original search is preserved (manifest and raw method details).
- Dedicated line populations contributed 906 raw rows across the executed study (1D summary table).
- 0/160 primary known-feasible runs attained target-pair RMSE<1e-8; 0 attained <1e-10. (Table1.)
- Certified small-N gold-standard rows: 192; uncertified results remain explicitly labeled (Table1/status table).
- Explicit timeouts, unavailable cases and other nonstandard statuses: 180 rows; their reasons remain in the raw data (failures.csv).
- Algorithmic superiority, large-N practicality, epsilon benefits and sparse-target gains require the corresponding replicated experiments; configuration alone supplies no evidence (coverage, Tables2–4).
