# Staged validation record

The implementation retains the existing package GBFS as the primary GFS method. Additional searches are comparisons. Original user modifications were compared byte for byte against the saved patch and remain unchanged.

The final suite passed 55 tests. The end-to-end smoke produced 59 rows and all nine PDF/300-dpi PNG figures and CSV/LaTeX tables without numerical failures. It includes ordered line populations, sparse targeting, and tiny moderate/large smoke cases; these cases do not constitute replicated moderate/large experiments.

The corrected small-N pilot produced 1,716 rows at N=8,10,12,14, both line and planar coordinates, nine target families and focused ablations. It uses two replications and at most 200 candidates per optimizer. All 192 exact benchmarks were certified (172 feasible LPs and 20 infeasible-target QPs). Maximum certificate gap was 7.17e-14; maximum solver primal error was 1.70e-14. Across methods, maximum FIP error was 1.43e-12, probability-sum error 1.45e-14 and independently checked incremental SIP error 1.30e-15. There were no numerical, memory or solver failures. Explicit unavailable cases are retained.

At this short budget none of the primary or additional GFS methods recovered the 160 known-feasible small-stage cases below target RMSE 1e-8. CP(2) recovered 128/160 and exact optimization 160/160. This is a finite-budget result, not a reachability or infeasibility finding. Longer recovery validation is evaluated separately and must not be pooled with this pilot.

Raw data and the generated pilot report are in `../results/small_validation/`. Earlier versions are explicitly archived as pre-certainty-fix and pre-review diagnostics. Configured paper stages remain distinct from executed validation stages. The large replicated study must not be claimed complete on the basis of these pilot rows.

Profiling output is in `profile.json`. In its illustrative N=250,n=5,100-candidate case, primary GBFS required about 0.127s and Directed about 0.024s. These are isolated profiling measurements. Integer-mask set differences were faster and smaller than tuples in the microbenchmark; no representation replacement was made to the primary optimizer.


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
