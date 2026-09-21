# N12 n4 paper-ready results

## Experiment

Two fixed N=12,n=4 designs (495 possible samples) were constructed with prescribed equal or moderately unequal FIP. Valid target-independent random GFS exchanges produced frozen feasible target SIP. Target snapshots were chosen from dependence-distance/entropy trajectories. Reconstruction methods received n, pi and target SIP, without target sample probabilities. All 66 pairs were scored with D2 and RMSE=sqrt(D2/66). Original GreedyBestFirstSearch.run and its high-precision comparison sensitivity had a 5,000 scored-design limit and RMSE<1e-10 early stopping; twenty independent saved seeds per target/criterion were prescribed. DSD projection/order fitting also used twenty independent repetitions with a 5,000 scored-kernel limit within its restricted family. Actual saved coverage is reported in the full report.

## Main results table

| method | E_median_RMSE | E_best_RMSE | E_median_Recovery_percent | E_median_runtime_seconds | E_status | U_median_RMSE | U_best_RMSE | U_median_Recovery_percent | U_median_runtime_seconds | U_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Initial | 0.053911109 | 0.053911109 | 0 | 0 | {"initial": 1} | 0.015571882 | 0.015571882 | 0 | 0 | {"initial": 1} |
| Original GFS | 0.053911109 | 0.028591312 | 0 | 3.0746797 | {"search_completed": 1, "stagnated": 19} | 0.0091586971 | 0.0063775424 | 41.184391 | 15.790507 | {"budget_exhausted": 15, "search_completed": 1, "stagnated": 4} |
| Original GFS high precision | 0.053911109 | 0.028591312 | 0 | 3.5438754 | {"search_completed": 1, "stagnated": 19} | 0.009158361 | 0.0063775424 | 41.18655 | 14.949788 | {"budget_exhausted": 15, "search_completed": 1, "stagnated": 4} |
| CP(2) | 2.5463431e-09 | 2.5463431e-09 | 99.999995 | 0.060371082 | {"completed": 1} | 2.6932492e-09 | 2.6932492e-09 | 99.999983 | 0.040711607 | {"completed": 1} |
| Sinha | NA | NA | NA | 7.6696999e-05 | {"NA — theoretical restrictions not satisfied": 1} | NA | NA | NA | 8.1916e-05 | {"NA — theoretical restrictions not satisfied": 1} |
| DSD | 0.10363853 | 0.10363853 | -92.239649 | 1.0964037 | {"completed": 20} | 0.099549157 | 0.099549157 | -539.2879 | 1.6804607 | {"completed": 20} |
| Exact LP/QP | 6.4164603e-16 | 6.4164603e-16 | 100 | 0.00932633 | {"certified_feasible": 1} | 2.8617203e-16 | 2.8617203e-16 | 100 | 0.007654428 | {"certified_feasible": 1} |

Table caption. Final SIP reconstruction accuracy, recovery relative to the initial design and recorded runtime for two frozen feasible targets. Stochastic rows summarize independent optimizer repetitions; deterministic rows represent one fit. NA denotes unavailable/inapplicable results, with explicit statuses.

## Results paragraph

For equal fip, Original GFS had median RMSE 0.053911109 (best 0.028591312; median recovery 0%), and 0/20 saved runs attained RMSE<1e-10. The high-precision sensitivity had median RMSE 0.053911109; the oracle achieved 6.4164603e-16. CP(2) and DSD median RMSE were 2.5463431e-09 and 0.10363853. For unequal fip, Original GFS had median RMSE 0.0091586971 (best 0.0063775424; median recovery 41.184391%), and 0/20 saved runs attained RMSE<1e-10. The high-precision sensitivity had median RMSE 0.009158361; the oracle achieved 2.8617203e-16. CP(2) and DSD median RMSE were 2.6932492e-09 and 0.099549157.

Deterministic CP(2), exact LP/QP and Sinha fits ran serially before the worker pool. Stochastic Original GFS, high-precision GFS and DSD runtimes were measured under contention from up to twelve concurrent workers. These are descriptive execution costs, not controlled speed ratios. Manifest execution_seconds measures the comparison and artifact generation interval after deterministic fitting; it excludes deterministic fits, target preparation and the timing pilot, so it is not full end-to-end experiment time.

## Recommended figures and captions

Figure A. Target construction: entropy, standardized pair-dependence distance, and SIP RMSE from the initial design over accepted random exchanges. Dashed lines mark the frozen target snapshot. The target walk is independent of the later reconstruction objective.

[PDF](../results/N12_n4/figures/figure_A.pdf) · [PNG](../results/N12_n4/figures/figure_A.png)

Figure C. Reconstruction RMSE against scored candidate evaluations, with median and interquartile bands over Original GFS repetitions, the high-precision sensitivity, and restricted DSD projection/order fitting. GFS counts scored sampling designs; DSD counts scored projection/order kernels. Curves show incumbent RMSE medians and interquartile bands. Within each observed run, the last recorded checkpoint is carried forward to the union checkpoint grid; this is LOCF, not linear interpolation. Grids are restricted to the intersection of observed run ranges, so no run is carried forward after its final checkpoint. The number of represented runs is saved in convergence CSV files. This common range may end before the candidate budget; supplementary full-trace figures show every recorded GFS run through its actual endpoint.

[PDF](../results/N12_n4/figures/figure_C.pdf) · [PNG](../results/N12_n4/figures/figure_C.png)

Figure E. Final stochastic RMSE distributions, individual runs, and deterministic reference levels. Exact LP/QP is an enumerated numerical oracle. Values below 1e-16 are floored only for log-scale display; tabulated values retain full precision.

[PDF](../results/N12_n4/figures/figure_E.pdf) · [PNG](../results/N12_n4/figures/figure_E.png)

Figure F. Target versus recovered SIP for all 66 unordered pairs. Representative runs use the lowest saved replication, with seed/path tie breaks, independently of performance. Best runs minimize final RMSE with the same tie breaks. Panels and legends identify selection and replication; figure_F_selections.csv records exact choices. Deterministic methods appear in both selection panels when available.

[PDF](../results/N12_n4/figures/figure_F.pdf) · [PNG](../results/N12_n4/figures/figure_F.png)

Supplementary Figure C full traces. Observed incumbent RMSE traces for every Original GFS and high-precision repetition, with method-specific colors and alpha=0.25. Each line ends at that run's final recorded checkpoint; no interpolation or extrapolation is used. These traces reveal budget-ending improvement or stagnation after the common-range summary has stopped. Recorded plateaus do not establish local optimality. Horizontal axis: candidate evaluations.

[PDF](../results/N12_n4/figures/figure_C_full_traces.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_C_full_traces.png)

Supplementary Figure D full traces. Observed incumbent RMSE traces for every Original GFS and high-precision repetition, with method-specific colors and alpha=0.25. Each line ends at that run's final recorded checkpoint; no interpolation or extrapolation is used. These traces reveal budget-ending improvement or stagnation after the common-range summary has stopped. Recorded plateaus do not establish local optimality. Horizontal axis: elapsed runtime.

[PDF](../results/N12_n4/figures/figure_D_full_traces.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_D_full_traces.png)

Projection DSD satisfies pi_ij<=pi_i*pi_j for every pair. Both frozen targets violate this condition with positive covariances (equal fip: 18 pairs, maximum 0.2026164747193881; unequal fip: 19 pairs, maximum 0.19406985406092198), making exact recovery impossible for any projection DSD, regardless of order fitting.

Figures B (SIP matrices) and D (runtime convergence) are recommended as supplementary figures.

## Limitations paragraph

Two fixed targets measure algorithmic variability for these instances, not population-wide recovery guarantees. The independent equal/unequal targets have different baseline errors and entropy; their outcome difference does not identify a causal effect of FIP inequality. Finite-budget plateaus are not proofs of local optimality. Nine-decimal D2 comparisons can round D2<5e-10 to zero (RMSE approximately 2.75e-6), while the frontier retains raw-loss ordering; removing rounding is a sensitivity of the original algorithm. CP(2) uses enumerated dual L-BFGS calibration plus a FIP tilt, differing from the published iterative/Gibbs implementation. DSD is a restricted projection benchmark and Sinha requires a nonnegative pair-removal mixture. The enumerated LP/QP oracle demonstrates numerical feasibility only to solver/audit precision and does not imply scalable sampling. Final design validity is independently audited from saved support.
