# Fixed N12 n4 prescribed-SIP report

Exactly two fixed reconstruction problems are represented: N=12, n=4, 495 enumerated samples and all 66 unordered pair SIP. Twenty independently seeded optimization repetitions per case for each Original GFS criterion and DSD order fitting are prescribed. Saved-run coverage below distinguishes actual rows from expected repetitions. No N>12 results are included.

D2=sum over i<j of squared SIP residuals; RMSE=sqrt(D2/66); D_inf=max absolute pair residual. Recovery %=100[1-RMSE/RMSE(Pi0,Pistar)]. Negative recovery is retained. All threshold comparisons are strict (<).

## Frozen targets

| case | H0 | Hstar | support0 | supportstar | selected_exchanges | initial_RMSE | d_rho | max_FIP_error | max_row_identity_error | pair_sum_identity_error | abs_SIP_difference_min | abs_SIP_difference_median | abs_SIP_difference_max | abs_rho_difference_min | abs_rho_difference_median | abs_rho_difference_max | metadata_initial_RMSE | metadata_d_rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | 1.0986123 | 2.4692376 | 3 | 495 | 4950 | 0.053911109 | 0.24259999 | 6.1062266e-16 | 1.110223e-15 | 5.3290705e-15 | 0.00063540653 | 0.020199362 | 0.13430015 | 0.0028593294 | 0.090897129 | 0.60435065 | 0.053911109 | 0.24259999 |
| U | 1.9763676 | 2.5988645 | 12 | 495 | 4950 | 0.015571882 | 0.070773616 | 2.7755576e-16 | 5.5511151e-16 | 8.8817842e-16 | 5.3656123e-05 | 0.0086709051 | 0.042258923 | 0.00023230317 | 0.041319689 | 0.19401078 | 0.015571882 | 0.070773616 |

### Equal FIP

Exact saved float64 x vector:
```json
[
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0
]
```
Exact saved float64 pi vector:
```json
[
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333,
  0.3333333333333333
]
```
Target metadata (including selection and initial/target validation):
```json
{
  "H0": 1.0986122886681096,
  "Hstar": 2.4692375999245533,
  "support0": 3,
  "supportstar": 495,
  "selected_exchanges": 4950,
  "walk_exchanges": 5000,
  "walk_attempts": 5000,
  "initial_RMSE": 0.053911109280500136,
  "d_rho": 0.24259999176225058,
  "maximum_observed_d_rho": 0.24534671246001014,
  "entropy_increase": 1.3706253112564437,
  "abs_SIP_difference": {
    "min": 0.0006354065283718367,
    "median": 0.0201993620955654,
    "max": 0.1343001451566045
  },
  "abs_rho_difference": {
    "min": 0.002859329377673265,
    "median": 0.0908971294300443,
    "max": 0.6043506532047201
  },
  "validation": {
    "max_probability_sum_error": 9.992007221626409e-16,
    "max_FIP_error": 6.106226635438361e-16,
    "min_sample_probability": 5.906464243849398e-10,
    "max_SIP_recompute_error": 0.0,
    "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
    "total_pair_identity_error": 5.329070518200751e-15,
    "max_SIP_diagonal_FIP_error": 6.106226635438361e-16,
    "positive_samples_correct_size": true
  },
  "initial_validation": {
    "max_probability_sum_error": 8.881784197001252e-16,
    "max_FIP_error": 2.7755575615628914e-16,
    "min_sample_probability": 0.33333333333333304,
    "max_SIP_recompute_error": 0.0,
    "max_fixed_size_row_identity_error": 8.881784197001252e-16,
    "total_pair_identity_error": 5.329070518200751e-15,
    "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16,
    "positive_samples_correct_size": true
  },
  "target_selection": "maximum entropy among snapshots with dependence distance >=80% of observed maximum and entropy increase >=max(0.5,0.1*H0)",
  "target_generator": "objective-blind uniformly chosen valid support-pair/unit exchanges; epsilon uniform in [0.05,0.95] times available mass",
  "case": "E",
  "N": 12,
  "n": 4,
  "pi": [
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333
  ],
  "x": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "target_seed": 7197644690166350729,
  "initial_seed": 20260918,
  "coordinates": "equally spaced ordered line, units 0..11"
}
```

The selected snapshot reaches 98.880474% of the largest observed dependence distance. Entropy increase Hstar-H0=1.3706253. Snapshot selection uses target-generation dependence/entropy diagnostics, without fitting reconstruction methods. Accepted target-walk counts include valid null exchange transactions when the two samples differ by exactly one unit; such a transaction preserves the design without changing its moments.

### Unequal FIP

Exact saved float64 x vector:
```json
[
  0.8659957791684043,
  1.0572939535852075,
  0.9903401870014739,
  1.2072415355788895,
  0.9502770153419586,
  1.1407824505144104,
  1.1915437840775212,
  1.2205334074612852,
  0.7729940318879492,
  1.030478701115088,
  1.004608171952895,
  1.0502144381541125
]
```
Exact saved float64 pi vector:
```json
[
  0.2775115289360453,
  0.3388137317205207,
  0.31735815124353345,
  0.3868649852489027,
  0.3045197607008732,
  0.36556792728212506,
  0.38183458310977675,
  0.3911244144253911,
  0.24770877734953456,
  0.3302206855524033,
  0.3219303794390422,
  0.33654507499185154
]
```
Target metadata (including selection and initial/target validation):
```json
{
  "H0": 1.9763675783877785,
  "Hstar": 2.598864522127317,
  "support0": 12,
  "supportstar": 495,
  "selected_exchanges": 4950,
  "walk_exchanges": 5000,
  "walk_attempts": 5000,
  "initial_RMSE": 0.015571881835906796,
  "d_rho": 0.07077361590820672,
  "maximum_observed_d_rho": 0.07111249044316345,
  "entropy_increase": 0.6224969437395382,
  "abs_SIP_difference": {
    "min": 5.3656123199741224e-05,
    "median": 0.008670905098902962,
    "max": 0.04225892274172349
  },
  "abs_rho_difference": {
    "min": 0.00023230316826941744,
    "median": 0.04131968936562977,
    "max": 0.1940107837408974
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 2.7755575615628914e-16,
    "min_sample_probability": 2.3224288087937894e-10,
    "max_SIP_recompute_error": 0.0,
    "max_fixed_size_row_identity_error": 5.551115123125783e-16,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16,
    "positive_samples_correct_size": true
  },
  "initial_validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 2.220446049250313e-16,
    "min_sample_probability": 0.00874289719330934,
    "max_SIP_recompute_error": 0.0,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 2.220446049250313e-16,
    "positive_samples_correct_size": true
  },
  "target_selection": "maximum entropy among snapshots with dependence distance >=80% of observed maximum and entropy increase >=max(0.5,0.1*H0)",
  "target_generator": "objective-blind uniformly chosen valid support-pair/unit exchanges; epsilon uniform in [0.05,0.95] times available mass",
  "case": "U",
  "N": 12,
  "n": 4,
  "pi": [
    0.2775115289360453,
    0.3388137317205207,
    0.31735815124353345,
    0.3868649852489027,
    0.3045197607008732,
    0.36556792728212506,
    0.38183458310977675,
    0.3911244144253911,
    0.24770877734953456,
    0.3302206855524033,
    0.3219303794390422,
    0.33654507499185154
  ],
  "x": [
    0.8659957791684043,
    1.0572939535852075,
    0.9903401870014739,
    1.2072415355788895,
    0.9502770153419586,
    1.1407824505144104,
    1.1915437840775212,
    1.2205334074612852,
    0.7729940318879492,
    1.030478701115088,
    1.004608171952895,
    1.0502144381541125
  ],
  "target_seed": 5741324855659215378,
  "initial_seed": 20260918,
  "coordinates": "equally spaced ordered line, units 0..11"
}
```

The selected snapshot reaches 99.523467% of the largest observed dependence distance. Entropy increase Hstar-H0=0.62249694. Snapshot selection uses target-generation dependence/entropy diagnostics, without fitting reconstruction methods. Accepted target-walk counts include valid null exchange transactions when the two samples differ by exactly one unit; such a transaction preserves the design without changing its moments.

## Main results

| method | E_median_RMSE | E_best_RMSE | E_median_Recovery_percent | E_median_runtime_seconds | E_status | U_median_RMSE | U_best_RMSE | U_median_Recovery_percent | U_median_runtime_seconds | U_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Initial | 0.053911109 | 0.053911109 | 0 | 0 | {"initial": 1} | 0.015571882 | 0.015571882 | 0 | 0 | {"initial": 1} |
| Original GFS | 0.053911109 | 0.028591312 | 0 | 3.0746797 | {"search_completed": 1, "stagnated": 19} | 0.0091586971 | 0.0063775424 | 41.184391 | 15.790507 | {"budget_exhausted": 15, "search_completed": 1, "stagnated": 4} |
| Original GFS high precision | 0.053911109 | 0.028591312 | 0 | 3.5438754 | {"search_completed": 1, "stagnated": 19} | 0.009158361 | 0.0063775424 | 41.18655 | 14.949788 | {"budget_exhausted": 15, "search_completed": 1, "stagnated": 4} |
| CP(2) | 2.5463431e-09 | 2.5463431e-09 | 99.999995 | 0.060371082 | {"completed": 1} | 2.6932492e-09 | 2.6932492e-09 | 99.999983 | 0.040711607 | {"completed": 1} |
| Sinha | NA | NA | NA | 7.6696999e-05 | {"NA — theoretical restrictions not satisfied": 1} | NA | NA | NA | 8.1916e-05 | {"NA — theoretical restrictions not satisfied": 1} |
| DSD | 0.10363853 | 0.10363853 | -92.239649 | 1.0964037 | {"completed": 20} | 0.099549157 | 0.099549157 | -539.2879 | 1.6804607 | {"completed": 20} |
| Exact LP/QP | 6.4164603e-16 | 6.4164603e-16 | 100 | 0.00932633 | {"certified_feasible": 1} | 2.8617203e-16 | 2.8617203e-16 | 100 | 0.007654428 | {"certified_feasible": 1} |

Table caption: median and best final RMSE, median recovery and median recorded runtime for the same frozen equal/unequal target. NA cells retain absent or inapplicable methods; status counts are explicit. Runtime/work summaries include failed attempts; accuracy summaries use finite, nonfailed returned results.

For equal fip, Original GFS had median RMSE 0.053911109 (best 0.028591312; median recovery 0%), and 0/20 saved runs attained RMSE<1e-10. The high-precision sensitivity had median RMSE 0.053911109; the oracle achieved 6.4164603e-16. CP(2) and DSD median RMSE were 2.5463431e-09 and 0.10363853. For unequal fip, Original GFS had median RMSE 0.0091586971 (best 0.0063775424; median recovery 41.184391%), and 0/20 saved runs attained RMSE<1e-10. The high-precision sensitivity had median RMSE 0.009158361; the oracle achieved 2.8617203e-16. CP(2) and DSD median RMSE were 2.6932492e-09 and 0.099549157.

## Supplementary results and threshold attainment

The complete metric means, medians, sample SD (undefined for one result), IQR, quartiles, minima and maxima are saved in tables/supplementary_results.csv, .md and .tex. D_inf and runtime/evaluation distributions are included. Threshold counts/proportions use all saved attempts, including failures, as denominator. Expected repetitions and actual attempts are both reported. Missing methods have no attainment proportion.

| case | method | expected_runs | runs | valid_results | status_counts |
| --- | --- | --- | --- | --- | --- |
| E | Initial | 1 | 1 | 1 | {"initial": 1} |
| E | Original GFS | 20 | 20 | 20 | {"search_completed": 1, "stagnated": 19} |
| E | Original GFS high precision | 20 | 20 | 20 | {"search_completed": 1, "stagnated": 19} |
| E | CP(2) | 1 | 1 | 1 | {"completed": 1} |
| E | Sinha | 1 | 1 | 0 | {"NA — theoretical restrictions not satisfied": 1} |
| E | DSD | 20 | 20 | 20 | {"completed": 20} |
| E | Exact LP/QP | 1 | 1 | 1 | {"certified_feasible": 1} |
| U | Initial | 1 | 1 | 1 | {"initial": 1} |
| U | Original GFS | 20 | 20 | 20 | {"budget_exhausted": 15, "search_completed": 1, "stagnated": 4} |
| U | Original GFS high precision | 20 | 20 | 20 | {"budget_exhausted": 15, "search_completed": 1, "stagnated": 4} |
| U | CP(2) | 1 | 1 | 1 | {"completed": 1} |
| U | Sinha | 1 | 1 | 0 | {"NA — theoretical restrictions not satisfied": 1} |
| U | DSD | 20 | 20 | 20 | {"completed": 20} |
| U | Exact LP/QP | 1 | 1 | 1 | {"certified_feasible": 1} |

Threshold counts:

| case | method | runs | threshold_1e_02_count | threshold_1e_03_count | threshold_1e_04_count | threshold_1e_05_count | threshold_1e_06_count | threshold_1e_08_count | threshold_1e_10_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | Initial | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E | Original GFS | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E | Original GFS high precision | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E | CP(2) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| E | Sinha | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E | DSD | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E | Exact LP/QP | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| U | Initial | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U | Original GFS | 20 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| U | Original GFS high precision | 20 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| U | CP(2) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| U | Sinha | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U | DSD | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| U | Exact LP/QP | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

## Original optimizer and precision sensitivity

Original GFS invokes the package's GreedyBestFirstSearch.run and its existing exchange mechanism. The reconstruction criterion is D2. Nine-decimal D2 comparisons are retained in the main method. D2 below approximately 5e-10 can round to zero, corresponding to RMSE below sqrt(5e-10/66)=2.7524094e-6. This is comparison resolution, not a guaranteed RMSE floor: the frontier heap still orders its raw loss values, and other search decisions may use rounded values. The high-precision criterion executes the same search bytecode with score rounding removed; it is a numerical diagnostic rather than a new sampling method. The stochastic budget is at most 5,000 scored designs and early reconstruction stopping is RMSE<1e-10. Original GFS uses frontier size 50, ten exchange and two order neighbors, coefficient 0.75, one zone, a stagnation shock after 100 iterations, backtrack depth zero, and termination after 2,000 scored designs without incumbent improvement. The observer returns the best evaluated incumbent, including candidates scored in a batch interrupted by the candidate budget, using the same Original GFS or high-precision comparison rule as that run. Exact per-run settings, frontier/exchange parameters and termination details follow.

```json
{
  "E/Original GFS high precision": {
    "score": "D2",
    "score_rounding_decimals": null,
    "same_original_run_bytecode": true,
    "search_options": {
      "max_iterations": 10000,
      "max_open_set_size": 50,
      "top_k": 1,
      "num_new_order_nodes": 2,
      "num_new_exchange_nodes": 10,
      "num_clusters_range": [
        1,
        1
      ],
      "num_zones_range": [
        1,
        1
      ],
      "num_changes_range": [
        1,
        1
      ],
      "num_zone_changes": 0,
      "pull_strategy": "default",
      "exchange_coef": 0.75,
      "num_explore": 1,
      "n_jobs": 1,
      "backtrack_depth": 0,
      "stagnation_limit": 100,
      "output_mode": "silent"
    },
    "last_improvement_candidate": 0,
    "no_improvement_stop_candidates": 2000,
    "iterations": 510,
    "profile": {
      "criterion_seconds": 0.473483319987281,
      "SIP_computation_seconds": 0.02183088499805308,
      "support_extraction_seconds": 0.3597547020017373,
      "candidate_generation_seconds": 0.7390640189660189,
      "design_copy_seconds": 0.3650590680008463,
      "design_update_seconds": 0.36150197499046044,
      "validation_seconds": 0.03818399499868974
    },
    "evaluations_per_second": 783.7157230402778,
    "average_candidate_criterion_seconds": 0.0002367416599936405,
    "peak_process_RSS_MB": 272.11328125,
    "memory_measurement": "process-lifetime RSS high water; not incremental method memory",
    "high_precision_implementation": "same run bytecode; isolated round binding returns raw float",
    "initial_D2": 0.1918229084543658,
    "initial_RMSE": 0.05391110928050013
  },
  "E/Original GFS": {
    "score": "D2",
    "score_rounding_decimals": 9,
    "same_original_run_bytecode": true,
    "search_options": {
      "max_iterations": 10000,
      "max_open_set_size": 50,
      "top_k": 1,
      "num_new_order_nodes": 2,
      "num_new_exchange_nodes": 10,
      "num_clusters_range": [
        1,
        1
      ],
      "num_zones_range": [
        1,
        1
      ],
      "num_changes_range": [
        1,
        1
      ],
      "num_zone_changes": 0,
      "pull_strategy": "default",
      "exchange_coef": 0.75,
      "num_explore": 1,
      "n_jobs": 1,
      "backtrack_depth": 0,
      "stagnation_limit": 100,
      "output_mode": "silent"
    },
    "last_improvement_candidate": 0,
    "no_improvement_stop_candidates": 2000,
    "iterations": 510,
    "profile": {
      "criterion_seconds": 0.5192506360108382,
      "SIP_computation_seconds": 0.021330104042135645,
      "support_extraction_seconds": 0.40533894508553203,
      "candidate_generation_seconds": 0.7133344370249688,
      "design_copy_seconds": 0.3523373690131848,
      "design_update_seconds": 0.34887070792683517,
      "validation_seconds": 0.037607583988574333
    },
    "evaluations_per_second": 809.8709441978465,
    "average_candidate_criterion_seconds": 0.0002596253180054191,
    "peak_process_RSS_MB": 272.11328125,
    "memory_measurement": "process-lifetime RSS high water; not incremental method memory",
    "high_precision_implementation": "unmodified run and builtin round",
    "initial_D2": 0.1918229084543658,
    "initial_RMSE": 0.05391110928050013
  },
  "U/Original GFS high precision": {
    "score": "D2",
    "score_rounding_decimals": null,
    "same_original_run_bytecode": true,
    "search_options": {
      "max_iterations": 10000,
      "max_open_set_size": 50,
      "top_k": 1,
      "num_new_order_nodes": 2,
      "num_new_exchange_nodes": 10,
      "num_clusters_range": [
        1,
        1
      ],
      "num_zones_range": [
        1,
        1
      ],
      "num_changes_range": [
        1,
        1
      ],
      "num_zone_changes": 0,
      "pull_strategy": "default",
      "exchange_coef": 0.75,
      "num_explore": 1,
      "n_jobs": 1,
      "backtrack_depth": 0,
      "stagnation_limit": 100,
      "output_mode": "silent"
    },
    "last_improvement_candidate": 4992,
    "no_improvement_stop_candidates": 2000,
    "iterations": 835,
    "profile": {
      "criterion_seconds": 7.353945105933235,
      "SIP_computation_seconds": 0.10761991294384643,
      "support_extraction_seconds": 6.4219282420417585,
      "candidate_generation_seconds": 2.1931118790489563,
      "design_copy_seconds": 1.1032856661240658,
      "design_update_seconds": 1.0563372289543622,
      "validation_seconds": 0.21773863108865044
    },
    "evaluations_per_second": 194.83277299707518,
    "average_candidate_criterion_seconds": 0.001470789021186647,
    "peak_process_RSS_MB": 281.96484375,
    "memory_measurement": "process-lifetime RSS high water; not incremental method memory",
    "high_precision_implementation": "same run bytecode; isolated round binding returns raw float",
    "initial_D2": 0.016003911258155305,
    "initial_RMSE": 0.015571881835906796
  },
  "U/Original GFS": {
    "score": "D2",
    "score_rounding_decimals": 9,
    "same_original_run_bytecode": true,
    "search_options": {
      "max_iterations": 10000,
      "max_open_set_size": 50,
      "top_k": 1,
      "num_new_order_nodes": 2,
      "num_new_exchange_nodes": 10,
      "num_clusters_range": [
        1,
        1
      ],
      "num_zones_range": [
        1,
        1
      ],
      "num_changes_range": [
        1,
        1
      ],
      "num_zone_changes": 0,
      "pull_strategy": "default",
      "exchange_coef": 0.75,
      "num_explore": 1,
      "n_jobs": 1,
      "backtrack_depth": 0,
      "stagnation_limit": 100,
      "output_mode": "silent"
    },
    "last_improvement_candidate": 4987,
    "no_improvement_stop_candidates": 2000,
    "iterations": 835,
    "profile": {
      "criterion_seconds": 7.215911480921932,
      "SIP_computation_seconds": 0.10788751501786464,
      "support_extraction_seconds": 6.288163012070072,
      "candidate_generation_seconds": 2.131292147025306,
      "design_copy_seconds": 1.0720577639785915,
      "design_update_seconds": 1.0264995760589954,
      "validation_seconds": 0.21572243494119903
    },
    "evaluations_per_second": 199.85658089500654,
    "average_candidate_criterion_seconds": 0.0014431822961843864,
    "peak_process_RSS_MB": 281.6015625,
    "memory_measurement": "process-lifetime RSS high water; not incremental method memory",
    "high_precision_implementation": "unmodified run and builtin round",
    "initial_D2": 0.016003911258155305,
    "initial_RMSE": 0.015571881835906796
  }
}
```

## Plateau diagnostics

Last recorded improvement uses incumbent checkpoints and requires an RMSE reduction larger than max(1e-15,abs(previous incumbent RMSE)*1e-12), excluding roundoff-only final recomputation changes. actual_incumbent_last_improvement_candidates retains the optimizer's recorded details.last_improvement_candidate separately. Late relative improvement compares the last checkpoint with the best checkpoint at or before 80% of its candidate count. Checkpoint resolution limits this diagnosis. Finite-budget stagnation does not prove trapping in a local optimum; continuing improvement indicates that the tested budget ended before a recorded plateau.

| case | method | replication | status | final_candidate_count | checkpoints | last_recorded_improvement_candidates | actual_incumbent_last_improvement_candidates | late_relative_RMSE_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E | DSD | 0 | completed | 5000 | 101 | 150 | 145 | 0 |
| E | DSD | 1 | completed | 5000 | 101 | 50 | 41 | 0 |
| E | DSD | 2 | completed | 5000 | 101 | 100 | 80 | 0 |
| E | DSD | 3 | completed | 5000 | 101 | 150 | 110 | 0 |
| E | DSD | 4 | completed | 5000 | 101 | 100 | 95 | 0 |
| E | DSD | 5 | completed | 5000 | 101 | 350 | 313 | 0 |
| E | DSD | 6 | completed | 5000 | 101 | 100 | 63 | 0 |
| E | DSD | 7 | completed | 5000 | 101 | 200 | 152 | 0 |
| E | DSD | 8 | completed | 5000 | 101 | 100 | 77 | 0 |
| E | DSD | 9 | completed | 5000 | 101 | 150 | 133 | 0 |
| E | DSD | 10 | completed | 5000 | 101 | 250 | 238 | 0 |
| E | DSD | 11 | completed | 5000 | 101 | 150 | 142 | 0 |
| E | DSD | 12 | completed | 5000 | 101 | 200 | 181 | 0 |
| E | DSD | 13 | completed | 5000 | 101 | 150 | 130 | 0 |
| E | DSD | 14 | completed | 5000 | 101 | 100 | 88 | 0 |
| E | DSD | 15 | completed | 5000 | 101 | 150 | 107 | 0 |
| E | DSD | 16 | completed | 5000 | 101 | 300 | 260 | 0 |
| E | DSD | 17 | completed | 5000 | 101 | 150 | 139 | 0 |
| E | DSD | 18 | completed | 5000 | 101 | 350 | 301 | 0 |
| E | DSD | 19 | completed | 5000 | 101 | 200 | 166 | 0 |
| E | Original GFS high precision | 0 | stagnated | 2000 | 41 | 0 | 0 | 0 |
| E | Original GFS high precision | 1 | stagnated | 2002 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 2 | stagnated | 2006 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 3 | stagnated | 2000 | 41 | 0 | 0 | 0 |
| E | Original GFS high precision | 4 | stagnated | 3855 | 79 | 1900 | 1854 | 0 |
| E | Original GFS high precision | 5 | stagnated | 3588 | 73 | 1600 | 1587 | 0 |
| E | Original GFS high precision | 6 | stagnated | 2001 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 7 | stagnated | 3857 | 79 | 1900 | 1856 | 0 |
| E | Original GFS high precision | 8 | stagnated | 2002 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 9 | search_completed | 868 | 19 | 0 | 0 | 0 |
| E | Original GFS high precision | 10 | stagnated | 2001 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 11 | stagnated | 2001 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 12 | stagnated | 2001 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 13 | stagnated | 2002 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 14 | stagnated | 2000 | 41 | 0 | 0 | 0 |
| E | Original GFS high precision | 15 | stagnated | 2005 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 16 | stagnated | 2002 | 42 | 0 | 0 | 0 |
| E | Original GFS high precision | 17 | stagnated | 2792 | 57 | 800 | 785 | 0 |
| E | Original GFS high precision | 18 | stagnated | 2232 | 46 | 250 | 232 | 0 |
| E | Original GFS high precision | 19 | stagnated | 2509 | 52 | 550 | 509 | 0 |
| E | Original GFS | 0 | stagnated | 2000 | 41 | 0 | 0 | 0 |
| E | Original GFS | 1 | stagnated | 2002 | 42 | 0 | 0 | 0 |
| E | Original GFS | 2 | stagnated | 2006 | 42 | 0 | 0 | 0 |
| E | Original GFS | 3 | stagnated | 2000 | 41 | 0 | 0 | 0 |
| E | Original GFS | 4 | stagnated | 3855 | 79 | 1900 | 1854 | 0 |
| E | Original GFS | 5 | stagnated | 3588 | 73 | 1600 | 1587 | 0 |
| E | Original GFS | 6 | stagnated | 2001 | 42 | 0 | 0 | 0 |
| E | Original GFS | 7 | stagnated | 3509 | 72 | 1550 | 1508 | 0 |
| E | Original GFS | 8 | stagnated | 2002 | 42 | 0 | 0 | 0 |
| E | Original GFS | 9 | search_completed | 868 | 19 | 0 | 0 | 0 |
| E | Original GFS | 10 | stagnated | 2000 | 41 | 0 | 0 | 0 |
| E | Original GFS | 11 | stagnated | 2001 | 42 | 0 | 0 | 0 |
| E | Original GFS | 12 | stagnated | 2001 | 42 | 0 | 0 | 0 |
| E | Original GFS | 13 | stagnated | 2002 | 42 | 0 | 0 | 0 |
| E | Original GFS | 14 | stagnated | 2000 | 41 | 0 | 0 | 0 |
| E | Original GFS | 15 | stagnated | 2005 | 42 | 0 | 0 | 0 |
| E | Original GFS | 16 | stagnated | 2002 | 42 | 0 | 0 | 0 |
| E | Original GFS | 17 | stagnated | 2792 | 57 | 800 | 785 | 0 |
| E | Original GFS | 18 | stagnated | 2233 | 46 | 250 | 232 | 0 |
| E | Original GFS | 19 | stagnated | 2509 | 52 | 550 | 509 | 0 |
| U | DSD | 0 | completed | 5000 | 101 | 400 | 470 | 0 |
| U | DSD | 1 | completed | 5000 | 101 | 550 | 538 | 0 |
| U | DSD | 2 | completed | 5000 | 101 | 550 | 523 | 0 |
| U | DSD | 3 | completed | 5000 | 101 | 150 | 209 | 0 |
| U | DSD | 4 | completed | 5000 | 101 | 450 | 438 | 0 |
| U | DSD | 5 | completed | 5000 | 101 | 100 | 59 | 0 |
| U | DSD | 6 | completed | 5000 | 101 | 300 | 325 | 0 |
| U | DSD | 7 | completed | 5000 | 101 | 300 | 315 | 0 |
| U | DSD | 8 | completed | 5000 | 101 | 400 | 353 | 0 |
| U | DSD | 9 | completed | 5000 | 101 | 150 | 150 | 0 |
| U | DSD | 10 | completed | 5000 | 101 | 500 | 519 | 0 |
| U | DSD | 11 | completed | 5000 | 101 | 300 | 275 | 0 |
| U | DSD | 12 | completed | 5000 | 101 | 150 | 172 | 0 |
| U | DSD | 13 | completed | 5000 | 101 | 300 | 287 | 0 |
| U | DSD | 14 | completed | 5000 | 101 | 200 | 332 | 0 |
| U | DSD | 15 | completed | 5000 | 101 | 250 | 239 | 0 |
| U | DSD | 16 | completed | 5000 | 101 | 250 | 252 | 0 |
| U | DSD | 17 | completed | 5000 | 101 | 150 | 179 | 0 |
| U | DSD | 18 | completed | 5000 | 101 | 350 | 371 | 0 |
| U | DSD | 19 | completed | 5000 | 101 | 250 | 240 | 0 |
| U | Original GFS high precision | 0 | budget_exhausted | 5000 | 101 | 5000 | 4992 | 7.9008441e-05 |
| U | Original GFS high precision | 1 | budget_exhausted | 5000 | 101 | 4900 | 4861 | 0.0010941967 |
| U | Original GFS high precision | 2 | budget_exhausted | 5000 | 101 | 4850 | 4827 | 0.016896682 |
| U | Original GFS high precision | 3 | budget_exhausted | 5000 | 101 | 5000 | 5000 | 6.0444384e-05 |
| U | Original GFS high precision | 4 | budget_exhausted | 5000 | 101 | 5000 | 4985 | 0.0037702174 |
| U | Original GFS high precision | 5 | budget_exhausted | 5000 | 101 | 4750 | 4738 | 0.01383724 |
| U | Original GFS high precision | 6 | budget_exhausted | 5000 | 101 | 5000 | 4967 | 5.8695473e-06 |
| U | Original GFS high precision | 7 | budget_exhausted | 5000 | 101 | 5000 | 4995 | 7.324706e-05 |
| U | Original GFS high precision | 8 | budget_exhausted | 5000 | 101 | 5000 | 4997 | 0.00050957147 |
| U | Original GFS high precision | 9 | stagnated | 2409 | 50 | 450 | 407 | 0 |
| U | Original GFS high precision | 10 | stagnated | 3290 | 67 | 1300 | 1288 | 0 |
| U | Original GFS high precision | 11 | budget_exhausted | 5000 | 101 | 5000 | 4979 | 0.0015204367 |
| U | Original GFS high precision | 12 | stagnated | 4682 | 95 | 2700 | 2680 | 0 |
| U | Original GFS high precision | 13 | budget_exhausted | 5000 | 101 | 4850 | 4830 | 2.1101945e-05 |
| U | Original GFS high precision | 14 | budget_exhausted | 5000 | 101 | 4750 | 4742 | 0.0002945375 |
| U | Original GFS high precision | 15 | search_completed | 4768 | 97 | 4100 | 4057 | 5.8711302e-05 |
| U | Original GFS high precision | 16 | budget_exhausted | 5000 | 101 | 5000 | 4998 | 1.0860817e-05 |
| U | Original GFS high precision | 17 | budget_exhausted | 5000 | 101 | 4950 | 4923 | 0.00024062287 |
| U | Original GFS high precision | 18 | stagnated | 2829 | 58 | 850 | 829 | 0 |
| U | Original GFS high precision | 19 | budget_exhausted | 5000 | 101 | 5000 | 4979 | 6.4590694e-05 |
| U | Original GFS | 0 | budget_exhausted | 5000 | 101 | 5000 | 4987 | 7.900842e-05 |
| U | Original GFS | 1 | budget_exhausted | 5000 | 101 | 4600 | 4558 | 0.001094175 |
| U | Original GFS | 2 | budget_exhausted | 5000 | 101 | 4400 | 4382 | 0.0004060485 |
| U | Original GFS | 3 | budget_exhausted | 5000 | 101 | 5000 | 5000 | 6.0444384e-05 |
| U | Original GFS | 4 | budget_exhausted | 5000 | 101 | 5000 | 4985 | 0.0037702174 |
| U | Original GFS | 5 | budget_exhausted | 5000 | 101 | 4950 | 4934 | 0.013839675 |
| U | Original GFS | 6 | budget_exhausted | 5000 | 101 | 4750 | 4734 | 5.0628417e-06 |
| U | Original GFS | 7 | budget_exhausted | 5000 | 101 | 5000 | 4991 | 7.3246934e-05 |
| U | Original GFS | 8 | budget_exhausted | 5000 | 101 | 4950 | 4925 | 0.00043570223 |
| U | Original GFS | 9 | stagnated | 2409 | 50 | 450 | 407 | 0 |
| U | Original GFS | 10 | stagnated | 3290 | 67 | 1300 | 1288 | 0 |
| U | Original GFS | 11 | budget_exhausted | 5000 | 101 | 4950 | 4939 | 0.0015204367 |
| U | Original GFS | 12 | stagnated | 4684 | 95 | 2700 | 2680 | 0 |
| U | Original GFS | 13 | budget_exhausted | 5000 | 101 | 4950 | 4936 | 2.6228235e-05 |
| U | Original GFS | 14 | budget_exhausted | 5000 | 101 | 4750 | 4742 | 0.0002945375 |
| U | Original GFS | 15 | search_completed | 4768 | 97 | 4100 | 4057 | 5.8711302e-05 |
| U | Original GFS | 16 | budget_exhausted | 5000 | 101 | 5000 | 4985 | 1.0860817e-05 |
| U | Original GFS | 17 | budget_exhausted | 5000 | 101 | 4800 | 4758 | 0.00024062198 |
| U | Original GFS | 18 | stagnated | 2830 | 58 | 850 | 829 | 0 |
| U | Original GFS | 19 | budget_exhausted | 5000 | 101 | 5000 | 4979 | 6.4590694e-05 |

## Competitors and numerical oracle

CP(2) is fitted with exact enumeration of the exponential-family partition function, log-partition dual L-BFGS calibration and mandatory Newton FIP tilt. This calibration differs from Bondesson's published iterative/Gibbs implementation; exact enumeration and our optimizer are not presented as a reproduction of that sampling implementation. Intermediate calibration iterates need not preserve prescribed FIP. Final FIP checks, mandatory calibration time/evaluations and optimizer status are retained in run details.

DSD performs projection/order fitting within a restricted kernel family with the same FIP. Its SIP obey pi_ij=pi_i*pi_j-|K_ij|^2, restricting the pair structure; arbitrary prescribed SIP need not be representable. The existing projection factor construction is fitted by randomized swaps of the population order, then rows are restored to original unit identifiers. Twenty repetitions per case use at most 5,000 scored kernel candidates. This optimizes a restricted family, rather than arbitrary projection kernels. The final fixed-size design is enumerated as p(s)=det(V_s)^2; its moments and FIP are independently audited.

Sinha uses the nonnegative pair-removal mixture. For N=12,n=4 its coefficients are b_ij=7.5*pi_ij-2.5*(pi_i+pi_j)+1. All coefficients must be nonnegative (up to stated numerical tolerance) and sum to one. Unsatisfied theoretical restrictions produce NA, without clipping or modifying the method.

Exact LP/QP enumerates all 495 fixed-size samples and enforces nonnegative probabilities, unit mass, prescribed FIP and full SIP equalities. This is an oracle numerical benchmark, not a practical large-N sampling algorithm. A tiny objective is interpreted together with solver status and independently measured feasibility residuals; floating-point optimization is not symbolic exact arithmetic.

### E — CP(2)

Status: completed; optimum/returned RMSE=2.5463431e-09; D2 objective=4.2793496e-16; D_inf=5.5909855e-09.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "calibration_implementation": "enumerated log-partition dual L-BFGS + Newton FIP tilt",
    "first_order_calibration_evaluations": 1,
    "final_FIP_calibration_evaluations": 3,
    "mandatory_finalization_seconds": 0.00281271200037736,
    "optimizer_success": true,
    "exact_partition_function": true
  },
  "validation": {
    "max_probability_sum_error": 8.881784197001252e-16,
    "max_FIP_error": 6.661338147750939e-16,
    "min_sample_probability": 5.623701717665288e-11,
    "max_SIP_recompute_error": 0.0,
    "max_fixed_size_row_identity_error": 1.7763568394002505e-15,
    "total_pair_identity_error": 8.881784197001252e-16,
    "max_SIP_diagonal_FIP_error": 6.661338147750939e-16,
    "positive_samples_correct_size": true,
    "max_SIP_incremental_error": 0.0
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      6,
      10,
      5,
      9,
      4,
      8,
      3,
      7,
      2
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 145,
    "seed": 7322345195371504669,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 4.440892098500626e-16,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 4.440892098500626e-16,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      3,
      7,
      2,
      6,
      10,
      5,
      9,
      4,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 41,
    "seed": 5036730523617105529,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      6,
      10,
      5,
      0,
      1,
      11,
      9,
      4,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 80,
    "seed": 8017913645284567639,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 8.881784197001252e-16,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 8.881784197001252e-16,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      3,
      7,
      2,
      6,
      10,
      5,
      4,
      9,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 110,
    "seed": 4093828291588030673,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 8.881784197001252e-16,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 8.881784197001252e-16,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      6,
      10,
      5,
      9,
      4,
      8,
      0,
      1,
      11
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 95,
    "seed": 7832968096705483866,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 4.440892098500626e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 4.440892098500626e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      9,
      4,
      8,
      6,
      10,
      5,
      3,
      7,
      2,
      0,
      1,
      11
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 313,
    "seed": 3556446854515349322,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      6,
      10,
      5,
      9,
      4,
      8,
      0,
      1,
      11
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 63,
    "seed": 4856218541370079112,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 4.440892098500626e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 4.440892098500626e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      0,
      1,
      11,
      6,
      10,
      5,
      9,
      4,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 152,
    "seed": 6530883675518019135,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      9,
      4,
      8,
      6,
      5,
      10,
      3,
      7,
      2
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 77,
    "seed": 2276741675353979965,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 4.440892098500626e-16,
      "total_pair_identity_error": 8.881784197001252e-16,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 4.440892098500626e-16,
    "total_pair_identity_error": 8.881784197001252e-16,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      0,
      1,
      11,
      6,
      10,
      5,
      9,
      4,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 133,
    "seed": 5847951929411535538,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      9,
      4,
      8,
      3,
      7,
      2,
      6,
      10,
      5,
      0,
      1,
      11
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 238,
    "seed": 2230157817947606861,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      9,
      4,
      8,
      0,
      1,
      11,
      6,
      10,
      5,
      3,
      7,
      2
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 142,
    "seed": 2457623694639806224,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      0,
      1,
      11,
      9,
      4,
      8,
      6,
      10,
      5
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 181,
    "seed": 629027311195002555,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      6,
      10,
      5,
      3,
      7,
      2,
      9,
      4,
      8,
      0,
      1,
      11
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 130,
    "seed": 5991294925803958817,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 4.440892098500626e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 4.440892098500626e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      3,
      7,
      2,
      6,
      10,
      5,
      9,
      4,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 88,
    "seed": 5461990710695715632,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      6,
      5,
      10,
      3,
      7,
      2,
      0,
      1,
      11,
      9,
      4,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 107,
    "seed": 8737391001908991005,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 8.881784197001252e-16,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 8.881784197001252e-16,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      9,
      4,
      8,
      6,
      10,
      5,
      0,
      1,
      11
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 260,
    "seed": 2080753359753358332,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      6,
      10,
      5,
      9,
      4,
      8,
      0,
      1,
      11
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 139,
    "seed": 9147877860218336328,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 4.440892098500626e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 4.440892098500626e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      3,
      7,
      2,
      6,
      10,
      5,
      4,
      9,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 301,
    "seed": 2227902193069359399,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 8.881784197001252e-16,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 8.881784197001252e-16,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — DSD

Status: completed; optimum/returned RMSE=0.10363853; D2 objective=0.70890232; D_inf=0.20261647.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      3,
      7,
      2,
      6,
      10,
      5,
      0,
      1,
      11,
      9,
      4,
      8
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.10759436884651079,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 166,
    "seed": 3538213205828577373,
    "validation": {
      "max_probability_sum_error": 0.0,
      "max_FIP_error": 1.6653345369377348e-16,
      "min_sample_probability": 0.012345679012345663,
      "max_SIP_recompute_error": 1.6653345369377348e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 8.881784197001252e-16,
      "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 1.6653345369377348e-16,
    "min_sample_probability": 0.012345679012345663,
    "max_SIP_recompute_error": 1.6653345369377348e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 8.881784197001252e-16,
    "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16,
    "positive_samples_correct_size": true
  }
}
```

### E — Exact LP/QP

Status: certified_feasible; optimum/returned RMSE=6.4164603e-16; D2 objective=2.7172836e-29; D_inf=4.6698756e-15.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "LP_status": 0,
    "LP_message": "Optimization terminated successfully. (HiGHS Status 7: Optimal)",
    "enumerated_samples": 495,
    "feasible": true,
    "optimal_D2": 2.717283560610805e-29,
    "lower_bound_D2": 0.0,
    "certificate_gap": 2.717283560610805e-29,
    "primal_error": 1.942890293094024e-15,
    "status": "certified_feasible",
    "runtime_seconds": 0.009326330000476446
  },
  "validation": {
    "max_probability_sum_error": 7.771561172376096e-16,
    "max_FIP_error": 1.942890293094024e-15,
    "min_sample_probability": 2.5236945155576295e-05,
    "max_SIP_recompute_error": 0.0,
    "max_fixed_size_row_identity_error": 5.88418203051333e-15,
    "total_pair_identity_error": 4.440892098500626e-15,
    "max_SIP_diagonal_FIP_error": 1.942890293094024e-15,
    "positive_samples_correct_size": true
  }
}
```

### E — Sinha

Status: NA — theoretical restrictions not satisfied; optimum/returned RMSE=NA; D2 objective=NA; D_inf=NA.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "minimum_mixture_coefficient": -0.6619011177038778
  },
  "validation": {}
}
```

### U — CP(2)

Status: completed; optimum/returned RMSE=2.6932492e-09; D2 objective=4.7873703e-16; D_inf=7.5673668e-09.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "calibration_implementation": "enumerated log-partition dual L-BFGS + Newton FIP tilt",
    "first_order_calibration_evaluations": 9,
    "final_FIP_calibration_evaluations": 3,
    "mandatory_finalization_seconds": 0.00259877200005576,
    "optimizer_success": true,
    "exact_partition_function": true
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 5.551115123125783e-16,
    "min_sample_probability": 3.857476139072209e-09,
    "max_SIP_recompute_error": 0.0,
    "max_fixed_size_row_identity_error": 8.881784197001252e-16,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 5.551115123125783e-16,
    "positive_samples_correct_size": true,
    "max_SIP_incremental_error": 0.0
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 470,
    "seed": 9137925729275444370,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 538,
    "seed": 4804693668282405713,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 523,
    "seed": 313904572923105607,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 209,
    "seed": 4022327856969038635,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 438,
    "seed": 7173143235791977984,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099623714; D2 objective=0.65504238; D_inf=0.19406987.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      5,
      4,
      2,
      3,
      7,
      6,
      8,
      11,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 59,
    "seed": 3291435451682411669,
    "validation": {
      "max_probability_sum_error": 4.440892098500626e-16,
      "max_FIP_error": 3.0531133177191805e-16,
      "min_sample_probability": 1.5173268113177047e-80,
      "max_SIP_recompute_error": 3.0531133177191805e-16,
      "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
      "total_pair_identity_error": 3.552713678800501e-15,
      "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 4.440892098500626e-16,
    "max_FIP_error": 3.0531133177191805e-16,
    "min_sample_probability": 1.5173268113177047e-80,
    "max_SIP_recompute_error": 3.0531133177191805e-16,
    "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
    "total_pair_identity_error": 3.552713678800501e-15,
    "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 325,
    "seed": 3549434190123542994,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 315,
    "seed": 899820335042296476,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      11,
      1,
      0,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 353,
    "seed": 347212499464721922,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099623714; D2 objective=0.65504238; D_inf=0.19406987.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      5,
      4,
      3,
      2,
      7,
      6,
      8,
      9,
      11,
      10
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 150,
    "seed": 4173206086477835845,
    "validation": {
      "max_probability_sum_error": 2.220446049250313e-16,
      "max_FIP_error": 2.7755575615628914e-16,
      "min_sample_probability": 4.929423778198965e-81,
      "max_SIP_recompute_error": 2.7755575615628914e-16,
      "max_fixed_size_row_identity_error": 6.661338147750939e-16,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 2.220446049250313e-16,
    "max_FIP_error": 2.7755575615628914e-16,
    "min_sample_probability": 4.929423778198965e-81,
    "max_SIP_recompute_error": 2.7755575615628914e-16,
    "max_fixed_size_row_identity_error": 6.661338147750939e-16,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 519,
    "seed": 1631674949202766930,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 275,
    "seed": 8555542978525500281,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099623714; D2 objective=0.65504238; D_inf=0.19406987.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      5,
      4,
      2,
      3,
      7,
      6,
      8,
      9,
      11,
      10
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 172,
    "seed": 5968956535183140506,
    "validation": {
      "max_probability_sum_error": 2.220446049250313e-16,
      "max_FIP_error": 3.0531133177191805e-16,
      "min_sample_probability": 3.839886735088039e-45,
      "max_SIP_recompute_error": 3.0531133177191805e-16,
      "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 2.220446049250313e-16,
    "max_FIP_error": 3.0531133177191805e-16,
    "min_sample_probability": 3.839886735088039e-45,
    "max_SIP_recompute_error": 3.0531133177191805e-16,
    "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 287,
    "seed": 8033832533951440055,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 332,
    "seed": 3471989400043636910,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 239,
    "seed": 1036563774019930093,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099623714; D2 objective=0.65504238; D_inf=0.19406987.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      5,
      4,
      2,
      3,
      7,
      6,
      8,
      9,
      11,
      10
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 252,
    "seed": 6522884013157391093,
    "validation": {
      "max_probability_sum_error": 2.220446049250313e-16,
      "max_FIP_error": 3.0531133177191805e-16,
      "min_sample_probability": 3.839886735088039e-45,
      "max_SIP_recompute_error": 3.0531133177191805e-16,
      "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 2.220446049250313e-16,
    "max_FIP_error": 3.0531133177191805e-16,
    "min_sample_probability": 3.839886735088039e-45,
    "max_SIP_recompute_error": 3.0531133177191805e-16,
    "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099623714; D2 objective=0.65504238; D_inf=0.19406987.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      5,
      4,
      2,
      3,
      7,
      6,
      8,
      9,
      11,
      10
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 179,
    "seed": 6645161485909024897,
    "validation": {
      "max_probability_sum_error": 2.220446049250313e-16,
      "max_FIP_error": 3.0531133177191805e-16,
      "min_sample_probability": 3.839886735088039e-45,
      "max_SIP_recompute_error": 3.0531133177191805e-16,
      "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
      "total_pair_identity_error": 1.7763568394002505e-15,
      "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 2.220446049250313e-16,
    "max_FIP_error": 3.0531133177191805e-16,
    "min_sample_probability": 3.839886735088039e-45,
    "max_SIP_recompute_error": 3.0531133177191805e-16,
    "max_fixed_size_row_identity_error": 1.1102230246251565e-15,
    "total_pair_identity_error": 1.7763568394002505e-15,
    "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 371,
    "seed": 3342477498229052653,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — DSD

Status: completed; optimum/returned RMSE=0.099549157; D2 objective=0.65406228; D_inf=0.1940706.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "construction": "existing projection_factor, with fitted unit ordering only",
    "best_order": [
      0,
      1,
      11,
      2,
      4,
      3,
      8,
      7,
      6,
      5,
      10,
      9
    ],
    "order_search": "uniform random pair swaps of incumbent; independent random order every 100 candidates",
    "initial_fixed_order_RMSE": 0.09982809897512772,
    "kernel_family": "restricted ordered rotation projections; not arbitrary prescribed SIP",
    "last_improvement_candidate": 240,
    "seed": 501718556856130037,
    "validation": {
      "max_probability_sum_error": 1.1102230246251565e-16,
      "max_FIP_error": 4.996003610813204e-16,
      "min_sample_probability": 3.4971977144972687e-75,
      "max_SIP_recompute_error": 4.996003610813204e-16,
      "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
      "total_pair_identity_error": 0.0,
      "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
      "positive_samples_correct_size": true
    }
  },
  "validation": {
    "max_probability_sum_error": 1.1102230246251565e-16,
    "max_FIP_error": 4.996003610813204e-16,
    "min_sample_probability": 3.4971977144972687e-75,
    "max_SIP_recompute_error": 4.996003610813204e-16,
    "max_fixed_size_row_identity_error": 1.3322676295501878e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 4.996003610813204e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — Exact LP/QP

Status: certified_feasible; optimum/returned RMSE=2.8617203e-16; D2 objective=5.4050323e-30; D_inf=1.8908486e-15.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "LP_status": 0,
    "LP_message": "Optimization terminated successfully. (HiGHS Status 7: Optimal)",
    "enumerated_samples": 495,
    "feasible": true,
    "optimal_D2": 5.4050322990357304e-30,
    "lower_bound_D2": 0.0,
    "certificate_gap": 5.4050322990357304e-30,
    "primal_error": 6.661338147750939e-16,
    "status": "certified_feasible",
    "runtime_seconds": 0.007654428000023472
  },
  "validation": {
    "max_probability_sum_error": 0.0,
    "max_FIP_error": 6.661338147750939e-16,
    "min_sample_probability": 3.934807277406816e-05,
    "max_SIP_recompute_error": 0.0,
    "max_fixed_size_row_identity_error": 2.220446049250313e-15,
    "total_pair_identity_error": 0.0,
    "max_SIP_diagonal_FIP_error": 6.661338147750939e-16,
    "positive_samples_correct_size": true
  }
}
```

### U — Sinha

Status: NA — theoretical restrictions not satisfied; optimum/returned RMSE=NA; D2 objective=NA; D_inf=NA.
Solver/calibration/applicability details and reported feasibility residuals:
```json
{
  "details": {
    "minimum_mixture_coefficient": -0.8866287604394172
  },
  "validation": {}
}
```

### Projection DSD representational restriction

Every projection DSD satisfies pi_ij=pi_i*pi_j-|K_ij|^2<=pi_i*pi_j. A strictly positive target pair covariance therefore prevents exact representation by any projection DSD, including kernels outside the fitted order family.

Equal FIP target: 18 of 66 pairs have positive covariance above the 1e-12 numerical tolerance; maximum positive covariance is 0.2026164747193881. These violations impose a structural obstacle to exact projection-DSD recovery.

Unequal FIP target: 19 of 66 pairs have positive covariance above the 1e-12 numerical tolerance; maximum positive covariance is 0.19406985406092198. These violations impose a structural obstacle to exact projection-DSD recovery.

## Independent validation

| design | audit_status | probability_sum_error | min_probability | max_FIP_error | invalid_samples | max_saved_SIP_discrepancy | max_row_identity_error | pair_sum_identity_error | reported_validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E/p0 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | NA |
| E/pstar | passed | 9.9920072e-16 | 5.9064642e-10 | 6.1062266e-16 | 0 | 0 | 1.110223e-15 | 5.3290705e-15 | NA |
| U/p0 | passed | 0 | 0.0087428972 | 2.220446e-16 | 0 | 0 | 6.6613381e-16 | 0 | NA |
| U/pstar | passed | 0 | 2.3224288e-10 | 2.7755576e-16 | 0 | 0 | 5.5511151e-16 | 8.8817842e-16 | NA |
| E/CP(2)/r0 | passed | 2.220446e-16 | 5.6237017e-11 | 6.6613381e-16 | 0 | 0 | 1.7763568e-15 | 8.8817842e-16 | {"max_FIP_error": 6.661338147750939e-16, "max_SIP_diagonal_FIP_error": 6.661338147750939e-16, "max_SIP_incremental_error": 0.0, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 1.7763568394002505e-15, "max_probability_sum_error": 8.881784197001252e-16, "min_sample_probability": 5.623701717665288e-11, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| E/DSD/r0 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 4.4408921e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 4.440892098500626e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| E/DSD/r1 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r2 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| E/DSD/r3 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 8.8817842e-16 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| E/DSD/r4 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 4.4408921e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 4.440892098500626e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r5 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r6 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 4.4408921e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 4.440892098500626e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r7 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r8 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 4.4408921e-16 | 8.8817842e-16 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 4.440892098500626e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| E/DSD/r9 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r10 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r11 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| E/DSD/r12 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 8.8817842e-16 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| E/DSD/r13 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 4.4408921e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 4.440892098500626e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r14 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r15 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 8.8817842e-16 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| E/DSD/r16 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r17 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 4.4408921e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 4.440892098500626e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| E/DSD/r18 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 8.8817842e-16 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| E/DSD/r19 | passed | 0 | 0.012345679 | 1.6653345e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 1.6653345369377348e-16, "max_SIP_diagonal_FIP_error": 1.6653345369377348e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.012345679012345663, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| E/Exact LP/QP/r0 | passed | 7.7715612e-16 | 2.5236945e-05 | 1.9428903e-15 | 0 | 0 | 5.884182e-15 | 4.4408921e-15 | {"max_FIP_error": 1.942890293094024e-15, "max_SIP_diagonal_FIP_error": 1.942890293094024e-15, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 5.88418203051333e-15, "max_probability_sum_error": 7.771561172376096e-16, "min_sample_probability": 2.5236945155576295e-05, "positive_samples_correct_size": true, "total_pair_identity_error": 4.440892098500626e-15} |
| E/Initial/r0 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r0 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 4.3655745685100517e-11, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r1 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 4.263256414560596e-14, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r2 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 1.2789769243681793e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r3 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_min_sample_probability": 6.821210263296954e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r4 | passed | 9.9920072e-16 | 1.0058284e-07 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 1.1787051334977138e-09, "min_sample_probability": 1.0058283805847157e-07, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r5 | passed | 8.8817842e-16 | 1.5716068e-09 | 2.7755576e-16 | 0 | 0 | 9.9920072e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 9.992007221626409e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 2.046363078989086e-12, "min_sample_probability": 1.5716068446636183e-09, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r6 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 1.7053025658242384e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r7 | passed | 8.8817842e-16 | 3.6379788e-12 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 3.469446951953614e-18, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 2.3314683517128287e-15, "maximum_scored_design_max_SIP_recompute_error": 3.469446951953614e-18, "maximum_scored_design_max_probability_sum_error": 2.886579864025407e-15, "maximum_scored_design_min_sample_probability": 1.1241008124329698e-15, "min_sample_probability": 3.6379788070917097e-12, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r8 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 9.992007221626409e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.5543122344752192e-15, "maximum_scored_design_min_sample_probability": 2.2737367544323186e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r9 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 4.3655745685100504e-11, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r10 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 1.4432899320127035e-15, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 2.1094237467877974e-15, "maximum_scored_design_min_sample_probability": 2.6645352591003725e-15, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r11 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 1.065814103640149e-14, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r12 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 6.821210263296954e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r13 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 2.2737367544323186e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r14 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 1.7053025658242384e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r15 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 1.7053025658242384e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r16 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 3.274180926382538e-11, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r17 | passed | 8.8817842e-16 | 1.9311905e-05 | 2.220446e-16 | 0 | 0 | 9.9920072e-16 | 4.4408921e-15 | {"max_FIP_error": 2.220446049250313e-16, "max_SIP_diagonal_FIP_error": 2.220446049250313e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 9.992007221626409e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 1.885928213596342e-08, "min_sample_probability": 1.9311904907226542e-05, "positive_samples_correct_size": true, "total_pair_identity_error": 4.440892098500626e-15} |
| E/Original GFS high precision/r18 | passed | 8.8817842e-16 | 4.4703484e-08 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 1.7053025658242384e-13, "min_sample_probability": 4.4703483581542916e-08, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS high precision/r19 | passed | 8.8817842e-16 | 4.5776367e-05 | 2.7755576e-16 | 0 | 0 | 1.110223e-15 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 2.7939677238464322e-09, "min_sample_probability": 4.5776367187499946e-05, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r0 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 4.3655745685100517e-11, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r1 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 4.263256414560596e-14, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r2 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 1.2789769243681793e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r3 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_min_sample_probability": 6.821210263296954e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r4 | passed | 9.9920072e-16 | 1.0058284e-07 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 1.1787051334977138e-09, "min_sample_probability": 1.0058283805847157e-07, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r5 | passed | 8.8817842e-16 | 1.5716068e-09 | 2.7755576e-16 | 0 | 0 | 9.9920072e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 9.992007221626409e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 2.046363078989086e-12, "min_sample_probability": 1.5716068446636183e-09, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r6 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 1.7053025658242384e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r7 | passed | 9.9920072e-16 | 5.2386895e-10 | 3.3306691e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 3.3306690738754696e-16, "max_SIP_diagonal_FIP_error": 3.3306690738754696e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 9.592326932761342e-14, "min_sample_probability": 5.23868948221206e-10, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r8 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 9.992007221626409e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.5543122344752192e-15, "maximum_scored_design_min_sample_probability": 2.2737367544323186e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r9 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 4.3655745685100504e-11, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r10 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 9.43689570931383e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.5543122344752192e-15, "maximum_scored_design_min_sample_probability": 2.6645352591003725e-15, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r11 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 1.065814103640149e-14, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r12 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 6.821210263296954e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r13 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 2.2737367544323186e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r14 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 1.7053025658242384e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r15 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 1.7053025658242384e-13, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r16 | passed | 8.8817842e-16 | 0.33333333 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 0.0, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 3.274180926382538e-11, "min_sample_probability": 0.33333333333333304, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r17 | passed | 8.8817842e-16 | 1.9311905e-05 | 2.220446e-16 | 0 | 0 | 9.9920072e-16 | 4.4408921e-15 | {"max_FIP_error": 2.220446049250313e-16, "max_SIP_diagonal_FIP_error": 2.220446049250313e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 9.992007221626409e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 1.885928213596342e-08, "min_sample_probability": 1.9311904907226542e-05, "positive_samples_correct_size": true, "total_pair_identity_error": 4.440892098500626e-15} |
| E/Original GFS/r18 | passed | 8.8817842e-16 | 4.4703484e-08 | 2.7755576e-16 | 0 | 0 | 8.8817842e-16 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16, "maximum_scored_design_min_sample_probability": 2.3980817331903356e-14, "min_sample_probability": 4.4703483581542916e-08, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Original GFS/r19 | passed | 8.8817842e-16 | 4.5776367e-05 | 2.7755576e-16 | 0 | 0 | 1.110223e-15 | 5.3290705e-15 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 3.885780586188048e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 1.1102230246251565e-15, "maximum_scored_design_min_sample_probability": 2.7939677238464322e-09, "min_sample_probability": 4.5776367187499946e-05, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| E/Sinha/r0 | NA — no saved support | NA | NA | NA | NA | NA | NA | NA | {} |
| U/CP(2)/r0 | passed | 1.110223e-16 | 3.8574761e-09 | 5.5511151e-16 | 0 | 0 | 8.8817842e-16 | 0 | {"max_FIP_error": 5.551115123125783e-16, "max_SIP_diagonal_FIP_error": 5.551115123125783e-16, "max_SIP_incremental_error": 0.0, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.857476139072209e-09, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r0 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r1 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r2 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r3 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r4 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r5 | passed | 4.4408921e-16 | 1.5173268e-80 | 3.0531133e-16 | 0 | 0 | 1.110223e-15 | 2.6645353e-15 | {"max_FIP_error": 3.0531133177191805e-16, "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16, "max_SIP_recompute_error": 3.0531133177191805e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 4.440892098500626e-16, "min_sample_probability": 1.5173268113177047e-80, "positive_samples_correct_size": true, "total_pair_identity_error": 3.552713678800501e-15} |
| U/DSD/r6 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r7 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r8 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 1.7763568e-15 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r9 | passed | 2.220446e-16 | 4.9294238e-81 | 2.7755576e-16 | 0 | 0 | 6.6613381e-16 | 8.8817842e-16 | {"max_FIP_error": 2.7755575615628914e-16, "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16, "max_SIP_recompute_error": 2.7755575615628914e-16, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 2.220446049250313e-16, "min_sample_probability": 4.929423778198965e-81, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/DSD/r10 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r11 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r12 | passed | 2.220446e-16 | 3.8398867e-45 | 3.0531133e-16 | 0 | 0 | 1.110223e-15 | 1.7763568e-15 | {"max_FIP_error": 3.0531133177191805e-16, "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16, "max_SIP_recompute_error": 3.0531133177191805e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 2.220446049250313e-16, "min_sample_probability": 3.839886735088039e-45, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/DSD/r13 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r14 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r15 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r16 | passed | 2.220446e-16 | 3.8398867e-45 | 3.0531133e-16 | 0 | 0 | 1.110223e-15 | 1.7763568e-15 | {"max_FIP_error": 3.0531133177191805e-16, "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16, "max_SIP_recompute_error": 3.0531133177191805e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 2.220446049250313e-16, "min_sample_probability": 3.839886735088039e-45, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/DSD/r17 | passed | 2.220446e-16 | 3.8398867e-45 | 3.0531133e-16 | 0 | 0 | 1.110223e-15 | 1.7763568e-15 | {"max_FIP_error": 3.0531133177191805e-16, "max_SIP_diagonal_FIP_error": 3.0531133177191805e-16, "max_SIP_recompute_error": 3.0531133177191805e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 2.220446049250313e-16, "min_sample_probability": 3.839886735088039e-45, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/DSD/r18 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/DSD/r19 | passed | 1.110223e-16 | 3.4971977e-75 | 4.9960036e-16 | 0 | 0 | 1.3322676e-15 | 0 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 4.996003610813204e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 1.1102230246251565e-16, "min_sample_probability": 3.4971977144972687e-75, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/Exact LP/QP/r0 | passed | 0 | 3.9348073e-05 | 6.6613381e-16 | 0 | 0 | 2.220446e-15 | 0 | {"max_FIP_error": 6.661338147750939e-16, "max_SIP_diagonal_FIP_error": 6.661338147750939e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 2.220446049250313e-15, "max_probability_sum_error": 0.0, "min_sample_probability": 3.934807277406816e-05, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/Initial/r0 | passed | 0 | 0.0087428972 | 2.220446e-16 | 0 | 0 | 6.6613381e-16 | 0 | {"max_FIP_error": 2.220446049250313e-16, "max_SIP_diagonal_FIP_error": 2.220446049250313e-16, "max_SIP_recompute_error": 0.0, "max_fixed_size_row_identity_error": 6.661338147750939e-16, "max_probability_sum_error": 0.0, "min_sample_probability": 0.00874289719330934, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/Original GFS high precision/r0 | passed | 8.8817842e-16 | 8.0148238e-14 | 9.9920072e-16 | 0 | 0 | 2.6645353e-15 | 4.4408921e-15 | {"max_FIP_error": 9.992007221626409e-16, "max_SIP_diagonal_FIP_error": 9.992007221626409e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 2.6645352591003757e-15, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 9.71445146547012e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_min_sample_probability": 2.00370595034878e-14, "min_sample_probability": 8.01482380139512e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| U/Original GFS high precision/r1 | passed | 7.7715612e-16 | 4.0599046e-14 | 9.4368957e-16 | 0 | 0 | 2.6645353e-15 | 4.4408921e-15 | {"max_FIP_error": 9.43689570931383e-16, "max_SIP_diagonal_FIP_error": 9.43689570931383e-16, "max_SIP_recompute_error": 2.220446049250313e-16, "max_fixed_size_row_identity_error": 2.6645352591003757e-15, "max_probability_sum_error": 7.771561172376096e-16, "maximum_scored_design_max_FIP_error": 1.5543122344752192e-15, "maximum_scored_design_max_SIP_recompute_error": 2.220446049250313e-16, "maximum_scored_design_max_probability_sum_error": 1.4432899320127035e-15, "maximum_scored_design_min_sample_probability": 1.2736955598069995e-14, "min_sample_probability": 4.059904596884811e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| U/Original GFS high precision/r2 | passed | 2.8865799e-15 | 7.1645375e-15 | 3.1641356e-15 | 0 | 0 | 9.3258734e-15 | 1.687539e-14 | {"max_FIP_error": 3.164135620181696e-15, "max_SIP_diagonal_FIP_error": 3.164135620181696e-15, "max_SIP_recompute_error": 2.220446049250313e-16, "max_fixed_size_row_identity_error": 9.325873406851315e-15, "max_probability_sum_error": 2.886579864025407e-15, "maximum_scored_design_max_FIP_error": 3.6637359812630166e-15, "maximum_scored_design_max_SIP_recompute_error": 2.220446049250313e-16, "maximum_scored_design_max_probability_sum_error": 3.6637359812630166e-15, "maximum_scored_design_min_sample_probability": 1.343350785733945e-15, "min_sample_probability": 7.164537523914373e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-14} |
| U/Original GFS high precision/r3 | passed | 5.5511151e-16 | 6.2136858e-15 | 3.8857806e-16 | 0 | 0 | 1.3322676e-15 | 4.4408921e-15 | {"max_FIP_error": 3.885780586188048e-16, "max_SIP_diagonal_FIP_error": 3.885780586188048e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 5.551115123125783e-16, "maximum_scored_design_max_FIP_error": 5.551115123125783e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_min_sample_probability": 1.5534214613744454e-15, "min_sample_probability": 6.213685845497782e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 3.552713678800501e-15} |
| U/Original GFS high precision/r4 | passed | 6.6613381e-16 | 3.5621439e-14 | 5.5511151e-16 | 0 | 0 | 1.4432899e-15 | 3.5527137e-15 | {"max_FIP_error": 5.551115123125783e-16, "max_SIP_diagonal_FIP_error": 5.551115123125783e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 1.4432899320127035e-15, "max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_max_FIP_error": 7.216449660063518e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_min_sample_probability": 2.6716079337983736e-14, "min_sample_probability": 3.562143911731165e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 3.552713678800501e-15} |
| U/Original GFS high precision/r5 | passed | 1.8873791e-14 | 1.061413e-15 | 1.0713652e-14 | 0 | 0 | 3.1752379e-14 | 1.1457502e-13 | {"max_FIP_error": 1.071365218763276e-14, "max_SIP_diagonal_FIP_error": 1.071365218763276e-14, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 3.175237850427948e-14, "max_probability_sum_error": 1.887379141862766e-14, "maximum_scored_design_max_FIP_error": 1.3600232051658168e-14, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 2.4424906541753444e-14, "maximum_scored_design_min_sample_probability": 1.0614129665058324e-15, "min_sample_probability": 1.0614129665058328e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 1.1191048088221578e-13} |
| U/Original GFS high precision/r6 | passed | 2.220446e-16 | 2.9825692e-13 | 5.5511151e-16 | 0 | 0 | 1.3322676e-15 | 2.6645353e-15 | {"max_FIP_error": 5.551115123125783e-16, "max_SIP_diagonal_FIP_error": 5.551115123125783e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 2.220446049250313e-16, "maximum_scored_design_max_FIP_error": 6.106226635438361e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 4.440892098500626e-16, "maximum_scored_design_min_sample_probability": 1.258271383713301e-13, "min_sample_probability": 2.9825692058389357e-13, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| U/Original GFS high precision/r7 | passed | 1.6431301e-14 | 1.1650661e-15 | 1.0547119e-14 | 0 | 0 | 3.1530334e-14 | 9.8587805e-14 | {"max_FIP_error": 1.0547118733938987e-14, "max_SIP_diagonal_FIP_error": 1.0547118733938987e-14, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 3.1530333899354446e-14, "max_probability_sum_error": 1.6431300764452317e-14, "maximum_scored_design_max_FIP_error": 1.149080830487037e-14, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 1.84297022087776e-14, "maximum_scored_design_min_sample_probability": 1.1650660960308343e-15, "min_sample_probability": 1.1650660960308343e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 9.769962616701378e-14} |
| U/Original GFS high precision/r8 | passed | 1.5987212e-14 | 1.4152173e-15 | 9.1038288e-15 | 0 | 0 | 2.7311486e-14 | 9.6811448e-14 | {"max_FIP_error": 9.103828801926284e-15, "max_SIP_diagonal_FIP_error": 9.103828801926284e-15, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 2.731148640577885e-14, "max_probability_sum_error": 1.5987211554602254e-14, "maximum_scored_design_max_FIP_error": 9.71445146547012e-15, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 1.765254609153999e-14, "maximum_scored_design_min_sample_probability": 1.0614129665058328e-15, "min_sample_probability": 1.4152172886744437e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 9.592326932761353e-14} |
| U/Original GFS high precision/r9 | passed | 0 | 5.6664043e-06 | 3.3306691e-16 | 0 | 0 | 8.8817842e-16 | 8.8817842e-16 | {"max_FIP_error": 3.3306690738754696e-16, "max_SIP_diagonal_FIP_error": 3.3306690738754696e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 0.0, "maximum_scored_design_max_FIP_error": 4.440892098500626e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 4.440892098500626e-16, "maximum_scored_design_min_sample_probability": 1.846615403841436e-10, "min_sample_probability": 5.666404297974395e-06, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/Original GFS high precision/r10 | passed | 2.220446e-16 | 1.54442e-08 | 4.4408921e-16 | 0 | 0 | 8.8817842e-16 | 1.7763568e-15 | {"max_FIP_error": 4.440892098500626e-16, "max_SIP_diagonal_FIP_error": 4.440892098500626e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 2.220446049250313e-16, "maximum_scored_design_max_FIP_error": 5.551115123125783e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_min_sample_probability": 1.131166987695952e-11, "min_sample_probability": 1.5444199938675397e-08, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/Original GFS high precision/r11 | passed | 1.010303e-14 | 1.9183606e-15 | 6.4392935e-15 | 0 | 0 | 1.976197e-14 | 6.0396133e-14 | {"max_FIP_error": 6.439293542825908e-15, "max_SIP_diagonal_FIP_error": 6.439293542825908e-15, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 1.9761969838327786e-14, "max_probability_sum_error": 1.0103029524088925e-14, "maximum_scored_design_max_FIP_error": 8.43769498715119e-15, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 1.2101430968414206e-14, "maximum_scored_design_min_sample_probability": 1.0790778603486846e-15, "min_sample_probability": 1.9183606406198835e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 6.128431095930864e-14} |
| U/Original GFS high precision/r12 | passed | 2.220446e-16 | 1.7245773e-13 | 3.8857806e-16 | 0 | 0 | 8.8817842e-16 | 8.8817842e-16 | {"max_FIP_error": 3.885780586188048e-16, "max_SIP_diagonal_FIP_error": 3.885780586188048e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 2.220446049250313e-16, "maximum_scored_design_max_FIP_error": 1.942890293094024e-15, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 2.1094237467877974e-15, "maximum_scored_design_min_sample_probability": 2.694651983531908e-15, "min_sample_probability": 1.724577269460421e-13, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| U/Original GFS high precision/r13 | passed | 7.8825835e-15 | 1.0485595e-14 | 6.9388939e-15 | 0 | 0 | 2.1316282e-14 | 4.7961635e-14 | {"max_FIP_error": 6.938893903907228e-15, "max_SIP_diagonal_FIP_error": 6.938893903907228e-15, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 2.1316282072803006e-14, "max_probability_sum_error": 7.882583474838611e-15, "maximum_scored_design_max_FIP_error": 7.993605777301127e-15, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 9.325873406851315e-15, "maximum_scored_design_min_sample_probability": 1.1650660960308343e-15, "min_sample_probability": 1.0485594864277509e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 4.796163466380676e-14} |
| U/Original GFS high precision/r14 | passed | 1.110223e-16 | 6.7107807e-13 | 5.5511151e-16 | 0 | 0 | 1.110223e-15 | 8.8817842e-16 | {"max_FIP_error": 5.551115123125783e-16, "max_SIP_diagonal_FIP_error": 5.551115123125783e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 1.1102230246251565e-16, "maximum_scored_design_max_FIP_error": 8.326672684688674e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 3.3306690738754696e-16, "maximum_scored_design_min_sample_probability": 5.592317260948003e-14, "min_sample_probability": 6.710780713137606e-13, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/Original GFS high precision/r15 | passed | 4.4408921e-16 | 2.4854743e-14 | 6.1062266e-16 | 0 | 0 | 1.110223e-15 | 2.6645353e-15 | {"max_FIP_error": 6.106226635438361e-16, "max_SIP_diagonal_FIP_error": 6.106226635438361e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 4.440892098500626e-16, "maximum_scored_design_max_FIP_error": 4.440892098500626e-16, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_min_sample_probability": 6.213685845497782e-15, "min_sample_probability": 2.4854743381991127e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 2.6645352591003757e-15} |
| U/Original GFS high precision/r16 | passed | 1.6986412e-14 | 1.1368063e-15 | 1.3267165e-14 | 0 | 0 | 3.9301895e-14 | 1.030287e-13 | {"max_FIP_error": 1.326716514427062e-14, "max_SIP_diagonal_FIP_error": 1.326716514427062e-14, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 3.930189507173054e-14, "max_probability_sum_error": 1.6986412276764895e-14, "maximum_scored_design_max_FIP_error": 1.4377388168895777e-14, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 1.7985612998927536e-14, "maximum_scored_design_min_sample_probability": 1.136806305552524e-15, "min_sample_probability": 1.136806305552524e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 1.021405182655144e-13} |
| U/Original GFS high precision/r17 | passed | 3.3306691e-15 | 2.4882808e-15 | 3.4416914e-15 | 0 | 0 | 9.7699626e-15 | 1.9539925e-14 | {"max_FIP_error": 3.4416913763379853e-15, "max_SIP_diagonal_FIP_error": 3.4416913763379853e-15, "max_SIP_recompute_error": 2.220446049250313e-16, "max_fixed_size_row_identity_error": 9.769962616701378e-15, "max_probability_sum_error": 3.3306690738754696e-15, "maximum_scored_design_max_FIP_error": 4.163336342344337e-15, "maximum_scored_design_max_SIP_recompute_error": 2.220446049250313e-16, "maximum_scored_design_max_probability_sum_error": 5.10702591327572e-15, "maximum_scored_design_min_sample_probability": 2.4882808125189785e-15, "min_sample_probability": 2.4882808125189805e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 1.9539925233402755e-14} |
| U/Original GFS high precision/r18 | passed | 1.110223e-16 | 7.0638685e-10 | 3.8857806e-16 | 0 | 0 | 8.8817842e-16 | 1.7763568e-15 | {"max_FIP_error": 3.885780586188048e-16, "max_SIP_diagonal_FIP_error": 3.885780586188048e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 1.1102230246251565e-16, "maximum_scored_design_max_FIP_error": 4.996003610813204e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 3.3306690738754696e-16, "maximum_scored_design_min_sample_probability": 1.0778607934127632e-14, "min_sample_probability": 7.063868495709885e-10, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/Original GFS high precision/r19 | passed | 1.2101431e-14 | 1.1368063e-15 | 9.4924069e-15 | 0 | 0 | 2.8976821e-14 | 7.1942452e-14 | {"max_FIP_error": 9.492406860545088e-15, "max_SIP_diagonal_FIP_error": 9.492406860545088e-15, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 2.8976820942716586e-14, "max_probability_sum_error": 1.2101430968414206e-14, "maximum_scored_design_max_FIP_error": 1.0824674490095276e-14, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 1.4654943925052066e-14, "maximum_scored_design_min_sample_probability": 1.1368063055525238e-15, "min_sample_probability": 1.1368063055525238e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 7.283063041541027e-14} |
| U/Original GFS/r0 | passed | 8.8817842e-16 | 8.0148238e-14 | 1.0547119e-15 | 0 | 0 | 2.553513e-15 | 5.3290705e-15 | {"max_FIP_error": 1.0547118733938987e-15, "max_SIP_diagonal_FIP_error": 1.0547118733938987e-15, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 2.55351295663786e-15, "max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_max_FIP_error": 9.71445146547012e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 8.881784197001252e-16, "maximum_scored_design_min_sample_probability": 2.00370595034878e-14, "min_sample_probability": 8.01482380139512e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 5.329070518200751e-15} |
| U/Original GFS/r1 | passed | 0 | 3.8210867e-14 | 4.9960036e-16 | 0 | 0 | 8.8817842e-16 | 8.8817842e-16 | {"max_FIP_error": 4.996003610813204e-16, "max_SIP_diagonal_FIP_error": 4.996003610813204e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 0.0, "maximum_scored_design_max_FIP_error": 1.5543122344752192e-15, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 1.4432899320127035e-15, "maximum_scored_design_min_sample_probability": 1.2736955598069995e-14, "min_sample_probability": 3.8210866794209986e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/Original GFS/r2 | passed | 7.7715612e-16 | 3.1842389e-15 | 8.8817842e-16 | 0 | 0 | 2.6645353e-15 | 4.4408921e-15 | {"max_FIP_error": 8.881784197001252e-16, "max_SIP_diagonal_FIP_error": 8.881784197001252e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 2.6645352591003757e-15, "max_probability_sum_error": 7.771561172376096e-16, "maximum_scored_design_max_FIP_error": 1.7208456881689926e-15, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 2.220446049250313e-15, "maximum_scored_design_min_sample_probability": 1.791134380978593e-15, "min_sample_probability": 3.1842388995174988e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 4.440892098500626e-15} |
| U/Original GFS/r3 | passed | 5.5511151e-16 | 6.2136858e-15 | 3.8857806e-16 | 0 | 0 | 1.3322676e-15 | 4.4408921e-15 | {"max_FIP_error": 3.885780586188048e-16, "max_SIP_diagonal_FIP_error": 3.885780586188048e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 1.3322676295501878e-15, "max_probability_sum_error": 5.551115123125783e-16, "maximum_scored_design_max_FIP_error": 5.551115123125783e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_min_sample_probability": 1.5534214613744454e-15, "min_sample_probability": 6.213685845497782e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 3.552713678800501e-15} |
| U/Original GFS/r4 | passed | 6.6613381e-16 | 3.5621439e-14 | 5.5511151e-16 | 0 | 0 | 1.4432899e-15 | 3.5527137e-15 | {"max_FIP_error": 5.551115123125783e-16, "max_SIP_diagonal_FIP_error": 5.551115123125783e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 1.4432899320127035e-15, "max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_max_FIP_error": 7.216449660063518e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_min_sample_probability": 2.6716079337983736e-14, "min_sample_probability": 3.562143911731165e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 3.552713678800501e-15} |
| U/Original GFS/r5 | passed | 2.1871394e-14 | 1.061413e-15 | 1.2712054e-14 | 0 | 0 | 3.7081449e-14 | 1.3145041e-13 | {"max_FIP_error": 1.2712053631958042e-14, "max_SIP_diagonal_FIP_error": 1.2712053631958042e-14, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 3.708144902248023e-14, "max_probability_sum_error": 2.1871393585115584e-14, "maximum_scored_design_max_FIP_error": 1.3378187446733136e-14, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 2.4646951146678475e-14, "maximum_scored_design_min_sample_probability": 1.0614129665058324e-15, "min_sample_probability": 1.0614129665058328e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 1.305622276959184e-13} |
| U/Original GFS/r6 | passed | 2.220446e-16 | 1.2582714e-13 | 6.1062266e-16 | 0 | 0 | 1.110223e-15 | 1.7763568e-15 | {"max_FIP_error": 6.106226635438361e-16, "max_SIP_diagonal_FIP_error": 6.106226635438361e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 2.220446049250313e-16, "maximum_scored_design_max_FIP_error": 6.106226635438361e-16, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 4.440892098500626e-16, "maximum_scored_design_min_sample_probability": 3.145678459283253e-14, "min_sample_probability": 1.258271383713301e-13, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| U/Original GFS/r7 | passed | 1.6320278e-14 | 1.1650661e-15 | 1.0547119e-14 | 0 | 0 | 3.1530334e-14 | 9.8587805e-14 | {"max_FIP_error": 1.0547118733938987e-14, "max_SIP_diagonal_FIP_error": 1.0547118733938987e-14, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 3.1530333899354446e-14, "max_probability_sum_error": 1.63202784619898e-14, "maximum_scored_design_max_FIP_error": 1.149080830487037e-14, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 1.84297022087776e-14, "maximum_scored_design_min_sample_probability": 1.1650660960308343e-15, "min_sample_probability": 1.1650660960308343e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 9.85878045867139e-14} |
| U/Original GFS/r8 | passed | 8.9928065e-15 | 1.061413e-15 | 5.3845817e-15 | 0 | 0 | 1.6431301e-14 | 5.4178884e-14 | {"max_FIP_error": 5.384581669432009e-15, "max_SIP_diagonal_FIP_error": 5.384581669432009e-15, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 1.6431300764452317e-14, "max_probability_sum_error": 8.992806499463768e-15, "maximum_scored_design_max_FIP_error": 8.215650382226158e-15, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 1.2434497875801753e-14, "maximum_scored_design_min_sample_probability": 1.0614129665058328e-15, "min_sample_probability": 1.0614129665058328e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 5.5067062021407764e-14} |
| U/Original GFS/r9 | passed | 0 | 5.6664043e-06 | 3.3306691e-16 | 0 | 0 | 8.8817842e-16 | 8.8817842e-16 | {"max_FIP_error": 3.3306690738754696e-16, "max_SIP_diagonal_FIP_error": 3.3306690738754696e-16, "max_SIP_recompute_error": 5.551115123125783e-17, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 0.0, "maximum_scored_design_max_FIP_error": 4.440892098500626e-16, "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17, "maximum_scored_design_max_probability_sum_error": 4.440892098500626e-16, "maximum_scored_design_min_sample_probability": 1.846615403841436e-10, "min_sample_probability": 5.666404297974395e-06, "positive_samples_correct_size": true, "total_pair_identity_error": 0.0} |
| U/Original GFS/r10 | passed | 2.220446e-16 | 1.54442e-08 | 4.4408921e-16 | 0 | 0 | 8.8817842e-16 | 1.7763568e-15 | {"max_FIP_error": 4.440892098500626e-16, "max_SIP_diagonal_FIP_error": 4.440892098500626e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 2.220446049250313e-16, "maximum_scored_design_max_FIP_error": 5.551115123125783e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_min_sample_probability": 1.131166987695952e-11, "min_sample_probability": 1.5444199938675397e-08, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/Original GFS/r11 | passed | 1.010303e-14 | 1.9183606e-15 | 6.3837824e-15 | 0 | 0 | 1.976197e-14 | 5.9507954e-14 | {"max_FIP_error": 6.38378239159465e-15, "max_SIP_diagonal_FIP_error": 6.38378239159465e-15, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 1.9761969838327786e-14, "max_probability_sum_error": 1.0103029524088925e-14, "maximum_scored_design_max_FIP_error": 8.43769498715119e-15, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 1.2101430968414206e-14, "maximum_scored_design_min_sample_probability": 1.0790778603486846e-15, "min_sample_probability": 1.9183606406198835e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 5.950795411990839e-14} |
| U/Original GFS/r12 | passed | 2.220446e-16 | 1.7245773e-13 | 3.8857806e-16 | 0 | 0 | 8.8817842e-16 | 8.8817842e-16 | {"max_FIP_error": 3.885780586188048e-16, "max_SIP_diagonal_FIP_error": 3.885780586188048e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 2.220446049250313e-16, "maximum_scored_design_max_FIP_error": 1.942890293094024e-15, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 2.1094237467877974e-15, "maximum_scored_design_min_sample_probability": 2.694651983531908e-15, "min_sample_probability": 1.724577269460421e-13, "positive_samples_correct_size": true, "total_pair_identity_error": 8.881784197001252e-16} |
| U/Original GFS/r13 | passed | 1.3100632e-14 | 2.0712286e-15 | 1.1823875e-14 | 0 | 0 | 3.5749181e-14 | 7.9047879e-14 | {"max_FIP_error": 1.1823875212257917e-14, "max_SIP_diagonal_FIP_error": 1.1823875212257917e-14, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 3.574918139293004e-14, "max_probability_sum_error": 1.3100631690576847e-14, "maximum_scored_design_max_FIP_error": 1.27675647831893e-14, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 1.3988810110276972e-14, "maximum_scored_design_min_sample_probability": 1.1650660960308343e-15, "min_sample_probability": 2.0712286151659264e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 7.815970093361102e-14} |
| U/Original GFS/r14 | passed | 1.110223e-16 | 6.7107807e-13 | 5.5511151e-16 | 0 | 0 | 1.110223e-15 | 8.8817842e-16 | {"max_FIP_error": 5.551115123125783e-16, "max_SIP_diagonal_FIP_error": 5.551115123125783e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 1.1102230246251565e-16, "maximum_scored_design_max_FIP_error": 8.326672684688674e-16, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 3.3306690738754696e-16, "maximum_scored_design_min_sample_probability": 5.592317260948003e-14, "min_sample_probability": 6.710780713137606e-13, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/Original GFS/r15 | passed | 4.4408921e-16 | 2.4854743e-14 | 6.1062266e-16 | 0 | 0 | 1.110223e-15 | 2.6645353e-15 | {"max_FIP_error": 6.106226635438361e-16, "max_SIP_diagonal_FIP_error": 6.106226635438361e-16, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 1.1102230246251565e-15, "max_probability_sum_error": 4.440892098500626e-16, "maximum_scored_design_max_FIP_error": 4.440892098500626e-16, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 6.661338147750939e-16, "maximum_scored_design_min_sample_probability": 6.213685845497782e-15, "min_sample_probability": 2.4854743381991127e-14, "positive_samples_correct_size": true, "total_pair_identity_error": 2.6645352591003757e-15} |
| U/Original GFS/r16 | passed | 1.6986412e-14 | 1.1368063e-15 | 1.3267165e-14 | 0 | 0 | 3.9301895e-14 | 1.030287e-13 | {"max_FIP_error": 1.326716514427062e-14, "max_SIP_diagonal_FIP_error": 1.326716514427062e-14, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 3.930189507173054e-14, "max_probability_sum_error": 1.6986412276764895e-14, "maximum_scored_design_max_FIP_error": 1.4377388168895777e-14, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 1.7985612998927536e-14, "maximum_scored_design_min_sample_probability": 1.136806305552524e-15, "min_sample_probability": 1.136806305552524e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 1.0302869668521453e-13} |
| U/Original GFS/r17 | passed | 2.6645353e-15 | 2.4882808e-15 | 2.942091e-15 | 0 | 0 | 8.1046281e-15 | 1.5987212e-14 | {"max_FIP_error": 2.942091015256665e-15, "max_SIP_diagonal_FIP_error": 2.942091015256665e-15, "max_SIP_recompute_error": 1.6653345369377348e-16, "max_fixed_size_row_identity_error": 8.104628079763643e-15, "max_probability_sum_error": 2.6645352591003757e-15, "maximum_scored_design_max_FIP_error": 4.163336342344337e-15, "maximum_scored_design_max_SIP_recompute_error": 1.6653345369377348e-16, "maximum_scored_design_max_probability_sum_error": 5.10702591327572e-15, "maximum_scored_design_min_sample_probability": 2.4882808125189785e-15, "min_sample_probability": 2.4882808125189805e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 1.5987211554602254e-14} |
| U/Original GFS/r18 | passed | 1.110223e-16 | 7.0638685e-10 | 3.8857806e-16 | 0 | 0 | 8.8817842e-16 | 1.7763568e-15 | {"max_FIP_error": 3.885780586188048e-16, "max_SIP_diagonal_FIP_error": 3.885780586188048e-16, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 8.881784197001252e-16, "max_probability_sum_error": 1.1102230246251565e-16, "maximum_scored_design_max_FIP_error": 2.6645352591003757e-15, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 2.6645352591003757e-15, "maximum_scored_design_min_sample_probability": 1.5968308050559454e-15, "min_sample_probability": 7.063868495709885e-10, "positive_samples_correct_size": true, "total_pair_identity_error": 1.7763568394002505e-15} |
| U/Original GFS/r19 | passed | 1.2101431e-14 | 1.1368063e-15 | 9.4924069e-15 | 0 | 0 | 2.8976821e-14 | 7.1942452e-14 | {"max_FIP_error": 9.492406860545088e-15, "max_SIP_diagonal_FIP_error": 9.492406860545088e-15, "max_SIP_recompute_error": 1.1102230246251565e-16, "max_fixed_size_row_identity_error": 2.8976820942716586e-14, "max_probability_sum_error": 1.2101430968414206e-14, "maximum_scored_design_max_FIP_error": 1.0824674490095276e-14, "maximum_scored_design_max_SIP_recompute_error": 1.1102230246251565e-16, "maximum_scored_design_max_probability_sum_error": 1.4654943925052066e-14, "maximum_scored_design_min_sample_probability": 1.1368063055525238e-15, "min_sample_probability": 1.1368063055525238e-15, "positive_samples_correct_size": true, "total_pair_identity_error": 7.283063041541027e-14} |
| U/Sinha/r0 | NA — no saved support | NA | NA | NA | NA | NA | NA | NA | {} |

The artifact audit independently accumulates float64 indicator moments from each saved support without renormalizing its probabilities. It checks probability >=-1e-12, unit total mass within 1e-10, prescribed FIP within 1e-10, distinct in-range units and sample size four, complete saved SIP discrepancy <=1e-10, and both fixed-size SIP identities. Reported incremental/full-SIP discrepancies are retained in the reported_validation column. A missing saved support is explicitly unaudited. Target matrices are also checked directly for sum_j!=i pi_ij=3*pi_i and sum_i<j pi_ij=6 in target_diagnostics.

## Convergence and figure captions

Curves show incumbent RMSE medians and interquartile bands. Within each observed run, the last recorded checkpoint is carried forward to the union checkpoint grid; this is LOCF, not linear interpolation. Grids are restricted to the intersection of observed run ranges, so no run is carried forward after its final checkpoint. The number of represented runs is saved in convergence CSV files. This common range may end before the candidate budget; supplementary full-trace figures show every recorded GFS run through its actual endpoint.

Figure A. Target construction: entropy, standardized pair-dependence distance, and SIP RMSE from the initial design over accepted random exchanges. Dashed lines mark the frozen target snapshot. The target walk is independent of the later reconstruction objective.

[PDF](../results/N12_n4/figures/figure_A.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_A.png)

Figure B. Initial and target SIP matrices and their difference for equal and unequal FIP. Initial/target color scales are shared across cases, as are symmetric difference scales. Diagonals contain FIP.

[PDF](../results/N12_n4/figures/figure_B.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_B.png)

Figure C. Reconstruction RMSE against scored candidate evaluations, with median and interquartile bands over Original GFS repetitions, the high-precision sensitivity, and restricted DSD projection/order fitting. GFS counts scored sampling designs; DSD counts scored projection/order kernels. Curves show incumbent RMSE medians and interquartile bands. Within each observed run, the last recorded checkpoint is carried forward to the union checkpoint grid; this is LOCF, not linear interpolation. Grids are restricted to the intersection of observed run ranges, so no run is carried forward after its final checkpoint. The number of represented runs is saved in convergence CSV files. This common range may end before the candidate budget; supplementary full-trace figures show every recorded GFS run through its actual endpoint.

[PDF](../results/N12_n4/figures/figure_C.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_C.png)

Figure D. Reconstruction RMSE against recorded elapsed runtime, with median and interquartile bands. Runtime is measured under the saved parallel execution settings. Curves show incumbent RMSE medians and interquartile bands. Within each observed run, the last recorded checkpoint is carried forward to the union checkpoint grid; this is LOCF, not linear interpolation. Grids are restricted to the intersection of observed run ranges, so no run is carried forward after its final checkpoint. The number of represented runs is saved in convergence CSV files. This common range may end before the candidate budget; supplementary full-trace figures show every recorded GFS run through its actual endpoint.

[PDF](../results/N12_n4/figures/figure_D.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_D.png)

Figure E. Final stochastic RMSE distributions, individual runs, and deterministic reference levels. Exact LP/QP is an enumerated numerical oracle. Values below 1e-16 are floored only for log-scale display; tabulated values retain full precision.

[PDF](../results/N12_n4/figures/figure_E.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_E.png)

Figure F. Target versus recovered SIP for all 66 unordered pairs. Representative runs use the lowest saved replication, with seed/path tie breaks, independently of performance. Best runs minimize final RMSE with the same tie breaks. Panels and legends identify selection and replication; figure_F_selections.csv records exact choices. Deterministic methods appear in both selection panels when available.

[PDF](../results/N12_n4/figures/figure_F.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_F.png)

Supplementary Figure C full traces. Observed incumbent RMSE traces for every Original GFS and high-precision repetition, with method-specific colors and alpha=0.25. Each line ends at that run's final recorded checkpoint; no interpolation or extrapolation is used. These traces reveal budget-ending improvement or stagnation after the common-range summary has stopped. Recorded plateaus do not establish local optimality. Horizontal axis: candidate evaluations.

[PDF](../results/N12_n4/figures/figure_C_full_traces.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_C_full_traces.png)

Supplementary Figure D full traces. Observed incumbent RMSE traces for every Original GFS and high-precision repetition, with method-specific colors and alpha=0.25. Each line ends at that run's final recorded checkpoint; no interpolation or extrapolation is used. These traces reveal budget-ending improvement or stagnation after the common-range summary has stopped. Recorded plateaus do not establish local optimality. Horizontal axis: elapsed runtime.

[PDF](../results/N12_n4/figures/figure_D_full_traces.pdf) · [300-dpi PNG](../results/N12_n4/figures/figure_D_full_traces.png)

## Machine, timing pilot and provenance

Machine and actual worker settings:
```json
{
  "logical_CPUs_system": 16,
  "logical_CPUs_available": 16,
  "physical_cores_available": 8,
  "CPU_affinity": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15
  ],
  "workers": 12,
  "free_logical_CPU_slots": 4,
  "RAM_total_bytes": 32822620160,
  "RAM_available_bytes": 22016524288,
  "detected_UTC": "2026-09-18T09:57:20.121850+00:00",
  "library_threads_per_worker": 1,
  "numerical_thread_environment": {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
  },
  "runtime_comparison": "runtimes measured under documented concurrent independent-run load; not isolated hardware benchmarks"
}
```
Timing benchmark and projected runtime:
```json
{
  "pilot_candidates": 500,
  "original_pilots": [
    {
      "method": "Original GFS",
      "seed": 6618800300083929852,
      "status": "budget_exhausted",
      "failure_reason": "scored_candidate_budget",
      "final_D2": 0.19182290845436584,
      "final_RMSE": 0.053911109280500136,
      "final_Dinf": 0.1343001451566045,
      "recovery_percent": 0.0,
      "runtime_seconds": 0.20458469599907403,
      "candidate_evaluations": 500,
      "validation": {
        "max_probability_sum_error": 8.881784197001252e-16,
        "max_FIP_error": 2.7755575615628914e-16,
        "min_sample_probability": 0.33333333333333304,
        "max_SIP_recompute_error": 0.0,
        "max_fixed_size_row_identity_error": 8.881784197001252e-16,
        "total_pair_identity_error": 5.329070518200751e-15,
        "max_SIP_diagonal_FIP_error": 2.7755575615628914e-16,
        "positive_samples_correct_size": true,
        "maximum_scored_design_max_FIP_error": 3.3306690738754696e-16,
        "maximum_scored_design_max_probability_sum_error": 9.992007221626409e-16,
        "maximum_scored_design_min_sample_probability": 3.7252902984619108e-09,
        "maximum_scored_design_max_SIP_recompute_error": 0.0
      },
      "trace": [
        {
          "candidates": 0,
          "elapsed_seconds": 0.002310497000507894,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 50,
          "elapsed_seconds": 0.012490208000599523,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 100,
          "elapsed_seconds": 0.022150385000713868,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 150,
          "elapsed_seconds": 0.036470066999754636,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 200,
          "elapsed_seconds": 0.05006105999927968,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 250,
          "elapsed_seconds": 0.0716204939999443,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 300,
          "elapsed_seconds": 0.09313583999937691,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 350,
          "elapsed_seconds": 0.12215741600084584,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 400,
          "elapsed_seconds": 0.1463432410000678,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 450,
          "elapsed_seconds": 0.17552360000081535,
          "D2": 0.1918229084543658,
          "RMSE": 0.05391110928050013,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        },
        {
          "candidates": 500,
          "elapsed_seconds": 0.20457786999941163,
          "D2": 0.19182290845436584,
          "RMSE": 0.053911109280500136,
          "Dinf": 0.1343001451566045,
          "recovery_percent": 0.0
        }
      ],
      "details": {
        "score": "D2",
        "score_rounding_decimals": 9,
        "same_original_run_bytecode": true,
        "search_options": {
          "max_iterations": 1000,
          "max_open_set_size": 50,
          "top_k": 1,
          "num_new_order_nodes": 2,
          "num_new_exchange_nodes": 10,
          "num_clusters_range": [
            1,
            1
          ],
          "num_zones_range": [
            1,
            1
          ],
          "num_changes_range": [
            1,
            1
          ],
          "num_zone_changes": 0,
          "pull_strategy": "default",
          "exchange_coef": 0.75,
          "num_explore": 1,
          "n_jobs": 1,
          "backtrack_depth": 0,
          "stagnation_limit": 100,
          "output_mode": "silent"
        },
        "last_improvement_candidate": 0,
        "no_improvement_stop_candidates": 2000,
        "iterations": 88,
        "profile": {
          "criterion_seconds": 0.03917701596947154,
          "SIP_computation_seconds": 0.0031790510165592423,
          "support_extraction_seconds": 0.024952183006462292,
          "candidate_generation_seconds": 0.09569159698548901,
          "design_copy_seconds": 0.04678679497192206,
          "design_update_seconds": 0.04701931796444114,
          "validation_seconds": 0.005274132987324265
        },
        "evaluations_per_second": 2443.9755748018565,
        "average_candidate_criterion_seconds": 7.835403193894308e-05,
        "peak_process_RSS_MB": 256.46875,
        "memory_measurement": "process-lifetime RSS high water; not incremental method memory",
        "high_precision_implementation": "unmodified run and builtin round",
        "initial_D2": 0.1918229084543658,
        "initial_RMSE": 0.05391110928050013
      },
      "case": "E"
    },
    {
      "method": "Original GFS",
      "seed": 5694361346634055340,
      "status": "budget_exhausted",
      "failure_reason": "scored_candidate_budget",
      "final_D2": 0.00505066761448491,
      "final_RMSE": 0.008747872134761402,
      "final_Dinf": 0.024320468463903983,
      "recovery_percent": 43.82263989063987,
      "runtime_seconds": 0.24734037300004275,
      "candidate_evaluations": 500,
      "validation": {
        "max_probability_sum_error": 0.0,
        "max_FIP_error": 2.220446049250313e-16,
        "min_sample_probability": 1.1725138834086666e-09,
        "max_SIP_recompute_error": 5.551115123125783e-17,
        "max_fixed_size_row_identity_error": 6.661338147750939e-16,
        "total_pair_identity_error": 0.0,
        "max_SIP_diagonal_FIP_error": 2.220446049250313e-16,
        "positive_samples_correct_size": true,
        "maximum_scored_design_max_FIP_error": 3.885780586188048e-16,
        "maximum_scored_design_max_probability_sum_error": 2.220446049250313e-16,
        "maximum_scored_design_min_sample_probability": 7.328211771304166e-11,
        "maximum_scored_design_max_SIP_recompute_error": 5.551115123125783e-17
      },
      "trace": [
        {
          "candidates": 0,
          "elapsed_seconds": 0.0007634410012542503,
          "D2": 0.016003911258155305,
          "RMSE": 0.015571881835906796,
          "Dinf": 0.04225892274172349,
          "recovery_percent": 0.0
        },
        {
          "candidates": 50,
          "elapsed_seconds": 0.016337996001311694,
          "D2": 0.00650180245826259,
          "RMSE": 0.009925329136029903,
          "Dinf": 0.0258119589586705,
          "recovery_percent": 36.261209527397355
        },
        {
          "candidates": 100,
          "elapsed_seconds": 0.04379956100092386,
          "D2": 0.005905506638791626,
          "RMSE": 0.009459248031161074,
          "Dinf": 0.02754652857742993,
          "recovery_percent": 39.25430380964464
        },
        {
          "candidates": 150,
          "elapsed_seconds": 0.0679539240009035,
          "D2": 0.005902289539180572,
          "RMSE": 0.00945667115752281,
          "Dinf": 0.02754742906809239,
          "recovery_percent": 39.270852057733194
        },
        {
          "candidates": 200,
          "elapsed_seconds": 0.08826666800086969,
          "D2": 0.005505797822248522,
          "RMSE": 0.009133519536573927,
          "Dinf": 0.02754742906809239,
          "recovery_percent": 41.346077289687734
        },
        {
          "candidates": 250,
          "elapsed_seconds": 0.10817382800087216,
          "D2": 0.0053855540726460965,
          "RMSE": 0.009033233315430387,
          "Dinf": 0.027547654190758004,
          "recovery_percent": 41.99009849534762
        },
        {
          "candidates": 300,
          "elapsed_seconds": 0.13347373900069215,
          "D2": 0.0053855540726460965,
          "RMSE": 0.009033233315430387,
          "Dinf": 0.027547654190758004,
          "recovery_percent": 41.99009849534762
        },
        {
          "candidates": 350,
          "elapsed_seconds": 0.15926113200112013,
          "D2": 0.005350792397183999,
          "RMSE": 0.009004033100702453,
          "Dinf": 0.027547654190758004,
          "recovery_percent": 42.177617351678784
        },
        {
          "candidates": 400,
          "elapsed_seconds": 0.18408091300079832,
          "D2": 0.005133058406097918,
          "RMSE": 0.00881893486842968,
          "Dinf": 0.027547654190758004,
          "recovery_percent": 43.366286994971105
        },
        {
          "candidates": 450,
          "elapsed_seconds": 0.21321907400124473,
          "D2": 0.00505066761448491,
          "RMSE": 0.008747872134761402,
          "Dinf": 0.024320468463903976,
          "recovery_percent": 43.82263989063987
        },
        {
          "candidates": 500,
          "elapsed_seconds": 0.24733326000023226,
          "D2": 0.00505066761448491,
          "RMSE": 0.008747872134761402,
          "Dinf": 0.024320468463903976,
          "recovery_percent": 43.82263989063987
        }
      ],
      "details": {
        "score": "D2",
        "score_rounding_decimals": 9,
        "same_original_run_bytecode": true,
        "search_options": {
          "max_iterations": 1000,
          "max_open_set_size": 50,
          "top_k": 1,
          "num_new_order_nodes": 2,
          "num_new_exchange_nodes": 10,
          "num_clusters_range": [
            1,
            1
          ],
          "num_zones_range": [
            1,
            1
          ],
          "num_changes_range": [
            1,
            1
          ],
          "num_zone_changes": 0,
          "pull_strategy": "default",
          "exchange_coef": 0.75,
          "num_explore": 1,
          "n_jobs": 1,
          "backtrack_depth": 0,
          "stagnation_limit": 100,
          "output_mode": "silent"
        },
        "last_improvement_candidate": 430,
        "no_improvement_stop_candidates": 2000,
        "iterations": 74,
        "profile": {
          "criterion_seconds": 0.06089291203170433,
          "SIP_computation_seconds": 0.003600375990572502,
          "support_extraction_seconds": 0.042346615011410904,
          "candidate_generation_seconds": 0.09045587098626129,
          "design_copy_seconds": 0.04456018199380196,
          "design_update_seconds": 0.04406369997377624,
          "validation_seconds": 0.006036876993675833
        },
        "evaluations_per_second": 2021.505805685486,
        "average_candidate_criterion_seconds": 0.00012178582406340866,
        "peak_process_RSS_MB": 256.84375,
        "memory_measurement": "process-lifetime RSS high water; not incremental method memory",
        "high_precision_implementation": "unmodified run and builtin round",
        "initial_D2": 0.016003911258155305,
        "initial_RMSE": 0.015571881835906796
      },
      "case": "U"
    }
  ],
  "projected_seconds_per_full_GFS_run": 7.420211190001284,
  "projected_serial_seconds_for_20_GFS_repetitions": 148.40422380002568,
  "projected_parallel_seconds_for_20_GFS_repetitions": 24.734037300004278,
  "estimated_total_seconds": 652.2694825333505,
  "workers": 12,
  "estimate_assumptions": "3x serial pilot extrapolation for support growth/precision; 2x parallel slowdown; 40 DSD fits at 20s each; 420s oracle/calibration/artifact allowance",
  "full_GFS_runs": 80,
  "DSD_runs": 40,
  "repetitions_per_case_method": 20,
  "requires_approval": false,
  "reported_UTC": "2026-09-18T09:57:20.576053+00:00"
}
```
Saved manifest/source provenance:
```json
{
  "status": "completed",
  "N": 12,
  "n": 4,
  "cases": [
    "E",
    "U"
  ],
  "master_seed": 20260918,
  "repetitions": 20,
  "scored_candidate_budget": 5000,
  "workers": 12,
  "source_SHA256": {
    "graphical_sampling/structs.py": "5df373a8157fa3d67ddfdb17327d54cb97e554b7902b8846c06b9c3e6fadd00b",
    "graphical_sampling/order.py": "30e2563b1381d9f002b3205690afe3f40d0b9cf9a146e1ebf3dc204a0fb130d6",
    "graphical_sampling/__init__.py": "ca28e532b85d89410ce455548188841982e6cb4424b52f123aa2d6dfa997797e",
    "graphical_sampling/design.py": "eaaf3a973a2a9bf9127639f6802630cf87d2afb73bdb8bc3d6bb04c051c9ec8a",
    "graphical_sampling/population.py": "9353fef9ae32f645a67d5507272a92cae9c38ee5c3db993fd244b360d655ebf2",
    "graphical_sampling/random/generator.py": "a4467d763bf7bf5e93fdb1701fd167fce99dc34e57ae75676667e3fafaf6354c",
    "graphical_sampling/random/__init__.py": "2bc306618cfdb2f35157f830a97ba780615de1fea003a23534ac4c9c94b24404",
    "graphical_sampling/sampling/organizer.py": "3ade2db3481e288b5af479b44abdc47006623f6e6fe903326a732a373a3cf000",
    "graphical_sampling/sampling/order.py": "eaf5082b004a270382e62736287e5c97d089ae9025f87c985c843bdc7ec0c877",
    "graphical_sampling/sampling/entity.py": "4f531dc5d14cd61798fe61fa3e9371fe98c8792bc6770db2ff3bd48df1c4bde0",
    "graphical_sampling/sampling/kmeans_sampler.py": "1c7e6aa53eb4306bbb1d2ec89c2845e9082b03d96fe547db4133c9d44b14677e",
    "graphical_sampling/sampling/__init__.py": "02762dd513ff837f125566fa82d5a9572e0d4b4d51968ecfb7d2402e0a03873c",
    "graphical_sampling/sampling/border.py": "2ae52b3d1691a67d57f67cc53257fff53b20b9e7aadc7eb556aa1dd173f57af2",
    "graphical_sampling/sampling/random_sampling.py": "22de4ba945921c16bef475ade25800f1149b34c00473f09768748dc52d2b26bb",
    "graphical_sampling/sampling/kmeans_sampling_old.py": "ee538dbec766bfa2420bc26d3f19010395fcf5b173ac4d22b54c90495004772e",
    "graphical_sampling/sampling/builder.py": "0398b39f7adf322b7d247694fa3d2abe7def1f994b3c135e6f3f21182c5c0fef",
    "graphical_sampling/clustering/fip_balanced_nmeans.py": "3f4243e04a830888f86303cd601414818e4f43d17f3f6d9eb362ddcd920e74f9",
    "graphical_sampling/clustering/__init__.py": "97830c0886544c4d54a151b2fb9510128fe3d9fec068477119cb7fb746e472f9",
    "graphical_sampling/criteria/multi_objective.py": "2ece78e5237f65a8854950f58f830456f43c50743e761fe8201f8a7b2ccc1fa6",
    "graphical_sampling/criteria/__init__.py": "a3d6b9ec37b0046fbc684b31873c9cd3d2cf9aa265708df81e26cad81040d6ca",
    "graphical_sampling/criteria/criteria.py": "b9cd3c246e551ac05496359a5c6aa4ec5b1474976c6ccdbb42e08d761e22acf7",
    "graphical_sampling/index/moran.py": "48561e983446bf52352db9f74bcf9d49933b147b71e59693f2644cad1c7c7194",
    "graphical_sampling/index/__init__.py": "6f5520b573dd5e3a514f73625c5d21e225af28999a27437eeaa5ec2ab4d48069",
    "graphical_sampling/index/local_balance.py": "71e624ea0d2f57d01e2777e4f74098f361e83695e199b050e32f98c6d4377574",
    "graphical_sampling/index/voronoi.py": "659529083f34fb1ea1a1466d8ede10c437e5129e3fe46258972da66f2dc27498",
    "graphical_sampling/index/density_disparity.py": "a0c28d8d8968358828ab1ae908d8aaa87e8163eb25113ff5c2a783f832e06e43",
    "graphical_sampling/search/gbfs.py": "c4425ca56608c0772a9ec4a873a8faea0c7893ba0fc087986ab2123692412625",
    "graphical_sampling/search/__init__.py": "88cfaed6038ff87de12d68dce60a68170ac1eccd2764222775e462242ead4e8c",
    "graphical_sampling/index/store/voronoi_r.py": "be531aa8dd313a9723fde6c884e0a643ff68f15d6ce7aaa0906d4f46a89ff77e",
    "graphical_sampling/index/store/moran_r.py": "6502b7f9f5df0adec8da4ecbe4270bab9de8f796ed10067582bdd44bea63c467",
    "graphical_sampling/index/store/density.py": "0fc2c0d2afe9dbe76d613a0b9947534c8ef2243ae1fde80804207238f3a96339",
    "graphical_sampling/index/store/local_balance_r.py": "87680bd29a1567664316a1826bb0e4854ed7966fc6946cae2c6effa373dd634b",
    "prescribed_sip/n12_proof.py": "92d18766bd5f0eb3dab5b332ec8aa2f0b1bfe4ca9c5365f6b545a9787f97afae",
    "prescribed_sip/run_all.py": "e1289d6c922800608c9619f26641e97565a894523f55082559c10bd0309d64ce",
    "prescribed_sip/__init__.py": "28fcb7751448ff95649ab85ce43816a5d3cea05be827a39dbac95a01ae4f9300",
    "prescribed_sip/parallel.py": "3e8c462b98ad70db513be7f2fd5e158710c863cb053fe849a477e7578aa86d71",
    "prescribed_sip/experiments/diagnostics.py": "a0f63ef0c06b395bc3a3cab468fae30ae01c6acee20842094ec5c5cde9a12d6e",
    "prescribed_sip/experiments/profile.py": "75abf682364aebf9d3fe163b0eb1b8514d7c4eecc9e97b7cf98bc329d2030bda",
    "prescribed_sip/experiments/n12_targets.py": "8c1b5acf8d06eb6fa3d14f6e3bace1d80c21a34a44a91232a5de6668fde9e9ef",
    "prescribed_sip/experiments/targets.py": "5a4b408dccfd87628d5498f1bf37fd42c82647ba3d411b45f8926b055c880f0b",
    "prescribed_sip/experiments/n12_artifacts.py": "0d8da2203c5ef4e349834614a6103825aa0e321dd492568843d002f99a096eb5",
    "prescribed_sip/experiments/artifacts.py": "bbd333b46bd487a2720420a3be2bb025f7f0c975494afa6e801429df6cfe307b",
    "prescribed_sip/tests/test_parallel.py": "a84a5d05098c54bcab8e72e689c835a6ef548d8ea0d406028eca54a195bba367",
    "prescribed_sip/tests/test_pipeline.py": "07d7eb0e68044d843a1ce201607d981f91751b809f1e1c47ee84190fd0facac8",
    "prescribed_sip/tests/test_reporting.py": "5ea5b583cd51077f380df2223208fd6f58d57e3efa8a8123eaa72a90dee96d05",
    "prescribed_sip/tests/test_core.py": "511ed24f331d85c40b2818abb72f7cc84b811d6e7c6b8680647ce333fe89c1ea",
    "prescribed_sip/tests/test_package.py": "196769e7c3b205462fce10864b57528767afb04123f64b04b39494b73bb222a6",
    "prescribed_sip/tests/test_regressions.py": "2c7e219e9e4bf16def20bf5bc707bcaf13f1f837a9204f3ad35edae0578f6c84",
    "prescribed_sip/tests/test_diagnostics.py": "f05e814a029abfd54adf1d77390aeba95909b7b920901910da7b52d0933d15a4",
    "prescribed_sip/tests/test_competitors.py": "2e19261e157f8d79468f564aaeb42f7dd9e0d3eb426dc7fc4f6ae568af3a33a0",
    "prescribed_sip/tests/test_n12_artifacts.py": "89e1e5a5c47d7422d9b4cb10485456cdc8835a2724fe433328ca78835486cee3",
    "prescribed_sip/tests/test_targets.py": "5cc914ac153487c590785afe1a3d53c9378ddf632a3e3a12997f181413eea674",
    "prescribed_sip/tests/test_n12_proof.py": "015580addd2b658e707749ab0ea3b7d972f2df36866e37980c67c87e95d02249",
    "prescribed_sip/tests/test_search.py": "a58f22d6d2decc6c86a213cc24411aa50e87469b5916c1ef0fbffb86a5e55dfe",
    "prescribed_sip/tests/test_profile.py": "0826a8027f5213221ccaf10e198cbd17c471a540c07efda0cd91a45255b3b2eb",
    "prescribed_sip/methods/cp2.py": "54e4177a556cde38e68b3416458edc86b877419fec130ffb5bc2e9b17d691681",
    "prescribed_sip/methods/sinha.py": "cf1c4fe3fe80fa182c780b5dcf32ca928ebd6716535e41c7df2460fcb21872d5",
    "prescribed_sip/methods/dsd.py": "6b1d22d122b730c5c2256c06d207c472ea2e7685e2c897ebf56d12d18c52fab5",
    "prescribed_sip/methods/exact.py": "ac1f904d41940cdd90079168235950e9fafd383b1b8855689274068adbe73439",
    "prescribed_sip/methods/exchange_search.py": "d4929bd9ec6a236e6fbbcc367bfdb7ed51fb08ea6431a1fe4752c080f9204cfc",
    "prescribed_sip/methods/n12_original.py": "e7b4d4afe6e8547d309b4c0b315494196cd7b1c2bf6162a5c4a8a0555f6a166b",
    "prescribed_sip/methods/benchmarks.py": "b0e41ba1a1420d5a27a83cfac6050127f68b81c4dcf1bc32162a89982dfe64ea",
    "prescribed_sip/methods/package_gfs.py": "6678668ccddcd2f0365d1db3ce4001e2f1eef46f6ef854f32462f35c629e7be1",
    "prescribed_sip/core/runtime.py": "b26f46334aae87020a9d09f023100871a89b7dc2fa6b986121a3d9bafd2cd531",
    "prescribed_sip/core/objective.py": "3d8b108fff51bb0692755438c22eb2b4c69689ac5801bc49f9bed7ff4b20e9e0",
    "prescribed_sip/core/design.py": "d87d1699530eacb46ff79f1f205fa132e148b32f85c90b8272121fb02be11334"
  },
  "started_UTC": "2026-09-18T09:57:20.702132+00:00",
  "target_SHA256": {
    "E": "835242593f54044f2e69ddb31063fb93949f36db8d1c1b4b973fc8dd584a2f7c",
    "U": "da40192d48e1931b721ee291e1d11733135c35e2173a1c1bc4f2940f916da3ca"
  },
  "runtime_gate_estimate_seconds": 652.2694825333505,
  "score": "D2",
  "parallel_start_method": "spawn",
  "scope": "exactly two fixed targets; no larger population or broad-study restart",
  "finished_UTC": "2026-09-18T09:58:43.532893+00:00",
  "execution_seconds": 82.69153826799993,
  "results_rows": 128,
  "postprocessing_source_SHA256": {
    "prescribed_sip/experiments/n12_artifacts.py": "c6471c4a9de2ebe49020253186425404802c687646c5e4b10a1fd6e9d5bf1338",
    "prescribed_sip/tests/test_n12_artifacts.py": "58de7233544825870434d18f7dfb585845f8f3bcbb2d6fc583f542570b98ee2d"
  },
  "postprocessing_note": "Report-only changes after simulation: observed full traces, roundoff-safe plateau summaries, DSD covariance restrictions, timing context. Numerical runs unchanged.",
  "timing_context": "execution_seconds measures stochastic comparison plus initial artifact generation, after deterministic fits; excludes target generation and pilot. CP2/LP/Sinha serial; GFS/DSD concurrent independent workers.",
  "postprocessing_UTC": "2026-09-18T10:04:23.095420+00:00",
  "study_request_SHA256": "bc0d78cce615e489050882575a75b1df6e6f90c7aed1c3e233ba72ff192bc804",
  "returned_design_validation": {
    "max_dense_or_analytic_vs_support_SIP_error": 4.996003610813204e-16
  },
  "verification": {
    "test_suite": "tests and prescribed_sip/tests",
    "passed_tests": 78,
    "passed_saved_design_audits": 130,
    "expected_Sinha_NA": 2,
    "max_FIP_error": 1.326716514427062e-14,
    "max_probability_sum_error": 2.1871393585115584e-14,
    "max_saved_SIP_discrepancy": 0.0,
    "checked_UTC": "2026-09-18T10:05:43.417678+00:00"
  },
  "software_environment": {
    "Python": "3.12.13 (main, May 10 2026, 19:30:01) [Clang 22.1.3 ]",
    "platform": "Linux-7.0.0-31-generic-x86_64-with-glibc2.43",
    "git_HEAD": "dc2881af84a41e4f7658b6d21401e58470871b55",
    "versions": {
      "numpy": "2.4.4",
      "scipy": "1.17.1",
      "matplotlib": "3.10.8",
      "pandas": "3.0.2",
      "osqp": "1.1.3",
      "PyYAML": "6.0.3",
      "graphical-sampling": "0.1.0",
      "pytest": "9.0.2"
    }
  }
}
```

Independent repetitions are parallelized with one BLAS/OpenMP thread per process. Worker counts and timing estimates above come from saved provenance; absent metadata is not estimated.

Deterministic CP(2), exact LP/QP and Sinha fits ran serially before the worker pool. Stochastic Original GFS, high-precision GFS and DSD runtimes were measured under contention from up to twelve concurrent workers. These are descriptive execution costs, not controlled speed ratios. Manifest execution_seconds measures the comparison and artifact generation interval after deterministic fitting; it excludes deterministic fits, target preparation and the timing pilot, so it is not full end-to-end experiment time.

## Interpretation and limits

These are two particular frozen feasible targets and finite-budget runs. Compare equal and unequal outcomes descriptively; their independently generated targets differ in initial-to-target SIP RMSE, entropy and dependence, so reconstruction difficulty cannot be attributed causally to FIP inequality. These two examples cannot isolate an inequality effect across populations. The oracle establishes numerical attainability within its feasibility tolerance. Failure of a restricted GFS search to recover a feasible target does not establish infeasibility or a certified local minimum. Paired rounding sensitivity isolates the comparison criterion only when the saved initializations/seeds are paired. CP(2) enumeration and LP do not establish large-N practicality; DSD and Sinha have structural restrictions. The experiment stops here, with no extrapolated N>12 simulation.
