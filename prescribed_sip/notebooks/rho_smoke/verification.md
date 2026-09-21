# Dependence reporting verification

All 92 prescribed-SIP tests passed, including independent rho calculation with unequal FIP, boundary exclusions/no-valid-pair/zero-baseline handling, final mean/sample SD, serial/parallel equivalence and exact raw-search equivalence against the unchanged Directed search.

The executed Jupyter smoke uses the same N12 n4 unequal-FIP permuted target as the prior permutation verification: 20 accepted target exchanges, 60 Original scores and three Directed runs of 2,000 directions with two distinct worker PIDs. Per-run raw RMSE matched the previously saved smoke values to 1e-15. Every main table/optimizer trace includes raw RMSE, RMSE_rho, Recovery_rho, maximum SIP error, runtime, evaluations, support size and entropy. Saved settings include tolerance, pair counts, target permutation and seeds; final supports are also exported.

This is verification only, not the configured full experiment. Saved verification results: `prescribed_sip/results/notebook_directed/N12_n4_20260918T163021_031959Z/`.
