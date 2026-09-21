# Notebook repetitions verification

This folder contains a small executed verification copy, not the 30 × 100,000 experiment. The original notebook retains its user settings and has not been executed under the new full budget.

The real Jupyter kernel ran N=12, n=4, unequal FIP, 20 accepted target exchanges, 60 Original GFS candidate evaluations and three Directed runs of 200 directions with batch 50 using two distinct child worker PIDs. All six method rows included raw RMSE, NRMSE and Recovery_N. Seeds, saved CSV metrics and the frozen target matrix were checked for consistency. Each Directed design passed independent FIP, probability, fixed-size and SIP checks.

The notebook test suite also compares serial and parallel numerical results under the same seeds and recomputes final errors and mean/sample SD independently. No frozen paper outputs or package source were changed.

Saved verification results: `prescribed_sip/results/notebook_directed/N12_n4_20260918T151207_790653Z/`.
