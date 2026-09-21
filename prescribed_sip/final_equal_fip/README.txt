FINAL PRESCRIBED-SIP STUDY FINAL EXECUTED BUDGETS
Completed UTC: 2026-09-20T12:49:44.113991+00:00
Git commit: dc2881af84a41e4f7658b6d21401e58470871b55 (working source snapshot includes the user's uncommitted package changes).
Python/dependency versions and source hashes: configs/environment.json.
Source snapshot and working notebook: configs/code/. Full settings: configs/study.json.

Scenarios: equal pi_i = n/N. Units are ordered on a line.
  N=12, n=4, pairs=66, possible samples=495; population seed=12345, permutation seed=240930, exchange seed=240931
  N=20, n=5, pairs=190, possible samples=15504; population seed=13345, permutation seed=240938, exchange seed=240939
  N=30, n=5, pairs=435, possible samples=142506; population seed=14345, permutation seed=240948, exchange seed=240949

ONE frozen reachable target per scenario: reproducibly permute population order while preserving original IDs/pi_i; construct the package's standard fixed-size GFS design on that order; perform exactly 10000 accepted non-null random package Design.exchange operations (M=1, alpha=0.75); take endpoint SIP and restore original unit order. The objective is not used to select the target endpoint.
targets/*/metadata.json documents seeds, permutation, first-order probabilities, initial discrepancy and checksums. Target probability distributions are saved as AUDIT_ONLY and never loaded by reconstruction workers.
Both GFS reconstruction methods begin at the same standard original-order fixed-size GFS design p0. Competitors retain their existing native initializations: CP(2)'s dual starts at zero parameters after uniform-family FIP calibration, DSD first tries the original projection order then random orders, and the LP/QP oracle solves the sparse constraints with its existing solver. All methods receive the prescribed FIP and frozen SIP target; p0 is the COMMON recovery baseline. No reconstruction receives the target probability distribution.

Methods:
  Initial design: one deterministic reference per scenario.
  Original GFS + restarts: current actual package GreedyBestFirstSearch, frontier=50, top_k=1, new_orders=2, new_exchanges=10, exchange coefficient=0.75, partitions=1; documented wrapper restarts from best design after frontier exhaustion with 3 random package exchanges. No new optimizer replaces the package search. 5 independent repetitions, at most 20000 scored evaluations. N=30 per-run time cap 1800 s. The per-run details/restarts.csv give actual restart counts and settings. Stagnation is disabled as in the active working notebook.
  GFS-Directed: unchanged residual-directed four-support mass exchanges, exact feasible optimal step along each direction, best candidate from batch 50; 30 independent seeds, at most 100000 scored evaluations each. Raw SIP objective unchanged; early stop raw SIP RMSE <= 1e-10. Inclusive boundary is an explicitly requested stopping adaptation only.
  CP(2) — enumerated fit: existing full-support log-partition exponential-family fit, L-BFGS gtol=1e-11 / ftol=1e-15, with mandatory final Newton FIP calibration. Up to 50000 dual evaluations and 60 s for fitting, followed by separately reported mandatory finalization. Enumeration threshold 200000; includes a genuine N=30 attempt with 142506 samples. Limitations of this implementation do not imply a theoretical scalability limit of CP(2).
  DSD — projection/order fit: existing projection determinant design, 250 order fitting trials, full sample enumeration for independent returned probabilities.
  Exact LP/QP oracle: existing sparse LP feasibility / QP oracle; 60 s LP solver budget. N=30 optional attempt is bounded by the same solver budget. A certified feasible LP solution suffices for these reachable targets.
All algorithms use float64. Runs use 6 independent processes with BLAS threads restricted to one. Runtime is method wall time under parallel load; probability/SIP serialization and audit times are outside algorithm runtime. Evaluations count GFS scored candidates, CP(2) dual evaluations, DSD trials; exact LP/QP evaluations are 0 (solver iterations are separate). The auxiliary Task wall seconds field includes setup and method execution, measured before the final independent audit/serialization. Peak process RSS is a process-lifetime high-water mark; workers are reused.

Metrics: raw RMSE over ALL i<j SIP pairs. RMSE_rho uses the SAME prescribed pi_i denominators sqrt(pi_i(1-pi_i)pi_j(1-pi_j)); pairs involving pi within 1e-12 of 0 or 1 are excluded and counts explicitly saved. Recovery_rho=100*(1-final_RMSE_rho/initial_RMSE_rho). Undefined recovery is NA. Directed optimization minimizes raw SIP error, not RMSE_rho.
Tables: full precision CSV; between-run sample SD (ddof=1), median raw/rho RMSE, single-run SD=NA. SD measures reconstruction randomness conditional on the one fixed population/target in each case. Max SIP error/FIP error are maxima over repetitions. Compact tables show mean ± SD. Each scenario has PDF and 300-dpi PNG figures. Directed convergence connects the existing log-spaced recorded incumbent checkpoints. Mean/median aggregate curves use interpolation linear in log(RMSE_rho) between checkpoints, with stopped endpoints held constant to the common candidate budget; aggregate CSV values between recorded points are interpolated. Raw per-run convergence.csv retains all actual recorded values. Best Directed heatmap/scatter selection uses final independently recomputed RMSE_rho and is separate from across-run mean results.
Independent audit: validation/validation.csv and audit.json, reconstructed from serialized final probabilities; mass, FIP, full SIP, fixed sample sizes, fixed-size row identities, support bound, pair counts, frozen-target hashes, reported metrics and recovery are checked with tolerance 1e-10 (saved SIP agreement 1e-12).

Reproduction from the project root (same environment):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m prescribed_sip.final_study --fip-type equal --output NEW_OUTPUT_FOLDER
To resume an interrupted study, run ./resume.sh in this folder. It imports the frozen source snapshot and skips completed repetitions. Existing completed run and target files are retained and checked; only unfinished repetitions restart. Local computation does not require chat/API calls or tokens. configs/reconstruction_seeds.json records EVERY exact reconstruction seed. Code snapshot preserves this runner and the algorithms.

Unavailable methods / computation limitations:
N30_n5, Exact LP/QP oracle: timeout; Time limit reached. (HiGHS Status 13: model_status is Time limit reached; primal_status is None)
N30_n5, CP(2) — enumerated fit, repetition 1: returned valid design; status timeout; time_budget

Principal results: tables/Table_All_Scenarios_Compact.csv
Requested equal-FIP publication table: tables/Equal_FIP_Results.csv
