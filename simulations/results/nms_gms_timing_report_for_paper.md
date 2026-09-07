# Computational-time report for NMS and GMS

## Purpose and timing protocol

This report summarizes three new wall-clock executions of the current NMS/GMS implementation for the revised JSSAM paper. GMS optimized the modified standardized Moran's I used in the simulations. In every run, the exact best design object returned by the fresh NMS search was supplied to GMS.

Durations were measured using `time.perf_counter()` around the complete NMS and GMS calls. These are single-run practical benchmarks—not averages, expected runtimes, complexity results, or extrapolations.

## Summary

| Setting | Comparator MI | NMS MI | Best GMS MI | Iterations | NMS time | GMS time | Total | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Robertson aggregated, N=1000, n=100, EP | KMC: -0.690000 | -0.428684 | -0.690687 | 294 | 72.274 s | 608.537 s | 680.811 s | Strict win |
| Meuse, N=155, n=20, EP | WAVE: -0.579826 | -0.417490 | -0.577728 | 875 | 7.610 s | 421.3 s at stop | 428.9 s at stop | Near tie; manually stopped |
| Meuse, N=155, n=10, EP | WAVE: -0.447192 | -0.358825 | -0.447542 | 36 | 8.520 s | 22.388 s | 30.907 s | Strict win |

EP denotes equal first-order inclusion probabilities.

## 1. Robertson aggregated population, n=100

### Design and comparator

- Population size: `N = 1000`.
- Sample size: `n = 100`.
- Equal inclusion probabilities: `pi_i = 0.1`.
- Comparator: Robertson and Price KMC with `alpha = 0`, Moran's I `= -0.69`.
- Stopping rule: first best GMS design with Moran's I `< -0.69`.

### Results

- Initial NMS Moran's I: `-0.42868385295496586`.
- Final GMS Moran's I: `-0.6906866248701398`.
- Improvement from NMS: `0.26200277191517394` in the favorable, more-negative direction.
- Margin beyond KMC: `0.0006866248701398`.
- GMS iterations: `294` completed iterations. The terminal log index was 293 because logging is zero-based.
- NMS time: `72.27413853800044 s`.
- GMS time: `608.5366479050008 s`.
- Total time: `680.8107864430012 s`, approximately `11.35 min`.

### Integrity verification

- Support size: 10 samples.
- All support samples had size 100.
- Actual inclusion probabilities ranged from `0.10000000149011612` to `0.10000000149011612`.
- Maximum absolute error from the prescribed inclusion probabilities: `1.4901161277114383e-9`.
- Verification at absolute tolerance `1e-6`: passed.

### Suggested paper text

> In a single computational-time benchmark using the aggregated Robertson population (`N=1000`) with equal inclusion probabilities and `n=100`, initial NMS construction and search required 72.27 s and produced a design with modified standardized Moran's I of -0.4287. Starting from that design, GMS required 294 iterations and 608.54 s to attain Moran's I of -0.6907, thereby improving upon the criterion-matched KMC reference value of -0.69. Total NMS-plus-GMS wall-clock time was 680.81 s (11.35 min).

## 2. Meuse, n=20: near tie with WAVE

### Design and comparator

- Population size: `N = 155`.
- Sample size: `n = 20`.
- Equal inclusion probabilities: `pi_i = 20/155 = 0.12903225806451613`.
- WAVE mean Moran's I: `-0.5798260192958965`.
- WAVE Moran standard deviation: `0.05953491790047084`.
- WAVE result was calculated from 5,000 samples.
- Source: `simulations/results/equal_M=5000/summary_meuse_n=20_corr=0.0.csv`.

### Results at manual termination

- Initial NMS Moran's I: `-0.41748992` (live output precision).
- Best GMS Moran's I: `-0.57772765` (live output precision).
- GMS iterations completed: `875`.
- Remaining numerical gap to the WAVE mean: approximately `0.00209837`.
- Both WAVE and GMS round to `-0.58` at two decimal places.
- NMS time: `7.609535 s`.
- GMS elapsed time at manual stop: approximately `421.3 s`.
- Total elapsed time at manual stop: approximately `428.9 s`, or `7.15 min`.

### Interpretation

This run was stopped manually because both methods rounded to `-0.58`. It did not satisfy the original strict threshold, so it must not be described as a strict GMS win. It shows that GMS approached the WAVE mean to within about 0.0021 after roughly seven minutes. Because the run was interrupted before its normal result-writing stage, the live terminal measurements are the available record.

### Suggested paper text

> For Meuse with equal inclusion probabilities and `n=20`, WAVE had a simulated mean modified standardized Moran's I of -0.5798. A fresh NMS design had Moran's I of -0.4175 and required 7.61 s to construct and select. After 875 GMS iterations and approximately 421.3 s, the best GMS value was -0.5777. Thus, both methods rounded to -0.58, although GMS had not strictly surpassed the unrounded WAVE mean when the run was manually terminated.

## 3. Meuse, n=10: strict improvement over WAVE

### Design and comparator

- Population size: `N = 155`.
- Sample size: `n = 10`.
- Equal inclusion probabilities: `pi_i = 10/155 = 0.06451612903225808`.
- WAVE mean Moran's I: `-0.447192092079173`.
- WAVE Moran standard deviation: `0.0752814452985657`.
- WAVE result was calculated from 5,000 samples.
- Source: `simulations/results/equal_M=5000/summary_meuse_n=10_corr=0.0.csv`.
- Stopping rule: first best GMS design with Moran's I `< -0.447192092079173`.

### Results

- Initial NMS Moran's I: `-0.35882473002466087`.
- Final GMS Moran's I: `-0.4475423670025226`.
- Improvement from NMS: `0.08871763697786173`.
- Margin beyond the WAVE mean: `0.00035027492334960275`.
- GMS iterations: `36`.
- NMS time: `8.519667308999487 s`.
- GMS time: `22.387629620001462 s`.
- Total time: `30.90729692900095 s`.

### Integrity verification

- Support size: 31 samples.
- All support samples had size 10.
- Actual inclusion probabilities ranged from `0.06451614201068878` to `0.06451614201068878`.
- Maximum absolute error from the prescribed inclusion probabilities: `1.2978430705135047e-8`.
- Verification at absolute tolerance `1e-6`: passed.

### Suggested paper text

> For the Meuse population (`N=155`) with equal inclusion probabilities and `n=10`, WAVE had a mean modified standardized Moran's I of -0.4472 across 5,000 simulated samples. A fresh NMS search required 8.52 s and produced an initial design with Moran's I of -0.3588. Starting from that exact NMS design, GMS required 36 iterations and 22.39 s to reach Moran's I of -0.4475, strictly improving upon the WAVE mean. Total NMS-plus-GMS wall-clock time was 30.91 s.

## Search settings

- Objective: modified standardized Moran's I.
- Implemented objectives: `MI` and `VAR`, with weights `[1, 0]`.
- GMS scales: Moran `0.05`, variance `15000`.
- NMS initialization: FIP-balanced n-means with `expanded` initialization, `centroid_grid_x = 4`, `r_sample_per_cluster = 1`, and `split_size = 0.001`.
- NMS ordering search: 11 point strategies, 11 zone strategies, and two sweep directions, giving 242 evaluated orderings on the fitted clustering.
- GMS open-set limit: 2,000.
- New order neighbors per iteration: 100.
- New exchange neighbors per iteration: zero.
- Simultaneously explored frontier nodes: one.
- Order changes per neighbor: one.
- Search window: unrestricted.
- Requested joblib workers: `n_jobs = -1`; effective workers: 16.
- Backtrack depth: one.
- Stagnation limit: 5,000.

The threshold and progress instrumentation observed the current best design and terminated the existing search when requested. It did not alter candidate generation, scoring, neighborhood construction, or open-set handling.

## Computing environment

- CPU: 11th Gen Intel Core i9-11900H at 2.50 GHz.
- Logical CPUs: 16.
- RAM: 32,822,632,448 bytes (`30.57 GiB`).
- Operating system: Linux 7.0.0-31-generic, x86-64, glibc 2.43.
- Python: 3.12.13.
- joblib: 1.5.3.
- Effective joblib workers: 16.
- Repository commit: `046dbca432c8541d27547f25c879395660605d09`.
- The working tree contained local changes during execution.

## Reproducibility qualification

The scripts record the repository commit, global seed, settings, machine information, measured durations, and integrity checks. The current implementation also creates internally unseeded NumPy `default_rng()` instances. Therefore, the computational procedure is documented and rerunnable, but exact stochastic paths, selected NMS designs, iteration counts, and stopping times may vary between independent executions unless that existing RNG behavior is made explicitly seedable.

Completed result files:

- `simulations/results/robertson_gms_timing_n100/benchmark.json`.
- `simulations/results/meuse_gms_wave_timing_n10_ep/benchmark.json`.

The manually stopped Meuse n=20 run has no completed JSON result file.
