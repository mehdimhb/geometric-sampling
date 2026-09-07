# Meuse computational-time benchmark: NMS/GMS versus WAVE

For the Meuse population (N=155) with equal first-order inclusion probabilities and n=10, the comparator was WAVE's mean modified standardized Moran's I of -0.447192, estimated from 5,000 samples. A fresh NMS search produced an initial design with Moran's I=-0.358825 in 8.520 seconds. Starting from that exact NMS design, GMS first improved upon WAVE after 36 iterations, attaining Moran's I=-0.447542 in 22.388 seconds. Total NMS-plus-GMS wall-clock time was 30.907 seconds. Timings were measured with `time.perf_counter()` around the complete calls and were not estimated or extrapolated.

The benchmark ran on 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz with 16 logical CPUs and 30.57 GiB RAM, using 16 effective joblib workers, Python 3.12.13, and Linux-7.0.0-31-generic-x86_64-with-glibc2.43. The final GMS design preserved sample size n=10; its maximum absolute deviation from the prescribed equal inclusion probability (0.064516129032) was 1.298e-08.

Reproducibility note: the benchmark script records the repository commit and seed. The current implementation also contains internally unseeded `default_rng()` calls, so exact stochastic paths may differ across reruns unless that existing behavior is changed.
