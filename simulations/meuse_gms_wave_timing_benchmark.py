"""One-shot Meuse NMS/GMS benchmark against the best WAVE MI result."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

import graphical_sampling as gs


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "simulations/populations/real/meuse.csv"
WAVE_FILE = ROOT / "simulations/results/equal_M=5000/summary_meuse_n=10_corr=0.0.csv"
OUTPUT_DIR = ROOT / "simulations/results/meuse_gms_wave_timing_n10_ep"
OUTPUT_FILE = OUTPUT_DIR / "benchmark.json"
REPORT_FILE = OUTPUT_DIR / "paper_report.md"

N_SAMPLE = 10
SEED = 20260907
POINT_STRATEGIES = [
    "None", "lexico_xy", "lexico_yx", "projection", "dist_from_origin",
    "angle", "spiral", "hilbert", "snake", "knight_move", "random",
]
ZONE_STRATEGIES = POINT_STRATEGIES.copy()
ZONE_MODES = ["sweep_xy", "sweep_yx"]


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def ram_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def git_metadata() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=ROOT, check=False, capture_output=True, text=True
        ).stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "working_tree_dirty": bool(run("git", "status", "--porcelain")),
    }


def load_inputs() -> tuple[gs.Population, float, dict]:
    frame = pd.read_csv(DATA_FILE)
    coords = frame[["x", "y"]].to_numpy(float)
    N = len(frame)
    inclusions = np.full(N, N_SAMPLE / N, dtype=float)
    pop = gs.Population(
        coords, inclusions, n=N_SAMPLE, variable=frame["cadmium"].to_numpy(float)
    )
    wave = pd.read_csv(WAVE_FILE)
    row = wave.loc[wave["Method"].eq("Wave")].iloc[0]
    return pop, float(row["Mm"]), {
        "replications": int(row["ite"]), "wave_mi_sd": float(row["Ms"])
    }


def search_initial_nms(pop: gs.Population) -> dict:
    fbn = gs.clustering.FIPBalancedNMeans(
        n=N_SAMPLE,
        centroid_grid_x=4,
        r_sample_per_cluster=1,
        init_clust_method="expanded",
        split_size=0.001,
    )
    fbn.fit(pop)
    best = None
    evaluated = 0
    for zone_mode in ZONE_MODES:
        fbn.fit_zones(num_zones=(1, 1), mode=zone_mode)
        for point_strategy, zone_strategy in product(POINT_STRATEGIES, ZONE_STRATEGIES):
            try:
                order = gs.Order.from_clusters(
                    pop, fbn.clusters,
                    zone_strategy=zone_strategy,
                    point_strategy=point_strategy,
                    num_splits=1,
                    topological=True,
                    shift_jump=0,
                    shuffle=False,
                    grid_dims=(1, 1),
                )
                design = gs.Design.from_order(pop, order)
                mi = float(design.moran[0])
                design._nht_variance = None
                variance = float(design.nht_variance)
                objective = mi / (0.05 + 1e-12) + 0.0 * variance / (1000 + 1e-12)
                evaluated += 1
                if best is None or objective < best["objective"]:
                    best = {
                        "design": design, "mi": mi, "objective": objective,
                        "point_strategy": point_strategy,
                        "zone_strategy": zone_strategy, "zone_mode": zone_mode,
                    }
                    print(
                        f"NMS NEW BEST candidate={evaluated} MI={mi:.8f} "
                        f"{zone_mode} {zone_strategy}->{point_strategy}", flush=True
                    )
            except Exception:
                continue
    if best is None:
        raise RuntimeError("NMS ordering search produced no valid design")
    return {"best": best, "valid_designs": evaluated}


def verify_design(design: gs.Design) -> dict:
    samples, probabilities = design.all_samples_and_probs
    sizes = np.array([len(set(map(int, sample))) for sample in samples])
    actual, target = design.fip, design.pop.inclusions
    return {
        "number_of_support_samples": int(len(samples)),
        "probability_mass": float(probabilities.sum()),
        "sample_size_min": int(sizes.min()),
        "sample_size_max": int(sizes.max()),
        "sample_size_preserved": bool(np.all(sizes == N_SAMPLE)),
        "target_pi": float(target[0]),
        "actual_pi_min": float(actual.min()),
        "actual_pi_max": float(actual.max()),
        "max_abs_pi_error": float(np.max(np.abs(actual - target))),
        "first_order_inclusions_preserved_atol_1e-6": bool(
            np.allclose(actual, target, rtol=0.0, atol=1e-6)
        ),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    started = datetime.now().astimezone()
    pop, wave_mi, wave_meta = load_inputs()

    nms_start = perf_counter()
    nms = search_initial_nms(pop)
    nms_elapsed = perf_counter() - nms_start
    initial_design = nms["best"]["design"]
    initial_mi = float(initial_design.moran[0])
    print(f"NMS COMPLETE elapsed={nms_elapsed:.6f}s selected_MI={initial_mi:.8f}", flush=True)

    criterion = gs.criteria.MultiObjectiveCriteria(
        objectives=["MI", "VAR"], weights=[1, 0],
        scales={"MI": 0.05, "VAR": 15000}, auxiliary=pop.variable,
    )
    gms = gs.search.GreedyBestFirstSearch([initial_design], criterion)
    gms_starting_mi = float(gms.best_design.moran[0])
    if gms.best_design is not initial_design or gms_starting_mi != initial_mi:
        raise RuntimeError("GMS did not receive the selected NMS design")
    print(
        f"GMS START verified_initial_MI={gms_starting_mi:.8f} "
        f"WAVE_target={wave_mi:.8f} same_design_object=True", flush=True
    )

    gms_start = perf_counter()
    last_report = [gms_start]
    def report_progress(engine, iteration):
        now = perf_counter()
        if now - last_report[0] >= 60:
            print(
                f"LIVE elapsed={now-gms_start:.1f}s iteration={iteration} "
                f"best_MI={engine.best_design.moran[0]:.8f}", flush=True
            )
            last_report[0] = now

    gms.run(
        max_iterations=1_000_000, max_open_set_size=2000, top_k=1,
        num_new_order_nodes=100, num_new_exchange_nodes=0,
        num_clusters_range=(1, 1), num_zones_range=(1, 1),
        num_changes_range=(1, 1), num_zone_changes=0,
        pull_strategy="random", exchange_coef=0.33, num_explore=1,
        window=None, n_jobs=-1, backtrack_depth=1, stagnation_limit=5000,
        y_main=pop.variable, y_aux=pop.variable,
        stop_when=lambda design, value: design.moran[0] < wave_mi,
        progress_callback=report_progress,
    )
    gms_elapsed = perf_counter() - gms_start
    final_mi = float(gms.best_design.moran[0])
    if not final_mi < wave_mi:
        raise RuntimeError(f"GMS ended at {final_mi}, not below WAVE {wave_mi}")

    verification = verify_design(gms.best_design)
    if not verification["sample_size_preserved"] or not verification["first_order_inclusions_preserved_atol_1e-6"]:
        raise RuntimeError("Final-design integrity verification failed")

    machine_ram = ram_bytes()
    result = {
        "benchmark": "Meuse NMS/GMS timing versus WAVE",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "population": {
            "file": str(DATA_FILE.relative_to(ROOT)), "N": pop.N, "n": pop.n,
            "inclusion_probabilities": "equal", "pi_i": float(pop.inclusions[0]),
        },
        "comparator": {
            "method": "WAVE", "moran_i_mean": wave_mi, **wave_meta,
            "source": str(WAVE_FILE.relative_to(ROOT)),
            "stopping_rule": "GMS best modified standardized Moran's I < WAVE mean",
        },
        "reproducibility": {"seed": SEED, **git_metadata()},
        "settings": {
            "nms": {
                "point_strategies": POINT_STRATEGIES,
                "zone_strategies": ZONE_STRATEGIES,
                "zone_modes": ZONE_MODES,
                "num_zones_sweep": [1, 1], "centroid_grid_x": 4,
                "r_sample_per_cluster": 1, "init_clust_method": "expanded",
                "topological": True, "valid_designs_evaluated": nms["valid_designs"],
                "selected_point_strategy": nms["best"]["point_strategy"],
                "selected_zone_strategy": nms["best"]["zone_strategy"],
                "selected_zone_mode": nms["best"]["zone_mode"],
            },
            "gms": {
                "objective": "modified standardized Moran's I",
                "objectives": ["MI", "VAR"], "weights": [1, 0],
                "scales": {"MI": 0.05, "VAR": 15000},
                "max_open_set_size": 2000, "top_k": 1,
                "num_new_order_nodes": 100, "num_new_exchange_nodes": 0,
                "num_changes_range": [1, 1], "window": None,
                "requested_n_jobs": -1, "backtrack_depth": 1,
                "stagnation_limit": 5000,
            },
        },
        "machine": {
            "cpu_model": cpu_model(), "logical_cpu_count": os.cpu_count(),
            "ram_bytes": machine_ram, "ram_gib": machine_ram / 1024**3,
            "operating_system": platform.platform(),
            "python_version": platform.python_version(),
            "joblib_version": joblib.__version__,
            "effective_joblib_worker_count": joblib.effective_n_jobs(-1),
        },
        "results": {
            "wave_moran_i": wave_mi,
            "nms_initial_moran_i": initial_mi,
            "final_gms_moran_i": final_mi,
            "improvement_beyond_wave": wave_mi - final_mi,
            "gms_iterations_required": int(gms.iterations_completed),
            "nms_elapsed_seconds": nms_elapsed,
            "gms_elapsed_seconds": gms_elapsed,
            "total_elapsed_seconds": nms_elapsed + gms_elapsed,
        },
        "verification": verification,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    report = f"""# Meuse computational-time benchmark: NMS/GMS versus WAVE

For the Meuse population (N={pop.N}) with equal first-order inclusion probabilities and n={pop.n}, the comparator was WAVE's mean modified standardized Moran's I of {wave_mi:.6f}, estimated from {wave_meta['replications']:,} samples. A fresh NMS search produced an initial design with Moran's I={initial_mi:.6f} in {nms_elapsed:.3f} seconds. Starting from that exact NMS design, GMS first improved upon WAVE after {gms.iterations_completed} iterations, attaining Moran's I={final_mi:.6f} in {gms_elapsed:.3f} seconds. Total NMS-plus-GMS wall-clock time was {nms_elapsed + gms_elapsed:.3f} seconds. Timings were measured with `time.perf_counter()` around the complete calls and were not estimated or extrapolated.

The benchmark ran on {cpu_model()} with {os.cpu_count()} logical CPUs and {machine_ram / 1024**3:.2f} GiB RAM, using {joblib.effective_n_jobs(-1)} effective joblib workers, Python {platform.python_version()}, and {platform.platform()}. The final GMS design preserved sample size n={pop.n}; its maximum absolute deviation from the prescribed equal inclusion probability ({pop.inclusions[0]:.12f}) was {verification['max_abs_pi_error']:.3e}.

Reproducibility note: the benchmark script records the repository commit and seed. The current implementation also contains internally unseeded `default_rng()` calls, so exact stochastic paths may differ across reruns unless that existing behavior is changed.
"""
    REPORT_FILE.write_text(report)
    print(json.dumps(result, indent=2))
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
