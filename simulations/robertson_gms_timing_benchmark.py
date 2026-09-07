"""One-shot NMS/GMS wall-clock benchmark for the Robertson aggregated population."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

import graphical_sampling as gs


ROOT = Path(__file__).resolve().parents[1]
POPULATION_FILE = ROOT / "simulations/populations/real/AggregatedPop1000.csv"
OUTPUT_FILE = ROOT / "simulations/results/robertson_gms_timing_n100/benchmark.json"

N = 1000
N_SAMPLE = 100
PI = N_SAMPLE / N
MI_TARGET = -0.69
SEED = 20260907

POINT_STRATEGIES = [
    "None", "lexico_xy", "lexico_yx", "projection", "dist_from_origin",
    "angle", "spiral", "hilbert", "snake", "knight_move", "random",
]
ZONE_STRATEGIES = POINT_STRATEGIES.copy()
ZONE_MODES = ["sweep_xy", "sweep_yx"]

def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
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


def make_population() -> gs.Population:
    frame = pd.read_csv(POPULATION_FILE)
    coords = frame[["x", "y"]].to_numpy(dtype=float)
    if coords.shape != (N, 2):
        raise RuntimeError(f"Expected {(N, 2)} coordinates, found {coords.shape}")
    inclusions = np.full(N, PI, dtype=float)
    # The paper's zero-weight VAR component is still evaluated, so supply the
    # Robertson corrugated/plane auxiliary variable used in the notebook.
    x1, x2 = coords[:, 0], coords[:, 1]
    plane = 3 * (x1 + x2) + np.sin(6 * (x1 + x2))
    return gs.Population(coords, inclusions, n=N_SAMPLE, variable=plane)


def search_initial_nms(pop: gs.Population) -> dict:
    """Fit the NMS clustering once, then search the paper ordering grid."""
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
                    pop,
                    fbn.clusters,
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
                        "design": design,
                        "mi": mi,
                        "objective": objective,
                        "point_strategy": point_strategy,
                        "zone_strategy": zone_strategy,
                        "zone_mode": zone_mode,
                    }
                    print(
                        f"NMS NEW BEST candidate={evaluated} MI={mi:.8f} "
                        f"{zone_mode} {zone_strategy}->{point_strategy}",
                        flush=True,
                    )
            except Exception:
                continue
    if best is None:
        raise RuntimeError("NMS ordering search produced no valid design")
    return {"best": best, "valid_designs": evaluated}


def verify_design(design: gs.Design) -> dict:
    samples, probabilities = design.all_samples_and_probs
    sizes = np.asarray([len(set(map(int, sample))) for sample in samples])
    target = design.pop.inclusions
    actual = design.fip
    max_error = float(np.max(np.abs(actual - target)))
    return {
        "number_of_support_samples": int(len(samples)),
        "probability_mass": float(np.sum(probabilities)),
        "sample_size_min": int(sizes.min()),
        "sample_size_max": int(sizes.max()),
        "sample_size_preserved": bool(np.all(sizes == N_SAMPLE)),
        "target_pi": PI,
        "actual_pi_min": float(actual.min()),
        "actual_pi_max": float(actual.max()),
        "max_abs_pi_error": max_error,
        "first_order_inclusions_preserved_atol_1e-6": bool(
            np.allclose(actual, target, rtol=0.0, atol=1e-6)
        ),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    started = datetime.now().astimezone()
    pop = make_population()

    nms_start = perf_counter()
    nms = search_initial_nms(pop)
    nms_elapsed = perf_counter() - nms_start
    initial_design = nms["best"]["design"]
    initial_mi = float(initial_design.moran[0])
    print(f"NMS COMPLETE elapsed={nms_elapsed:.6f}s selected_MI={initial_mi:.8f}", flush=True)

    criterion = gs.criteria.MultiObjectiveCriteria(
        objectives=["MI", "VAR"],
        weights=[1, 0],
        scales={"MI": 0.05, "VAR": 15000},
        auxiliary=pop.variable,
    )
    gms = gs.search.GreedyBestFirstSearch([initial_design], criterion)
    gms_starting_mi = float(gms.best_design.moran[0])
    if gms.best_design is not initial_design or gms_starting_mi != initial_mi:
        raise RuntimeError(
            f"GMS did not receive the selected NMS design: "
            f"NMS MI={initial_mi}, GMS initial MI={gms_starting_mi}"
        )
    print(
        f"GMS START verified_initial_MI={gms_starting_mi:.8f} "
        f"same_design_object={gms.best_design is initial_design}",
        flush=True,
    )

    gms_start = perf_counter()
    last_report = [gms_start]

    def report_progress(engine, iteration):
        now = perf_counter()
        if now - last_report[0] >= 60:
            print(
                f"LIVE elapsed={now-gms_start:.1f}s iteration={iteration} "
                f"best_MI={engine.best_design.moran[0]:.8f}",
                flush=True,
            )
            last_report[0] = now

    gms.run(
        max_iterations=1_000_000,
        max_open_set_size=2000,
        top_k=1,
        num_new_order_nodes=100,
        num_new_exchange_nodes=0,
        num_clusters_range=(1, 1),
        num_zones_range=(1, 1),
        num_changes_range=(1, 1),
        num_zone_changes=0,
        pull_strategy="random",
        exchange_coef=0.33,
        num_explore=1,
        window=None,
        n_jobs=-1,
        backtrack_depth=1,
        stagnation_limit=5000,
        y_main=pop.variable,
        y_aux=pop.variable,
        stop_when=lambda design, value: design.moran[0] < MI_TARGET,
        progress_callback=report_progress,
    )
    gms_elapsed = perf_counter() - gms_start

    final_mi = float(gms.best_design.moran[0])
    if not final_mi < MI_TARGET:
        raise RuntimeError(f"GMS ended at MI={final_mi}, not below {MI_TARGET}")

    result = {
        "benchmark": "Robertson aggregated NMS/GMS timing",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "population": {
            "file": str(POPULATION_FILE.relative_to(ROOT)),
            "N": N,
            "n": N_SAMPLE,
            "inclusion_probabilities": "equal",
            "pi_i": PI,
        },
        "threshold": {"comparison": "strictly_less_than", "moran_i": MI_TARGET},
        "reproducibility": {"seed": SEED, **git_metadata()},
        "settings": {
            "nms": {
                "point_strategies": POINT_STRATEGIES,
                "zone_strategies": ZONE_STRATEGIES,
                "zone_modes": ZONE_MODES,
                "num_splits": [1],
                "shift_jumps": [0],
                "num_zones_sweep": [1, 1],
                "centroid_grid_x": 4,
                "r_sample_per_cluster": 1,
                "init_clust_method": "expanded",
                "topological": True,
                "shuffle": False,
                "valid_designs_evaluated": nms["valid_designs"],
                "selected_point_strategy": nms["best"]["point_strategy"],
                "selected_zone_strategy": nms["best"]["zone_strategy"],
                "selected_zone_mode": nms["best"]["zone_mode"],
            },
            "gms": {
                "objective": "modified standardized Moran's I",
                "objectives": ["MI", "VAR"],
                "weights": [1, 0],
                "scales": {"MI": 0.05, "VAR": 15000},
                "max_open_set_size": 2000,
                "top_k": 1,
                "num_new_order_nodes": 100,
                "num_new_exchange_nodes": 0,
                "num_changes_range": [1, 1],
                "window": None,
                "requested_n_jobs": -1,
                "backtrack_depth": 1,
                "stagnation_limit": 5000,
            },
        },
        "machine": {
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "ram_bytes": ram_bytes(),
            "ram_gib": ram_bytes() / 1024**3,
            "operating_system": platform.platform(),
            "python_version": platform.python_version(),
            "joblib_version": joblib.__version__,
            "effective_joblib_worker_count": joblib.effective_n_jobs(-1),
        },
        "results": {
            "nms_initial_moran_i": float(initial_design.moran[0]),
            "final_gms_moran_i": final_mi,
            "gms_iterations_required": int(gms.iterations_completed),
            "nms_elapsed_seconds": nms_elapsed,
            "gms_elapsed_seconds": gms_elapsed,
            "total_elapsed_seconds": nms_elapsed + gms_elapsed,
        },
        "verification": verify_design(gms.best_design),
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
