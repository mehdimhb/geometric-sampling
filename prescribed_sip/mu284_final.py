"""Resumable, dedicated MU284 (N=281, n=15) prescribed-SIP final study.

This is intentionally separate from the notebooks and the generic small-population
study.  It freezes all inputs below ``--output`` before any expensive work starts;
rerunning the same command only fills missing checkpoints/runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from math import fsum
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prescribed_sip.core.dependence import DependenceError
from prescribed_sip.core.design import Support
from prescribed_sip.experiments.n12_targets import independent_sip, validate_design
from prescribed_sip.methods.dsd import projection_factor
from prescribed_sip.methods.notebook_directed import directed_repetition
from prescribed_sip.methods.package_gfs import extract_support, initial_design
from prescribed_sip.run_all import execution_lock, save_json, support_json

REPO = Path(__file__).resolve().parents[1]
TARGET_CHECKPOINTS = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000]
DIRECTED_CHECKPOINTS = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000,
                        1_000_000, 1_500_000, 2_000_000]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def study_config(smoke: bool = False) -> dict:
    """The immutable full-study contract; smoke merely shortens all budgets."""
    scale = 100 if smoke else 1
    target = [min(x, 100) for x in TARGET_CHECKPOINTS] if smoke else TARGET_CHECKPOINTS
    directed = [min(x, 200) for x in DIRECTED_CHECKPOINTS] if smoke else DIRECTED_CHECKPOINTS
    return {
        "study": "MU284 prescribed-SIP final", "N": 281, "n": 15, "sip_pairs": 39_340,
        "size_column": "P75", "target_seed": 2026091801,
        "target_alpha": .75, "target_checkpoints": sorted(set(target)),
        "target_accepted_exchanges": target[-1], "directed_checkpoints": sorted(set(directed)),
        "directed_evaluations": directed[-1], "directed_batch": 100 if not smoke else 10,
        "directed_repetitions": 10 if not smoke else 1, "original_evaluations": 20_000 if not smoke else 40,
        "original_repetitions": 1, "original_time_seconds": 3600, "dsd_trials": 1000 if not smoke else 8,
        "raw_rmse_stop": 1e-10, "rho_boundary_tol": 1e-12, "validation_tolerance": 1e-10,
        "methods": ["Initial GFS baseline", "Original GFS + restarts", "GFS-Directed", "DSD — projection/order fit"],
        "smoke": smoke,
    }


def load_mu284(repo: Path = REPO) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the already-cleaned 281-row paper CSV without renumbering its IDs."""
    frame = pd.read_csv(repo / "simulations/abc/populations/real/MU284.csv")
    if len(frame) != 281 or "LABEL" not in frame or "P75" not in frame:
        raise RuntimeError("MU284.csv is not the expected cleaned 281-unit paper population")
    unit_ids = frame["LABEL"].to_numpy(copy=True)
    sizes = frame["P75"].to_numpy(dtype=np.float64, copy=True)
    if len(np.unique(unit_ids)) != 281 or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise RuntimeError("MU284 LABEL/P75 data are invalid")
    pi = 15 * sizes / sizes.sum()
    if not np.all((pi > 0) & (pi < 1)) or not np.isclose(pi.sum(), 15, atol=1e-10):
        raise RuntimeError("MU284 P75 inclusion probabilities do not meet the fixed-size contract")
    return unit_ids, sizes, pi


def _support(path: Path) -> Support:
    d = json.loads(path.read_text())
    return Support(d["N"], d["n"], dict(zip(map(tuple, d["samples"]), d["probabilities"])))


def _metrics(support: Support, pi: np.ndarray, target: np.ndarray, baseline: np.ndarray, tol: float) -> tuple[dict, np.ndarray, dict]:
    sip = independent_sip(support)
    validation = validate_design(support, pi)
    validation["probability_sum"] = fsum(support.mass.values())
    validation["fixed_size"] = all(len(s) == support.n for s in support.mass)
    # ``validate_design`` already checks its named numerical errors.  Do not treat
    # legitimate quantities such as min mass or probability sum itself as errors.
    if not validation["fixed_size"]:
        raise RuntimeError(f"independent fixed-FIP validation failed: {validation}")
    pair = np.triu_indices(support.N, 1)
    dependence = DependenceError(pi)
    rmse_rho = dependence.from_matrices(sip, target)
    baseline_rho = dependence.from_matrices(baseline, target)
    residual = (sip - target)[pair]
    return ({"RMSE": float(np.sqrt(np.mean(residual ** 2))), "RMSE_rho": rmse_rho,
             "Recovery_rho": dependence.recovery(rmse_rho, baseline_rho),
             "Max SIP error": float(np.max(np.abs(residual))),
             "FIP max error": float(np.max(np.abs(np.diag(sip) - pi))),
             "Support size": len(support.mass), "Entropy": support.entropy(), **dependence.counts()}, sip, validation)


def _restore_package(pi: np.ndarray, coords: np.ndarray, permutation: np.ndarray, support: Support):
    """Rehydrate a checkpoint into the actual package Design for a target-walk resume."""
    from graphical_sampling.structs import _MaxHeap, _Sample
    design = initial_design(pi, coords, permutation)
    design._heaps = [_MaxHeap([-_Sample(float(p), frozenset(s)) for s, p in support.mass.items()])]
    design._all_samples_and_probs = None
    return design


def prepare_target(root: Path, config: dict) -> dict:
    target_dir = root / "target"; checkpoints = target_dir / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(exist_ok=True)
    frozen = target_dir / "metadata.json"
    if frozen.exists():
        return json.loads(frozen.read_text())
    unit_ids, sizes, pi = load_mu284()
    coords = np.column_stack((np.linspace(0., 1., len(pi)), np.zeros(len(pi))))
    permutation = np.random.default_rng(config["target_seed"]).permutation(len(pi))
    baseline_design = initial_design(pi, coords)
    baseline = extract_support(baseline_design); baseline_sip = independent_sip(baseline)
    np.save(target_dir / "pi.npy", pi); np.save(target_dir / "unit_ids.npy", unit_ids)
    np.save(target_dir / "sizes_p75.npy", sizes); np.save(target_dir / "permutation.npy", permutation)
    save_json(target_dir / "initial_design.json", support_json(baseline))
    existing = sorted((int(p.stem.split("_")[-1]), p) for p in checkpoints.glob("support_*.json"))
    if existing:
        accepted, path = existing[-1]; current = _restore_package(pi, coords, permutation, _support(path))
    else:
        accepted = 0; current = initial_design(pi, coords, permutation)
    attempted = accepted
    rows = pd.read_csv(target_dir / "distance.csv") if (target_dir / "distance.csv").exists() else pd.DataFrame()
    for goal in config["target_checkpoints"]:
        while accepted < goal:
            attempted += 1
            proposal = current.copy()
            # Stateless proposal seeds make an interrupted checkpoint resume bitwise reproducible.
            proposal._rng = np.random.default_rng(_seed(config["target_seed"], "exchange", attempted))
            for heap in proposal._heaps:
                heap.rng = np.random.default_rng(_seed(config["target_seed"], "heap", attempted))
            before = extract_support(current)
            proposal.exchange(partitions=1, pull_strategy="random", exchange_coef=config["target_alpha"])
            proposal.merge_identical(); after = extract_support(proposal)
            if sum(abs(after.mass.get(s, 0.) - before.mass.get(s, 0.)) for s in before.mass.keys() | after.mass.keys()) <= 1e-14:
                continue
            current = proposal; accepted += 1
        support = extract_support(current); values, sip, validation = _metrics(support, pi, baseline_sip, baseline_sip, config["validation_tolerance"])
        save_json(checkpoints / f"support_{accepted}.json", support_json(support)); np.save(checkpoints / f"Pi_{accepted}.npy", sip)
        row = {"accepted_exchanges": accepted, "attempted_exchanges": attempted, **values,
               "validation_max": max(v for v in validation.values() if isinstance(v, (int, float)))}
        rows = pd.concat((rows.loc[rows.accepted_exchanges != accepted] if len(rows) else rows, pd.DataFrame([row])), ignore_index=True)
        rows.to_csv(target_dir / "distance.csv", index=False)
    rows = rows.sort_values("accepted_exchanges")
    endpoint = rows.iloc[-1]; farthest = rows.loc[rows.RMSE_rho.idxmax()]
    selected = endpoint if endpoint.RMSE_rho >= farthest.RMSE_rho * (1 - 1e-6) else farthest
    selected_at = int(selected.accepted_exchanges); target = np.load(checkpoints / f"Pi_{selected_at}.npy")
    np.save(target_dir / "Pi_star.npy", target); save_json(target_dir / "target_design_AUDIT_ONLY.json", support_json(_support(checkpoints / f"support_{selected_at}.json")))
    metadata = {"N": 281, "n": 15, "sip_pairs": 39_340, "sum_pi": float(pi.sum()), "min_pi": float(pi.min()), "max_pi": float(pi.max()),
                "target_seed": config["target_seed"], "selected_checkpoint": selected_at, "selection": "largest RMSE_rho from original-order baseline; endpoint selected when within 1e-6 relative", "created_UTC": utc()}
    save_json(frozen, metadata); return metadata


def _directed(root: Path, config: dict, repetition: int, pi: np.ndarray, target: np.ndarray, initial: Support, baseline: np.ndarray) -> dict:
    folder = root / "raw_runs" / f"directed_{repetition:02d}"; result = folder / "result.json"
    if result.exists(): return json.loads(result.read_text())
    folder.mkdir(parents=True, exist_ok=True); seed = _seed("MU284", "directed", repetition)
    outcome = directed_repetition(repetition, seed, initial, target, pi, config["directed_evaluations"], config["directed_batch"], float("inf"), rmse_stop=config["raw_rmse_stop"], stop_inclusive=True)[2]
    values, sip, validation = _metrics(outcome.support, pi, target, baseline, config["validation_tolerance"])
    trace = pd.DataFrame(outcome.checkpoints)
    # Preserve the complete native trace and an explicit cumulative-budget table (no restart between rows).
    trace.to_csv(folder / "convergence_full.csv", index=False)
    saved = []
    for budget in config["directed_checkpoints"]:
        eligible = trace.loc[trace.candidate_moves <= budget]
        row = (eligible.iloc[-1] if len(eligible) else trace.iloc[0]).to_dict(); row["requested_evaluations"] = budget; saved.append(row)
    pd.DataFrame(saved).to_csv(folder / "convergence_checkpoints.csv", index=False)
    np.save(folder / "Pi_hat.npy", sip); save_json(folder / "design.json", support_json(outcome.support)); save_json(folder / "validation.json", validation)
    row = {"Method": "GFS-Directed", "Repetition": repetition, "Seed": seed, "Runtime": outcome.runtime_seconds, "Evaluations": outcome.candidate_moves, "Status": outcome.status, **values}
    save_json(result, row); return row


def _dsd(root: Path, config: dict, pi: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> dict:
    folder = root / "raw_runs" / "dsd"; result = folder / "result.json"
    if result.exists(): return json.loads(result.read_text())
    folder.mkdir(parents=True, exist_ok=True); started = perf_counter(); rng = np.random.default_rng(_seed("MU284", "dsd")); best = None
    pair = np.triu_indices(len(pi), 1)
    for trial in range(config["dsd_trials"]):
        order = np.arange(len(pi)) if trial == 0 else rng.permutation(len(pi)); ordered = projection_factor(pi[order]); V = np.zeros_like(ordered); V[order] = ordered
        sip = np.outer(pi, pi) - (V @ V.T) ** 2; np.fill_diagonal(sip, pi); loss = float(np.sum((sip - target)[pair] ** 2))
        if best is None or loss < best[0]: best = (loss, sip, V)
    _, sip, V = best; dep = DependenceError(pi); residual = (sip-target)[pair]; rho = dep.from_matrices(sip, target); base = dep.from_matrices(baseline, target)
    row = {"Method": "DSD — projection/order fit", "Repetition": 1, "Seed": _seed("MU284", "dsd"), "Runtime": perf_counter()-started, "Evaluations": config["dsd_trials"], "Status": "completed", "RMSE": float(np.sqrt(np.mean(residual**2))), "RMSE_rho": rho, "Recovery_rho": dep.recovery(rho, base), "Max SIP error": float(np.max(abs(residual))), "FIP max error": float(np.max(abs(np.sum(V*V,axis=1)-pi))), "Support size": np.nan, "Entropy": np.nan, **dep.counts()}
    np.save(folder / "Pi_hat.npy", sip); save_json(result, row); return row


def _original_gfs(root: Path, config: dict, pi: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> dict:
    """Invoke the notebook's existing package GFS+restart wrapper, unchanged."""
    folder = root / "raw_runs" / "original_gfs"; result_path = folder / "result.json"
    if result_path.exists(): return json.loads(result_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    from prescribed_sip.final_study import cells, exec_cell
    notebook = REPO / "prescribed_sip/notebooks/GFS_prescribed_SIP_demo.ipynb"
    source = cells({"notebook": str(notebook)})
    ns = {"__name__": "__mu284_final_notebook__"}
    exec_cell(ns, source["settings"], "settings")
    ns.update(POPULATION="MU284", N=284, n=15, FIP_TYPE="unequal", SEED=_seed("MU284", "original"),
              GFS_MAX_EVALUATIONS=config["original_evaluations"], GFS_RMSE_STOP=config["raw_rmse_stop"],
              GFS_AUTO_RESTART=True, GFS_STAGNATION_EVALUATIONS=None, ALLOW_LONG_RUN=True)
    for tag in ("setup", "population", "helpers", "initial"):
        exec_cell(ns, source[tag], tag)
    # This is metric plumbing only; no target walk or target distribution is exposed.
    ns["Pi_star"] = target
    exec_cell(ns, source["target_plots"].split("pair_difference =")[0], "metric_helpers")
    exec_cell(ns, source["search_helpers"], "search_helpers")
    started = perf_counter(); response = ns["run_gfs"](config["original_evaluations"], verbose=False, max_seconds=config["original_time_seconds"])
    support = extract_support(response["design"]); values, sip, validation = _metrics(support, pi, target, baseline, config["validation_tolerance"])
    trace = response["trace"]; trace.to_csv(folder / "convergence_full.csv", index=False); response["restart_history"].to_csv(folder / "restarts.csv", index=False)
    np.save(folder / "Pi_hat.npy", sip); save_json(folder / "design.json", support_json(support)); save_json(folder / "validation.json", validation)
    row = {"Method": "Original GFS + restarts", "Repetition": 1, "Seed": ns["SEED"], "Runtime": response["runtime"], "Evaluations": response["evaluations"], "Status": response["status"], "Wall seconds": perf_counter()-started, **values}
    save_json(result_path, row); return row


def _plot(root: Path, target: np.ndarray, rows: pd.DataFrame) -> None:
    plots = root / "plots"; plots.mkdir(exist_ok=True); directed = rows.loc[rows.Method == "GFS-Directed"]
    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"): fig.savefig(plots / f"{name}.{ext}", dpi=300)
        plt.close(fig)
    fig, ax = plt.subplots(); ax.bar(rows.Method, rows.RMSE_rho); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("RMSE rho"); fig.tight_layout()
    save(fig, "method_comparison")
    fig, ax = plt.subplots(); im=ax.imshow(target,cmap="viridis"); fig.colorbar(im,ax=ax); ax.set_title("Frozen target Pi_star"); save(fig, "target_heatmap")
    if len(directed):
        best = directed.loc[directed.RMSE_rho.idxmin()]; reconstructed = np.load(root / "raw_runs" / f"directed_{int(best.Repetition):02d}" / "Pi_hat.npy")
        err = reconstructed-target; lim=max(float(np.max(abs(err))), 1e-18)
        fig, ax = plt.subplots(); im=ax.imshow(reconstructed,cmap="viridis"); fig.colorbar(im,ax=ax); ax.set_title("Best Directed reconstructed Pi"); save(fig,"directed_heatmap")
        fig, ax = plt.subplots(); im=ax.imshow(err,cmap="RdBu_r",vmin=-lim,vmax=lim); fig.colorbar(im,ax=ax); ax.set_title("Directed minus target SIP"); save(fig,"directed_minus_target_heatmap")
        pair=np.triu_indices(target.shape[0],1); fig,ax=plt.subplots(); ax.scatter(target[pair],reconstructed[pair],s=3,alpha=.4); hi=max(target[pair].max(),reconstructed[pair].max()); ax.plot([0,hi],[0,hi],"k--"); ax.set_xlabel("Target SIP"); ax.set_ylabel("Directed SIP"); save(fig,"target_vs_directed_scatter")
        traces=[]
        for rep in directed.Repetition:
            p=root/"raw_runs"/f"directed_{int(rep):02d}"/"convergence_checkpoints.csv"
            if p.exists(): traces.append(pd.read_csv(p))
        if traces:
            fig,ax=plt.subplots()
            for trace in traces: ax.plot(trace.requested_evaluations,trace.incumbent_RMSE_rho,alpha=.25,color="C0")
            grid=traces[0].requested_evaluations; median=np.median([t.incumbent_RMSE_rho.to_numpy() for t in traces],axis=0); ax.plot(grid,median,color="C0",lw=2,label="median"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("candidate evaluations"); ax.set_ylabel("RMSE rho"); ax.legend(); save(fig,"directed_convergence_rmse_rho")
            base=DependenceError(np.load(root/"target"/"pi.npy")).from_matrices(independent_sip(_support(root/"target"/"initial_design.json")),target)
            fig,ax=plt.subplots()
            for trace in traces: ax.plot(trace.requested_evaluations,100*(1-trace.incumbent_RMSE_rho/base),alpha=.25,color="C1")
            ax.plot(grid,100*(1-median/base),color="C1",lw=2); ax.set_xscale("log"); ax.set_xlabel("candidate evaluations"); ax.set_ylabel("Recovery rho (%)"); save(fig,"directed_convergence_recovery_rho")
        fig,ax=plt.subplots(); ax.hist(directed.RMSE_rho.dropna(),bins=max(1,min(10,len(directed)))); ax.set_xlabel("final RMSE rho"); save(fig,"directed_final_rmse_rho_distribution")
    distance = pd.read_csv(root / "target" / "distance.csv"); fig,ax=plt.subplots(); ax.plot(distance.accepted_exchanges,distance.RMSE_rho,"o-"); ax.set_xscale("log"); ax.set_xlabel("accepted exchanges"); ax.set_ylabel("distance RMSE rho from baseline"); save(fig,"target_generation_distance")


def execute(output: Path, config: dict) -> None:
    output = Path(output)
    for name in ("config", "target", "raw_runs", "tables", "plots", "validation"): (output / name).mkdir(parents=True, exist_ok=True)
    config_file = output / "config" / "study.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != config: raise RuntimeError("output already belongs to a different MU284 configuration")
    else: save_json(config_file, config)
    metadata = prepare_target(output, config)
    pi = np.load(output / "target" / "pi.npy"); target = np.load(output / "target" / "Pi_star.npy"); initial = _support(output / "target" / "initial_design.json"); baseline = independent_sip(initial)
    initial_values, _, initial_validation = _metrics(initial, pi, target, baseline, config["validation_tolerance"])
    rows = [{"Method": "Initial GFS baseline", "Repetition": 1, "Seed": config["target_seed"], "Runtime": 0., "Evaluations": 0, "Status": "completed", **initial_values}]
    rows.append(_original_gfs(output, config, pi, target, baseline))
    rows += [_directed(output, config, r, pi, target, initial, baseline) for r in range(1, config["directed_repetitions"] + 1)]
    rows.append(_dsd(output, config, pi, target, baseline))
    frame = pd.DataFrame(rows); frame.to_csv(output / "tables" / "all_runs.csv", index=False)
    save_json(output / "validation" / "initial.json", initial_validation); save_json(output / "validation" / "target_selection.json", metadata)
    _plot(output, target, frame)
    (output / "README.md").write_text("# MU284 prescribed-SIP final study\n\n"
        f"Frozen target checkpoint: {metadata['selected_checkpoint']}. N=281, n=15, 39,340 SIP pairs.\n\n"
        "CP(2) enumerated fit and Exact LP/QP enumeration are intentionally not run: this is an implementation/enumeration limit at C(281,15), not a theoretical limitation of CP(2).\n"
        "The target audit distribution is separate from reconstruction inputs; Directed reconstructions receive only pi, n, and Pi_star.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True, type=Path); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with execution_lock(args.output): execute(args.output, study_config(args.smoke))


if __name__ == "__main__": main()
