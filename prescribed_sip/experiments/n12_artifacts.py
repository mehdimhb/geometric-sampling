"""Publication artifacts derived solely from the two saved N=12,n=4 problems.

This module never fits a method or generates a target. Counts include failed
attempts, absent methods stay visible, and convergence does not extrapolate
beyond the common observed range of the repetitions.
"""
import json
from math import fsum
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("Initial", "Original GFS", "Original GFS high precision", "CP(2)",
           "Sinha", "DSD", "Exact LP/QP")
GFS_METHODS = METHODS[1:3]
STOCHASTIC = (*GFS_METHODS, "DSD")
CASES = {"E": "Equal FIP", "U": "Unequal FIP"}
TARGET_DIRS = {"E": "target_equal", "U": "target_unequal"}
THRESHOLDS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10)
METRICS = ("final_RMSE", "final_D2", "final_Dinf", "recovery_percent",
           "runtime_seconds", "candidate_evaluations")
CURVE_NOTE = ("Curves show incumbent RMSE medians and interquartile bands. "
              "Within each observed run, the last recorded checkpoint is carried "
              "forward to the union checkpoint grid; this is LOCF, not linear "
              "interpolation. Grids are restricted to the intersection of observed "
              "run ranges, so no run is carried forward after its final checkpoint. "
              "The number of represented runs is saved in convergence CSV files. "
              "This common range may end before the candidate budget; supplementary "
              "full-trace figures show every recorded GFS run through its actual endpoint.")
FULL_TRACE_NOTE = ("Observed incumbent RMSE traces for every Original GFS and "
                   "high-precision repetition, with method-specific colors and alpha=0.25. "
                   "Each line ends at that run's final recorded checkpoint; no interpolation "
                   "or extrapolation is used. These traces reveal budget-ending improvement "
                   "or stagnation after the common-range summary has stopped. "
                   "Recorded plateaus do not establish local optimality.")
RUNTIME_NOTE = ("Deterministic CP(2), exact LP/QP and Sinha fits ran serially before "
                "the worker pool. Stochastic Original GFS, high-precision GFS and DSD "
                "runtimes were measured under contention from up to twelve concurrent "
                "workers. These are descriptive execution costs, not controlled speed "
                "ratios. Manifest execution_seconds measures the comparison and artifact "
                "generation interval after deterministic fitting; it excludes deterministic "
                "fits, target preparation and the timing pilot, so it is not full "
                "end-to-end experiment time.")


def _finite(value):
    try:
        return np.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _usable(row):
    status = str(row.get("status", "")).lower()
    return (_finite(row.get("final_RMSE")) and
            not any(word in status for word in ("fail", "not_applicable", "unavailable"))
            and not status.startswith("na"))


def _threshold_name(threshold):
    return f"threshold_{threshold:.0e}".replace("-", "_")


def summarize(rows):
    """One row for every case/method, with strict threshold counts and all statuses."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    result = []
    for case in CASES:
        for method in METHODS:
            group = [r for r in rows if r.get("case") == case and r.get("method") == method]
            valid = [r for r in group if _usable(r)]
            statuses = {}
            for r in group:
                status = str(r.get("status", "unspecified"))
                statuses[status] = statuses.get(status, 0) + 1
            if not group:
                statuses["NA — no saved run"] = 0
            row = dict(case=case, method=method, runs=len(group), valid_results=len(valid),
                       expected_runs=20 if method in STOCHASTIC else 1,
                       status_counts=json.dumps(statuses, ensure_ascii=False, sort_keys=True))
            for metric in METRICS:
                # Actual time/work is retained even for failed attempts.
                source = group if metric in ("runtime_seconds", "candidate_evaluations") else valid
                values = np.array([float(r[metric]) for r in source if _finite(r.get(metric))])
                stats = {name: np.nan for name in ("mean", "median", "SD", "q25", "q75", "IQR", "min", "max")}
                if len(values):
                    q25, q75 = np.quantile(values, [.25, .75])
                    stats.update(mean=float(values.mean()), median=float(np.median(values)),
                                 SD=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                                 q25=float(q25), q75=float(q75), IQR=float(q75-q25),
                                 min=float(values.min()), max=float(values.max()))
                row.update({metric+"_"+name: value for name, value in stats.items()})
            for threshold in THRESHOLDS:
                count = sum(float(r["final_RMSE"]) < threshold for r in valid)
                name = _threshold_name(threshold)
                row[name+"_count"] = count
                row[name+"_proportion"] = count/len(group) if group else np.nan
            result.append(row)
    return pd.DataFrame(result)


def convergence_summary(rows, axis="candidates"):
    """Incumbent median/IQR on a common-range LOCF grid; never extend short runs."""
    if axis not in ("candidates", "elapsed_seconds"):
        raise ValueError("axis must be candidates or elapsed_seconds")
    columns = ["case", "method", axis, "RMSE_median", "RMSE_q25", "RMSE_q75", "runs_represented"]
    output = []
    for case in CASES:
        for method in STOCHASTIC:
            traces = []
            for row in rows:
                if row.get("case") != case or row.get("method") != method:
                    continue
                points = [p for p in row.get("trace", [])
                          if _finite(p.get(axis)) and _finite(p.get("RMSE"))]
                if not points:
                    continue
                points.sort(key=lambda p: float(p[axis]))
                by_x = {}
                best = np.inf
                for point in points:
                    best = min(best, float(point["RMSE"]))
                    by_x[float(point[axis])] = best
                x = np.array(list(by_x), dtype=float)
                traces.append((x, np.array(list(by_x.values()), dtype=float)))
            if not traces:
                continue
            lower = max(x[0] for x, _ in traces)
            upper = min(x[-1] for x, _ in traces)
            grid = sorted({value for x, _ in traces for value in x if lower <= value <= upper})
            for value in grid:
                y = [values[np.searchsorted(x, value, side="right")-1] for x, values in traces]
                q25, median, q75 = np.quantile(y, [.25, .5, .75])
                output.append(dict(case=case, method=method, **{axis: value}, RMSE_median=median,
                                   RMSE_q25=q25, RMSE_q75=q75, runs_represented=len(y)))
    return pd.DataFrame(output, columns=columns)


def _read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _fmt(value):
    return f"{float(value):.8g}" if _finite(value) else "NA"


def _fmt_recovery(value):
    return "0" if _finite(value) and abs(float(value)) < 1e-9 else _fmt(value)


def _markdown(frame):
    # Avoid an optional tabulate dependency and retain explicit NA cells.
    def cell(value, column):
        if isinstance(value, (int, float, np.number)):
            return _fmt_recovery(value) if "recovery" in column.lower() else _fmt(value)
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines.extend("| " + " | ".join(cell(v, str(column)) for column, v in zip(frame.columns, row)) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def _table(root, name, frame):
    folder = root/"tables"
    frame.to_csv(folder/f"{name}.csv", index=False)
    (folder/f"{name}.md").write_text(_markdown(frame)+"\n")
    if name == "main_results":
        # Group the two cases and use short publication headings. The CSV and
        # supplementary table retain the complete machine-readable statuses.
        lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
                 r"\caption{SIP reconstruction for two fixed feasible $N=12,n=4$ targets.}",
                 r"\label{tab:n12-reconstruction}", r"\resizebox{\textwidth}{!}{%",
                 r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
                 r"& \multicolumn{4}{c}{Equal FIP} & \multicolumn{4}{c}{Unequal FIP} \\",
                 r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
                 r"Method & Median RMSE & Best RMSE & Recovery (\%) & Time (s) & Median RMSE & Best RMSE & Recovery (\%) & Time (s) \\",
                 r"\midrule"]
        for row in frame.to_dict("records"):
            values = [str(row["method"]).replace("%", r"\%").replace("&", r"\&")]
            values += [(_fmt_recovery if metric == "median_Recovery_percent" else _fmt)(row[case+"_"+metric]) for case in CASES
                       for metric in ("median_RMSE", "best_RMSE", "median_Recovery_percent", "median_runtime_seconds")]
            lines.append(" & ".join(values)+r" \\")
        lines += [r"\bottomrule", r"\end{tabular}%", "}",
                  r"\par\footnotesize NA denotes unavailable or inapplicable results; see the supplementary status counts.",
                  r"\end{table}"]
        (folder/f"{name}.tex").write_text("\n".join(lines)+"\n")
    else:
        frame.to_latex(folder/f"{name}.tex", index=False, escape=True, na_rep="NA",
                       float_format=lambda v: f"{v:.8g}", longtable=name == "supplementary_results")


def _targets(root):
    targets = {}
    for case, dirname in TARGET_DIRS.items():
        folder = root/dirname
        metadata = _read_json(folder/"metadata.json", {})
        data = dict(metadata=metadata)
        for name in ("Pi0", "Pistar", "pi", "x"):
            data[name] = np.load(folder/f"{name}.npy", allow_pickle=False)
        if data["Pi0"].shape != (12, 12) or data["Pistar"].shape != (12, 12):
            raise ValueError("N12 artifacts require 12 by 12 SIP matrices")
        data["trajectory"] = pd.read_csv(folder/"trajectory.csv")
        targets[case] = data
    return targets


def _audit_support(root, label, path, pi, sip_path=None):
    """Independently accumulate float64 first and second moments from saved mass."""
    result = dict(design=label, audit_status="NA — no saved support", probability_sum_error=np.nan,
                  min_probability=np.nan, max_FIP_error=np.nan, invalid_samples=np.nan,
                  max_saved_SIP_discrepancy=np.nan, max_row_identity_error=np.nan,
                  pair_sum_identity_error=np.nan)
    if not path or not (root/path).exists():
        return result
    design = _read_json(root/path)
    samples, probabilities = design["samples"], np.asarray(design["probabilities"], dtype=np.float64)
    N, n = int(design["N"]), int(design["n"])
    if len(samples) != len(probabilities) or N != 12 or n != 4 or not len(probabilities):
        result["audit_status"] = "failed — malformed support"
        return result
    fip, sip = np.zeros(N), np.zeros((N, N))
    invalid = 0
    for sample, probability in zip(samples, probabilities):
        valid = (len(sample) == n and len(set(sample)) == n and
                 all(isinstance(i, int) and 0 <= i < N for i in sample))
        if not valid:
            invalid += 1
            continue
        indicator = np.zeros(N)
        indicator[sample] = 1.
        fip += probability*indicator
        sip += probability*np.outer(indicator, indicator)
    pairs = np.triu_indices(N, 1)
    result.update(probability_sum_error=abs(fsum(probabilities)-1.),
                  min_probability=float(np.min(probabilities)), max_FIP_error=float(np.max(abs(fip-pi))),
                  invalid_samples=invalid,
                  max_row_identity_error=float(np.max(abs(sip.sum(axis=1)-np.diag(sip)-(n-1)*pi))),
                  pair_sum_identity_error=abs(float(sip[pairs].sum())-6.))
    if sip_path and (root/sip_path).exists():
        saved = np.load(root/sip_path, allow_pickle=False)
        result["max_saved_SIP_discrepancy"] = float(np.max(abs(sip-saved)))
    good = (np.all(np.isfinite(probabilities)) and result["min_probability"] >= -1e-12 and
            result["probability_sum_error"] <= 1e-10 and result["max_FIP_error"] <= 1e-10 and invalid == 0)
    if _finite(result["max_saved_SIP_discrepancy"]):
        good &= result["max_saved_SIP_discrepancy"] <= 1e-10
    result["audit_status"] = "passed" if good else "failed"
    return result


def _validation(root, rows, targets):
    output = []
    for case, dirname in TARGET_DIRS.items():
        for name, sip in (("p0", "Pi0"), ("pstar", "Pistar")):
            output.append(_audit_support(root, f"{case}/{name}", f"{dirname}/{name}.json",
                                         targets[case]["pi"], f"{dirname}/{sip}.npy"))
    for row in rows:
        case = row["case"]
        result = _audit_support(root, f"{case}/{row['method']}/r{row.get('replication', 0)}",
                                row.get("design_path"), targets[case]["pi"], row.get("sip_path"))
        result["reported_validation"] = json.dumps(row.get("validation", {}), sort_keys=True)
        output.append(result)
    return pd.DataFrame(output)


def _save_figure(root, family, fig):
    fig.savefig(root/f"figures/figure_{family}.pdf", bbox_inches="tight")
    fig.savefig(root/f"figures/figure_{family}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _unavailable(ax, title="No saved numerical result"):
    ax.text(.5, .5, title, ha="center", va="center", transform=ax.transAxes, fontsize=9)


def _plot_curves(ax, data, case, axis):
    for method in STOCHASTIC:
        group = data[(data.case == case) & (data.method == method)]
        if group.empty:
            continue
        x = group[axis].to_numpy(float)
        ax.plot(x, np.maximum(group.RMSE_median.to_numpy(float), 1e-16), label=method)
        ax.fill_between(x, np.maximum(group.RMSE_q25.to_numpy(float), 1e-16),
                        np.maximum(group.RMSE_q75.to_numpy(float), 1e-16), alpha=.2)
    if ax.lines:
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    else:
        _unavailable(ax, "No recorded stochastic checkpoints")
    ax.set_title(CASES[case])
    ax.set_ylabel("Incumbent SIP RMSE")
    ax.set_xlabel("Scored candidate designs" if axis == "candidates" else "Elapsed time (seconds)")
    ax.grid(alpha=.2)


def plot_full_traces(ax, rows, case, axis):
    """Show actual GFS checkpoint endpoints, without common-range truncation."""
    if axis not in ("candidates", "elapsed_seconds"):
        raise ValueError("axis must be candidates or elapsed_seconds")
    for method, color in zip(GFS_METHODS, ("tab:blue", "tab:orange")):
        labeled = False
        for row in rows:
            if row.get("case") != case or row.get("method") != method:
                continue
            points = sorted((point for point in row.get("trace", [])
                             if _finite(point.get(axis)) and _finite(point.get("RMSE"))),
                            key=lambda point: float(point[axis]))
            if not points:
                continue
            x = np.array([float(point[axis]) for point in points])
            y = np.minimum.accumulate([float(point["RMSE"]) for point in points])
            ax.plot(x, np.maximum(y, 1e-16), color=color, alpha=.25, lw=.9,
                    label=method if not labeled else "_nolegend_")
            labeled = True
    if ax.lines:
        ax.set_yscale("log")
        legend = ax.legend(fontsize=7)
        for handle in legend.legend_handles:
            handle.set_alpha(1.)
    else:
        _unavailable(ax, "No recorded GFS checkpoints")
    if axis == "candidates":
        ax.set_xlim(0, 5000)
    ax.set(title=CASES[case], ylabel="Observed incumbent SIP RMSE",
           xlabel="Scored candidate designs" if axis == "candidates" else "Elapsed time (seconds)")
    ax.grid(alpha=.2)


def _figures(root, rows, targets, evaluation_curves, time_curves):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for panels, (case, data) in zip(axes, targets.items()):
        trajectory, metadata = data["trajectory"], data["metadata"]
        for ax, metric, label in zip(panels, ("entropy", "d_rho", "RMSE_from_initial"),
                                     ("Entropy", "Dependence distance d_rho", "SIP RMSE from initial")):
            ax.plot(trajectory.exchanges, trajectory[metric], lw=1.2)
            chosen = metadata.get("selected_exchanges")
            if chosen is not None:
                ax.axvline(chosen, color="darkred", ls="--", lw=1, label=f"Chosen: {chosen}")
                ax.legend(fontsize=7)
            ax.set(xlabel="Accepted random exchanges", ylabel=label, title=f"{CASES[case]}: {label}")
            ax.grid(alpha=.2)
    _save_figure(root, "A", fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    positive_max = max(float(np.max(data[name])) for data in targets.values() for name in ("Pi0", "Pistar"))
    difference_max = max(float(np.max(abs(data["Pistar"]-data["Pi0"]))) for data in targets.values())
    for panels, (case, data) in zip(axes, targets.items()):
        for ax, matrix, title in zip(panels, (data["Pi0"], data["Pistar"], data["Pistar"]-data["Pi0"]),
                                     ("Initial Pi0", "Target Pistar", "Pistar minus Pi0")):
            delta = title.startswith("Pistar minus")
            scale = max(difference_max, 1e-16) if delta else positive_max
            img = ax.imshow(matrix, cmap="RdBu_r" if delta else "viridis",
                            vmin=-scale if delta else 0, vmax=scale)
            ax.set(title=f"{CASES[case]}: {title}", xlabel="Unit", ylabel="Unit")
            fig.colorbar(img, ax=ax, shrink=.75)
    _save_figure(root, "B", fig)
    for family, data, axis in (("C", evaluation_curves, "candidates"), ("D", time_curves, "elapsed_seconds")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, case in zip(axes, CASES):
            _plot_curves(ax, data, case, axis)
        _save_figure(root, family, fig)
    for family, axis in (("C_full_traces", "candidates"), ("D_full_traces", "elapsed_seconds")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, case in zip(axes, CASES):
            plot_full_traces(ax, rows, case, axis)
        _save_figure(root, family, fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, case in zip(axes, CASES):
        values, labels = [], []
        for method in STOCHASTIC:
            group = [float(r["final_RMSE"]) for r in rows if r["case"] == case and r["method"] == method and _usable(r)]
            if group:
                values.append(np.maximum(group, 1e-16))
                labels.append(method)
        if values:
            ax.boxplot(values, tick_labels=labels, showfliers=True, medianprops={"color": "black"})
            for index, group in enumerate(values, 1):
                ax.scatter(np.full(len(group), index), group, s=12, alpha=.5)
        else:
            _unavailable(ax, "No saved stochastic results")
        for method, color in (("Exact LP/QP", "black"), ("CP(2)", "tab:green"), ("Initial", "gray")):
            refs = [float(r["final_RMSE"]) for r in rows if r["case"] == case and r["method"] == method and _usable(r)]
            if refs:
                ax.axhline(max(float(np.median(refs)), 1e-16), color=color, ls="--", lw=1,
                           label=f"{method}: {_fmt(np.median(refs))}")
        ax.set_yscale("log")
        ax.set(title=CASES[case], ylabel="Final SIP RMSE (floor 1e-16 for display)")
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=.2)
    _save_figure(root, "E", fig)
    # Outcome-independent representative = first replication. Best = minimum
    # RMSE, with deterministic replication/seed/path ties. Both are identified.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    selections = []
    pairs = np.triu_indices(12, 1)
    for panels, (case, data) in zip(axes, targets.items()):
        for ax, selection in zip(panels, ("representative", "best")):
            used = False
            upper = max(float(np.max(data["Pistar"][pairs])), 1e-16)
            for method in (*STOCHASTIC, "CP(2)", "Exact LP/QP"):
                group = [r for r in rows if r["case"] == case and r["method"] == method and _usable(r)
                         and r.get("sip_path") and (root/r["sip_path"]).exists()]
                if not group:
                    continue
                tie = lambda r: (int(r.get("replication", 0)), int(r.get("seed") or 0), str(r.get("sip_path", "")))
                chosen = min(group, key=tie if selection == "representative" else lambda r: (float(r["final_RMSE"]), *tie(r)))
                recovered = np.load(root/chosen["sip_path"], allow_pickle=False)
                upper = max(upper, float(np.max(recovered[pairs])))
                label = f"{method} r{chosen.get('replication', 0)}"
                ax.scatter(data["Pistar"][pairs], recovered[pairs], s=13, alpha=.5, label=label)
                selections.append(dict(case=case, selection=selection, method=method,
                                       replication=chosen.get("replication", 0), seed=chosen.get("seed"),
                                       final_RMSE=chosen["final_RMSE"], sip_path=chosen["sip_path"]))
                used = True
            ax.plot([0, upper], [0, upper], color="black", lw=1, label="Identity")
            if not used:
                _unavailable(ax, "No saved reconstructed SIP matrix")
            else:
                ax.legend(fontsize=6)
            ax.set(title=f"{CASES[case]}: {selection}", xlabel="Target pair SIP", ylabel="Recovered pair SIP")
            ax.set_aspect("equal", adjustable="box")
    _save_figure(root, "F", fig)
    pd.DataFrame(selections, columns=["case", "selection", "method", "replication", "seed", "final_RMSE", "sip_path"]
                 ).to_csv(root/"tables/figure_F_selections.csv", index=False)


def _target_table(targets):
    output = []
    pairs = np.triu_indices(12, 1)
    for case, data in targets.items():
        metadata = data["metadata"]
        pi, initial, target = data["pi"], data["Pi0"], data["Pistar"]
        delta = (target-initial)[pairs]
        scale = np.sqrt(pi*(1-pi))
        dependence = delta/np.outer(scale, scale)[pairs]
        row = dict(case=case, H0=metadata.get("H0"), Hstar=metadata.get("Hstar"),
                   support0=metadata.get("support0"), supportstar=metadata.get("supportstar"),
                   selected_exchanges=metadata.get("selected_exchanges"),
                   initial_RMSE=float(np.sqrt(np.mean(delta**2))), d_rho=float(np.sqrt(np.mean(dependence**2))),
                   max_FIP_error=metadata.get("validation", {}).get("max_FIP_error", np.nan),
                   max_row_identity_error=float(np.max(abs(target.sum(axis=1)-np.diag(target)-3*pi))),
                   pair_sum_identity_error=abs(float(target[pairs].sum())-6.))
        for label, values in (("abs_SIP_difference", abs(delta)), ("abs_rho_difference", abs(dependence))):
            row.update({label+"_min": float(np.min(values)), label+"_median": float(np.median(values)),
                        label+"_max": float(np.max(values))})
        row["metadata_initial_RMSE"] = metadata.get("initial_RMSE")
        row["metadata_d_rho"] = metadata.get("d_rho")
        output.append(row)
    return pd.DataFrame(output)


def _main_table(summary):
    output = []
    for method in METHODS:
        row = dict(method=method)
        for case in CASES:
            result = summary[(summary.case == case) & (summary.method == method)].iloc[0]
            for destination, source in (("median_RMSE", "final_RMSE_median"), ("best_RMSE", "final_RMSE_min"),
                                        ("median_Recovery_percent", "recovery_percent_median"),
                                        ("median_runtime_seconds", "runtime_seconds_median")):
                row[case+"_"+destination] = result[source]
            row[case+"_status"] = result.status_counts
        output.append(row)
    return pd.DataFrame(output)


def _plateau_table(rows):
    output = []
    for row in rows:
        if row.get("method") not in STOCHASTIC:
            continue
        points = sorted((p for p in row.get("trace", []) if _finite(p.get("RMSE")) and _finite(p.get("candidates"))),
                        key=lambda p: p["candidates"])
        result = dict(case=row["case"], method=row["method"], replication=row.get("replication", 0),
                      status=row.get("status"), final_candidate_count=row.get("candidate_evaluations"),
                      checkpoints=len(points), last_recorded_improvement_candidates=np.nan,
                      actual_incumbent_last_improvement_candidates=row.get("details", {}).get("last_improvement_candidate", np.nan),
                      late_relative_RMSE_improvement=np.nan)
        if points:
            best, last = np.inf, 0
            for point in points:
                tolerance = max(1e-15, abs(best)*1e-12) if np.isfinite(best) else 0.
                if float(point["RMSE"]) < best-tolerance:
                    best, last = float(point["RMSE"]), point["candidates"]
            cutoff = .8*float(points[-1]["candidates"])
            earlier = [p for p in points if float(p["candidates"]) <= cutoff]
            if earlier:
                previous = min(float(p["RMSE"]) for p in earlier)
                difference = previous-best
                if abs(difference) <= max(1e-15, abs(previous)*1e-12):
                    difference = 0.
                result["late_relative_RMSE_improvement"] = difference/previous if previous > 0 else 0.
            result["last_recorded_improvement_candidates"] = last
        output.append(result)
    return pd.DataFrame(output)


CAPTIONS = {
    "A": "Target construction: entropy, standardized pair-dependence distance, and SIP RMSE from the initial design over accepted random exchanges. Dashed lines mark the frozen target snapshot. The target walk is independent of the later reconstruction objective.",
    "B": "Initial and target SIP matrices and their difference for equal and unequal FIP. Initial/target color scales are shared across cases, as are symmetric difference scales. Diagonals contain FIP.",
    "C": "Reconstruction RMSE against scored candidate evaluations, with median and interquartile bands over Original GFS repetitions, the high-precision sensitivity, and restricted DSD projection/order fitting. GFS counts scored sampling designs; DSD counts scored projection/order kernels. "+CURVE_NOTE,
    "D": "Reconstruction RMSE against recorded elapsed runtime, with median and interquartile bands. Runtime is measured under the saved parallel execution settings. "+CURVE_NOTE,
    "E": "Final stochastic RMSE distributions, individual runs, and deterministic reference levels. Exact LP/QP is an enumerated numerical oracle. Values below 1e-16 are floored only for log-scale display; tabulated values retain full precision.",
    "F": "Target versus recovered SIP for all 66 unordered pairs. Representative runs use the lowest saved replication, with seed/path tie breaks, independently of performance. Best runs minimize final RMSE with the same tie breaks. Panels and legends identify selection and replication; figure_F_selections.csv records exact choices. Deterministic methods appear in both selection panels when available.",
}


def _json_block(value):
    return "```json\n"+json.dumps(value, indent=2, ensure_ascii=False, default=str)+"\n```"


def _results_paragraph(summary):
    sentences = []
    for case in CASES:
        primary = summary[(summary.case == case) & (summary.method == "Original GFS")].iloc[0]
        high = summary[(summary.case == case) & (summary.method == "Original GFS high precision")].iloc[0]
        oracle = summary[(summary.case == case) & (summary.method == "Exact LP/QP")].iloc[0]
        cp = summary[(summary.case == case) & (summary.method == "CP(2)")].iloc[0]
        dsd = summary[(summary.case == case) & (summary.method == "DSD")].iloc[0]
        sentences.append(f"For {CASES[case].lower()}, Original GFS had median RMSE {_fmt(primary.final_RMSE_median)} "
                         f"(best {_fmt(primary.final_RMSE_min)}; median recovery {_fmt_recovery(primary.recovery_percent_median)}%), "
                         f"and {primary[_threshold_name(1e-10)+'_count']}/{int(primary.runs)} saved runs attained RMSE<1e-10. "
                         f"The high-precision sensitivity had median RMSE {_fmt(high.final_RMSE_median)}; "
                         f"the oracle achieved {_fmt(oracle.final_RMSE_min)}. "
                         f"CP(2) and DSD median RMSE were {_fmt(cp.final_RMSE_median)} and {_fmt(dsd.final_RMSE_median)}.")
    return " ".join(sentences)


def _reports(root, rows, targets, summary, main, diagnostics, validation, plateau):
    machine = _read_json(root/"machine.json", {})
    benchmark = _read_json(root/"benchmark.json", {})
    manifest = _read_json(root/"manifest.json", {})
    lines = ["# Fixed N12 n4 prescribed-SIP report", "",
             "Exactly two fixed reconstruction problems are represented: N=12, n=4, 495 enumerated samples and all 66 unordered pair SIP. "
             "Twenty independently seeded optimization repetitions per case for each Original GFS criterion and DSD order fitting are prescribed. "
             "Saved-run coverage below distinguishes actual rows from expected repetitions. No N>12 results are included.", "",
             "D2=sum over i<j of squared SIP residuals; RMSE=sqrt(D2/66); D_inf=max absolute pair residual. "
             "Recovery %=100[1-RMSE/RMSE(Pi0,Pistar)]. Negative recovery is retained. All threshold comparisons are strict (<).", "",
             "## Frozen targets", "", _markdown(diagnostics), ""]
    for case, data in targets.items():
        lines += [f"### {CASES[case]}", "", "Exact saved float64 x vector:",
                  _json_block(data["x"].tolist()), "Exact saved float64 pi vector:",
                  _json_block(data["pi"].tolist()), "Target metadata (including selection and initial/target validation):",
                  _json_block(data["metadata"]), ""]
        trajectory = data["trajectory"]
        chosen = data["metadata"].get("selected_exchanges")
        selected = trajectory[trajectory.exchanges == chosen]
        largest = float(trajectory.d_rho.max())
        ratio = float(selected.d_rho.iloc[0])/largest if len(selected) and largest > 0 else np.nan
        lines += [f"The selected snapshot reaches {_fmt(100*ratio)}% of the largest observed dependence distance. "
                  f"Entropy increase Hstar-H0={_fmt((data['metadata'].get('Hstar') or 0)-(data['metadata'].get('H0') or 0))}. "
                  "Snapshot selection uses target-generation dependence/entropy diagnostics, without fitting reconstruction methods. "
                  "Accepted target-walk counts include valid null exchange transactions when the two samples differ "
                  "by exactly one unit; such a transaction preserves the design without changing its moments.", ""]
    lines += ["## Main results", "", _markdown(main), "",
              "Table caption: median and best final RMSE, median recovery and median recorded runtime for the same frozen "
              "equal/unequal target. NA cells retain absent or inapplicable methods; status counts are explicit. "
              "Runtime/work summaries include failed attempts; accuracy summaries use finite, nonfailed returned results.", "",
              _results_paragraph(summary), "", "## Supplementary results and threshold attainment", "",
              "The complete metric means, medians, sample SD (undefined for one result), IQR, quartiles, minima and maxima "
              "are saved in tables/supplementary_results.csv, .md and .tex. D_inf and runtime/evaluation distributions "
              "are included. Threshold counts/proportions use all saved attempts, including failures, as denominator. "
              "Expected repetitions and actual attempts are both reported. Missing methods have no attainment proportion.", ""]
    coverage = summary[["case", "method", "expected_runs", "runs", "valid_results", "status_counts"]]
    lines += [_markdown(coverage), "", "Threshold counts:", "",
              _markdown(summary[["case", "method", "runs"]+[_threshold_name(t)+"_count" for t in THRESHOLDS]]), "",
              "## Original optimizer and precision sensitivity", "",
              "Original GFS invokes the package's GreedyBestFirstSearch.run and its existing exchange mechanism. "
              "The reconstruction criterion is D2. Nine-decimal D2 comparisons are retained in the main method. "
              "D2 below approximately 5e-10 can round to zero, corresponding to RMSE below "
              "sqrt(5e-10/66)=2.7524094e-6. This is comparison resolution, not a guaranteed RMSE floor: "
              "the frontier heap still orders its raw loss values, and other search decisions may use rounded values. "
              "The high-precision criterion executes the same "
              "search bytecode with score rounding removed; it is a numerical diagnostic rather than a new sampling method. "
              "The stochastic budget is at most 5,000 scored designs and early reconstruction stopping is RMSE<1e-10. "
              "Original GFS uses frontier size 50, ten exchange and two order neighbors, coefficient 0.75, one zone, "
              "a stagnation shock after 100 iterations, backtrack depth zero, and termination after 2,000 scored "
              "designs without incumbent improvement. "
              "The observer returns the best evaluated incumbent, including candidates scored in a batch interrupted "
              "by the candidate budget, using the same Original GFS or high-precision comparison rule as that run. "
              "Exact per-run settings, frontier/exchange parameters and termination details follow.", ""]
    settings = {}
    for row in rows:
        if row["method"] in GFS_METHODS:
            settings.setdefault(f"{row['case']}/{row['method']}", row.get("details", {}))
    lines += [_json_block(settings), "", "## Plateau diagnostics", "",
              "Last recorded improvement uses incumbent checkpoints and requires an RMSE reduction larger than "
              "max(1e-15,abs(previous incumbent RMSE)*1e-12), excluding roundoff-only final recomputation changes. "
              "actual_incumbent_last_improvement_candidates retains the optimizer's recorded "
              "details.last_improvement_candidate separately. Late relative improvement compares the "
              "last checkpoint with the best checkpoint at or before 80% of its candidate count. Checkpoint resolution "
              "limits this diagnosis. Finite-budget stagnation does not prove trapping in a local optimum; continuing "
              "improvement indicates that the tested budget ended before a recorded plateau.", ""]
    if not plateau.empty:
        lines += [_markdown(plateau), ""]
    lines += ["## Competitors and numerical oracle", "",
              "CP(2) is fitted with exact enumeration of the exponential-family partition function, log-partition dual "
              "L-BFGS calibration and mandatory Newton FIP tilt. This calibration differs from Bondesson's published "
              "iterative/Gibbs implementation; exact enumeration and our optimizer are not presented as a reproduction "
              "of that sampling implementation. Intermediate calibration iterates need not preserve prescribed FIP. "
              "Final FIP checks, mandatory calibration time/evaluations and optimizer status are retained in run details.", "",
              "DSD performs projection/order fitting within a restricted kernel family with the same FIP. Its SIP obey "
              "pi_ij=pi_i*pi_j-|K_ij|^2, restricting the pair structure; arbitrary prescribed SIP need not be representable. "
              "The existing projection factor construction is fitted by randomized swaps of the population order, "
              "then rows are restored to original unit identifiers. Twenty repetitions per case use at most 5,000 "
              "scored kernel candidates. This optimizes a restricted family, rather than arbitrary projection kernels. "
              "The final fixed-size design is enumerated as p(s)=det(V_s)^2; its moments and FIP are independently audited.", "",
              "Sinha uses the nonnegative pair-removal mixture. For N=12,n=4 its coefficients are "
              "b_ij=7.5*pi_ij-2.5*(pi_i+pi_j)+1. All coefficients must be nonnegative (up to stated numerical tolerance) "
              "and sum to one. Unsatisfied theoretical restrictions produce NA, without clipping or modifying the method.", "",
              "Exact LP/QP enumerates all 495 fixed-size samples and enforces nonnegative probabilities, unit mass, "
              "prescribed FIP and full SIP equalities. This is an oracle numerical benchmark, not a practical large-N "
              "sampling algorithm. A tiny objective is interpreted together with solver status and independently "
              "measured feasibility residuals; floating-point optimization is not symbolic exact arithmetic.", ""]
    for row in rows:
        if row["method"] not in GFS_METHODS and row["method"] != "Initial":
            lines += [f"### {row['case']} — {row['method']}", "",
                      f"Status: {row.get('status')}; optimum/returned RMSE={_fmt(row.get('final_RMSE'))}; "
                      f"D2 objective={_fmt(row.get('final_D2'))}; D_inf={_fmt(row.get('final_Dinf'))}.",
                      "Solver/calibration/applicability details and reported feasibility residuals:",
                      _json_block(dict(details=row.get("details", {}), validation=row.get("validation", {}))), ""]
    lines += ["### Projection DSD representational restriction", "",
              "Every projection DSD satisfies pi_ij=pi_i*pi_j-|K_ij|^2<=pi_i*pi_j. "
              "A strictly positive target pair covariance therefore prevents exact representation "
              "by any projection DSD, including kernels outside the fitted order family.", ""]
    for case, data in targets.items():
        covariance = (data["Pistar"]-np.outer(data["pi"], data["pi"]))[np.triu_indices(12, 1)]
        positive = covariance[covariance > 1e-12]
        lines += [f"{CASES[case]} target: {len(positive)} of 66 pairs have positive covariance "
                  f"above the 1e-12 numerical tolerance; maximum positive covariance is "
                  f"{float(np.max(positive)) if len(positive) else 0.:.17g}. "
                  "These violations impose a structural obstacle to exact projection-DSD recovery.", ""]
    lines += ["## Independent validation", "", _markdown(validation), "",
              "The artifact audit independently accumulates float64 indicator moments from each saved support without "
              "renormalizing its probabilities. It checks probability >=-1e-12, unit total mass within 1e-10, prescribed "
              "FIP within 1e-10, distinct in-range units and sample size four, complete saved SIP discrepancy <=1e-10, "
              "and both fixed-size SIP identities. Reported incremental/full-SIP discrepancies are retained in the "
              "reported_validation column. A missing saved support is explicitly unaudited. Target matrices are also "
              "checked directly for sum_j!=i pi_ij=3*pi_i and sum_i<j pi_ij=6 in target_diagnostics.", "",
              "## Convergence and figure captions", "", CURVE_NOTE, ""]
    for family, caption in CAPTIONS.items():
        lines += [f"Figure {family}. {caption}", "",
                  f"[PDF](../figures/figure_{family}.pdf) · [300-dpi PNG](../figures/figure_{family}.png)", ""]
    for family, axis in (("C", "candidate evaluations"), ("D", "elapsed runtime")):
        lines += [f"Supplementary Figure {family} full traces. {FULL_TRACE_NOTE} "
                  f"Horizontal axis: {axis}.", "",
                  f"[PDF](../figures/figure_{family}_full_traces.pdf) · "
                  f"[300-dpi PNG](../figures/figure_{family}_full_traces.png)", ""]
    lines += ["## Machine, timing pilot and provenance", "", "Machine and actual worker settings:", _json_block(machine),
              "Timing benchmark and projected runtime:", _json_block(benchmark),
              "Saved manifest/source provenance:", _json_block(manifest), "",
              "Independent repetitions are parallelized with one BLAS/OpenMP thread per process. "
              "Worker counts and timing estimates above come from saved provenance; absent metadata is not estimated.", "",
              RUNTIME_NOTE, "",
              "## Interpretation and limits", "",
              "These are two particular frozen feasible targets and finite-budget runs. Compare equal and unequal "
              "outcomes descriptively; their independently generated targets differ in initial-to-target SIP RMSE, "
              "entropy and dependence, so reconstruction difficulty cannot be attributed causally to FIP inequality. "
              "These two examples cannot isolate an inequality effect across populations. "
              "The oracle establishes numerical attainability within its feasibility tolerance. Failure of a restricted "
              "GFS search to recover a feasible target does not establish infeasibility or a certified local minimum. "
              "Paired rounding sensitivity isolates the comparison criterion only when the saved initializations/seeds "
              "are paired. CP(2) enumeration and LP do not establish large-N practicality; DSD and Sinha have structural "
              "restrictions. The experiment stops here, with no extrapolated N>12 simulation."]
    full = "\n".join(lines)+"\n"
    paper = ["# N12 n4 paper-ready results", "", "## Experiment", "",
             "Two fixed N=12,n=4 designs (495 possible samples) were constructed with prescribed equal or moderately "
             "unequal FIP. Valid target-independent random GFS exchanges produced frozen feasible target SIP. "
             "Target snapshots were chosen from dependence-distance/entropy trajectories. Reconstruction methods "
             "received n, pi and target SIP, without target sample probabilities. All 66 pairs were scored with "
             "D2 and RMSE=sqrt(D2/66). Original GreedyBestFirstSearch.run and its high-precision comparison "
             "sensitivity had a 5,000 scored-design limit and RMSE<1e-10 early stopping; twenty independent saved "
             "seeds per target/criterion were prescribed. DSD projection/order fitting also used twenty independent "
             "repetitions with a 5,000 scored-kernel limit within its restricted family. Actual saved coverage is reported in the full report.", "",
             "## Main results table", "", _markdown(main), "",
             "Table caption. Final SIP reconstruction accuracy, recovery relative to the initial design and recorded "
             "runtime for two frozen feasible targets. Stochastic rows summarize independent optimizer repetitions; "
             "deterministic rows represent one fit. NA denotes unavailable/inapplicable results, with explicit statuses.", "",
             "## Results paragraph", "", _results_paragraph(summary), "",
             RUNTIME_NOTE, "",
             "## Recommended figures and captions", ""]
    for family in ("A", "C", "E", "F"):
        paper += [f"Figure {family}. {CAPTIONS[family]}", "",
                  f"[PDF](../figures/figure_{family}.pdf) · [PNG](../figures/figure_{family}.png)", ""]
    for family, axis in (("C", "candidate evaluations"), ("D", "elapsed runtime")):
        paper += [f"Supplementary Figure {family} full traces. {FULL_TRACE_NOTE} "
                  f"Horizontal axis: {axis}.", "",
                  f"[PDF](../figures/figure_{family}_full_traces.pdf) · "
                  f"[300-dpi PNG](../figures/figure_{family}_full_traces.png)", ""]
    positive_covariances = []
    for case, data in targets.items():
        covariance = (data["Pistar"]-np.outer(data["pi"], data["pi"]))[np.triu_indices(12, 1)]
        positive = covariance[covariance > 1e-12]
        positive_covariances.append(f"{CASES[case].lower()}: {len(positive)} pairs, "
                                    f"maximum {float(np.max(positive)) if len(positive) else 0.:.17g}")
    paper += ["Projection DSD satisfies pi_ij<=pi_i*pi_j for every pair. Both frozen targets "
              "violate this condition with positive covariances ("+"; ".join(positive_covariances)+
              "), making exact recovery impossible for any projection DSD, regardless of order fitting.", ""]
    paper += ["Figures B (SIP matrices) and D (runtime convergence) are recommended as supplementary figures.", "",
              "## Limitations paragraph", "",
              "Two fixed targets measure algorithmic variability for these instances, not population-wide recovery "
              "guarantees. The independent equal/unequal targets have different baseline errors and entropy; their "
              "outcome difference does not identify a causal effect of FIP inequality. Finite-budget plateaus are "
              "not proofs of local optimality. Nine-decimal D2 comparisons can round D2<5e-10 to zero "
              "(RMSE approximately 2.75e-6), while the frontier retains raw-loss ordering; removing rounding is "
              "a sensitivity of the original algorithm. CP(2) uses enumerated "
              "dual L-BFGS calibration plus a FIP tilt, differing from the published iterative/Gibbs implementation. "
              "DSD is a restricted projection benchmark and Sinha requires a nonnegative pair-removal mixture. "
              "The enumerated LP/QP oracle demonstrates numerical feasibility only to solver/audit precision and "
              "does not imply scalable sampling. Final design validity is independently audited from saved support.", ""]
    folder = root/"reports"
    name_full, name_paper = "N12_n4_prescribed_SIP_report.md", "N12_n4_paper_ready_results.md"
    (folder/name_full).write_text(full)
    (folder/name_paper).write_text("\n".join(paper))
    # Mirror only an actual prescribed_sip/results artifact tree, never a test
    # fixture or an arbitrary external directory supplied by a caller.
    if root.parent.name == "results" and root.parent.parent.name == "prescribed_sip":
        public = root.parent.parent/"reports"
        public.mkdir(exist_ok=True)
        prefix = f"../results/{root.name}/"
        (public/name_full).write_text(full.replace("../figures/", prefix+"figures/"))
        (public/name_paper).write_text("\n".join(paper).replace("../figures/", prefix+"figures/"))


def generate(root):
    """Create tables, six combined-panel figure families and numerical reports."""
    root = Path(root)
    for directory in ("tables", "figures", "reports"):
        (root/directory).mkdir(parents=True, exist_ok=True)
    targets = _targets(root)
    rows = []
    for path in sorted((root/"raw_runs").glob("*.json")):
        saved = _read_json(path)
        rows.extend(saved if isinstance(saved, list) else [saved])
    if any(row.get("case") not in CASES or row.get("method") not in METHODS for row in rows):
        raise ValueError("Raw runs must contain only the seven prescribed methods and cases E/U")
    summary = summarize(rows)
    main = _main_table(summary)
    diagnostics = _target_table(targets)
    validation = _validation(root, rows, targets)
    plateau = _plateau_table(rows)
    summary.to_csv(root/"summary.csv", index=False)
    for name, frame in (("main_results", main), ("supplementary_results", summary),
                        ("target_diagnostics", diagnostics), ("validation_checks", validation)):
        _table(root, name, frame)
    threshold_columns = ["case", "method", "expected_runs", "runs", "valid_results", "status_counts"]
    threshold_columns += [name for t in THRESHOLDS for name in (_threshold_name(t)+"_count", _threshold_name(t)+"_proportion")]
    _table(root, "threshold_counts", summary[threshold_columns])
    plateau.to_csv(root/"tables/plateau_diagnostics.csv", index=False)
    evaluation_curves = convergence_summary(rows, "candidates")
    time_curves = convergence_summary(rows, "elapsed_seconds")
    evaluation_curves.to_csv(root/"tables/convergence_by_candidates.csv", index=False)
    time_curves.to_csv(root/"tables/convergence_by_runtime.csv", index=False)
    _figures(root, rows, targets, evaluation_curves, time_curves)
    _reports(root, rows, targets, summary, main, diagnostics, validation, plateau)
    return summary
