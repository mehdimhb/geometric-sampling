#!/usr/bin/env python3
"""Exact lambda sensitivity analysis for the Meuse unequal-IP VarB estimator."""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from package_sampling.utils import inclusion_probabilities


HERE = Path(__file__).resolve().parent
POPULATION_FILE = HERE / "populations" / "real" / "meuse.csv"
DESIGN_DIR = HERE / "best_designs" / "meuse"
REFERENCE_FILE = HERE / "results" / "variance_outputs" / "variance_results_full.csv"
OUTPUT_DIR = HERE / "results" / "meuse_lambda_sensitivity"

DETAIL_CSV = OUTPUT_DIR / "meuse_lambda_sensitivity_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "meuse_lambda_sensitivity_summary.csv"
FIGURE_PNG = OUTPUT_DIR / "meuse_lambda_sensitivity.png"
FIGURE_PDF = OUTPUT_DIR / "meuse_lambda_sensitivity.pdf"

LAMBDAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SAMPLE_SIZES = (5, 10, 20)
DESIGNS = {"NMS": "initial", "GMS": "best"}
TARGET_ROUNDED = {
    ("NMS", 5): 0.39, ("NMS", 10): 0.39, ("NMS", 20): -0.16,
    ("GMS", 5): 0.45, ("GMS", 10): 0.08, ("GMS", 20): 0.17,
}


def load_design(kind, n):
    path = DESIGN_DIR / f"{kind}_design_df_meuse_{n}_pik_up.pkl"
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    return obj.best_design if hasattr(obj, "best_design") else obj


def samples_and_probs(design):
    obj = design.all_samples_and_probs
    samples, probs = obj() if callable(obj) else obj
    samples = np.asarray(samples, dtype=int)
    probs = np.asarray(probs, dtype=float)
    return samples, probs / probs.sum()


def true_design_variance(samples, probs, y, pi):
    """Same exact NHT design variance used by the manuscript notebook."""
    true_total = float(y.sum())
    estimates = np.array([np.sum(y[s] / pi[s]) for s in samples])
    mean_nht = float(np.sum(probs * estimates))
    true_var = float(np.sum(probs * (estimates - true_total) ** 2))
    return true_var, mean_nht, true_total


def component_bootstrap_totals(sample, all_samples, y, pi, coords):
    """Return raw-y and z-imputation totals for every replication sample.

    This is an algebraically vectorized form of the notebook's
    ``var_b_resampling`` implementation. Nearest neighbors use ``argmin`` and
    therefore retain its first-index tie handling.
    """
    sample = np.asarray(sample, dtype=int)
    distances2 = np.sum(
        (coords[sample, None, :] - coords[None, :, :]) ** 2, axis=2
    )
    nearest_position = np.argmin(distances2, axis=0)
    nearest_unit = sample[nearest_position]

    raw_by_unit = y[nearest_unit] / pi
    z_by_unit = y[nearest_unit] / pi[nearest_unit]
    return raw_by_unit[all_samples].sum(axis=1), z_by_unit[all_samples].sum(axis=1)


def weighted_variance(values, probs):
    mean = float(np.sum(probs * values))
    return float(np.sum(probs * (values - mean) ** 2))


def analyse_scenario(design_label, kind, n, population):
    y = population["cadmium"].to_numpy(float)
    coords = population[["x", "y"]].to_numpy(float)
    pi = inclusion_probabilities(population["copper"].to_numpy(float).copy(), n)
    samples, probs = samples_and_probs(load_design(kind, n))
    true_var, mean_nht, true_total = true_design_variance(samples, probs, y, pi)

    expected = np.zeros(len(LAMBDAS))
    for sample_probability, sample in zip(probs, samples):
        raw, z = component_bootstrap_totals(sample, samples, y, pi, coords)
        for i, lam in enumerate(LAMBDAS):
            replicate_totals = lam * raw + (1.0 - lam) * z
            expected[i] += sample_probability * weighted_variance(replicate_totals, probs)

    return pd.DataFrame({
        "design": design_label,
        "n": n,
        "lambda": LAMBDAS,
        "true_variance": true_var,
        "expected_variance_estimate": expected,
        "relative_bias": expected / true_var - 1.0,
        "mean_nht": mean_nht,
        "true_total": true_total,
        "number_of_design_samples": len(samples),
    })


def validate(results):
    reference = pd.read_csv(REFERENCE_FILE)
    reference = reference[(reference["population"] == "meuse") & (reference["mode"] == "up")]
    failures = []
    rows = []
    for design_label, kind in DESIGNS.items():
        for n in SAMPLE_SIZES:
            actual = float(results.query("design == @design_label and n == @n and `lambda` == 0.35")["relative_bias"].iloc[0])
            ref = float(reference.query("design_type == @kind and n == @n")["RB_HybridB_a035"].iloc[0])
            rounded_ok = np.isclose(actual, TARGET_ROUNDED[(design_label, n)], atol=0.005)
            exact_ok = np.isclose(actual, ref, rtol=1e-11, atol=1e-11)
            rows.append((design_label, n, actual, ref, exact_ok, rounded_ok))
            if not (exact_ok and rounded_ok):
                failures.append(rows[-1])
    if failures:
        message = "\n".join(str(row) for row in failures)
        raise RuntimeError("lambda=0.35 validation failed; no outputs were written:\n" + message)
    return rows


def make_figure(results):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10.5, "axes.labelsize": 11.5,
        "axes.titlesize": 12, "legend.fontsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    colors = {5: "#0072B2", 10: "#D55E00", 20: "#009E73"}
    markers = {"NMS": "o", "GMS": "s"}
    styles = {"NMS": "-", "GMS": "--"}
    for design in ("NMS", "GMS"):
        for n in SAMPLE_SIZES:
            data = results.query("design == @design and n == @n").sort_values("lambda")
            ax.plot(data["lambda"], data["relative_bias"], label=f"{design}, n={n}",
                    color=colors[n], linestyle=styles[design], marker=markers[design],
                    markersize=4.2, linewidth=1.7, markevery=1)
    ax.axhline(0, color="0.25", linewidth=1.0, zorder=0)
    ax.axvline(0.35, color="0.4", linewidth=1.1, linestyle=":", zorder=0)
    ax.text(0.35, 0.985, r"$\lambda=0.35$", transform=ax.get_xaxis_transform(),
            ha="right", va="top", color="0.3", rotation=90)
    ax.set(xlabel=r"Hybrid weight $\lambda$", ylabel="Relative bias",
           xlim=(0, 1), xticks=np.arange(0, 1.01, 0.1))
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(ncol=2, frameon=False, loc="best")
    fig.savefig(FIGURE_PNG, dpi=400, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    population = pd.read_csv(POPULATION_FILE)
    results = pd.concat([
        analyse_scenario(label, kind, n, population)
        for label, kind in DESIGNS.items() for n in SAMPLE_SIZES
    ], ignore_index=True)

    checks = validate(results)  # Hard gate: outputs are created only after validation.
    summary = (results.assign(abs_RB=results["relative_bias"].abs())
               .groupby("lambda", as_index=False)["abs_RB"]
               .agg(mean_abs_RB="mean", max_abs_RB="max"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (DETAIL_CSV, SUMMARY_CSV, FIGURE_PNG, FIGURE_PDF):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    results.to_csv(DETAIL_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    make_figure(results)

    print("lambda=0.35 validation: PASSED")
    for design, n, actual, reference, _, _ in checks:
        print(f"  {design:3s} n={n:2d}: computed={actual: .9f}, reference={reference: .9f}")
    print("\nSummary by lambda:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.9f}"))
    print("\nRelative bias by scenario:")
    table = results.pivot(index="lambda", columns=["design", "n"], values="relative_bias")
    print(table.to_string(float_format=lambda x: f"{x:.9f}"))
    print(f"\nBest mean |RB|: lambda={summary.loc[summary.mean_abs_RB.idxmin(), 'lambda']:.2f}")
    print(f"Best max |RB|:  lambda={summary.loc[summary.max_abs_RB.idxmin(), 'lambda']:.2f}")
    print(f"\nDetail CSV:  {DETAIL_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"PNG:         {FIGURE_PNG}")
    print(f"PDF:         {FIGURE_PDF}")


if __name__ == "__main__":
    main()
