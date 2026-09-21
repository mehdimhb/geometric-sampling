"""Handchecked aggregate fixtures catch omitted failures and extrapolated traces."""
import json

import numpy as np
import pandas as pd
import pytest


def test_summary_keeps_all_methods_and_counts_strict_thresholds_and_failures():
    from prescribed_sip.experiments.n12_artifacts import summarize

    # At 1e-2 the boundary value fails the strict threshold; the failed run
    # remains in the denominator. No optimizer or target builder is involved.
    rows = [
        dict(case="E", method="Original GFS", status="budget_exhausted",
             final_RMSE=.01, final_D2=.0066, final_Dinf=.02,
             recovery_percent=50., runtime_seconds=2., candidate_evaluations=100),
        dict(case="E", method="Original GFS", status="completed",
             final_RMSE=.002, final_D2=.000264, final_Dinf=.004,
             recovery_percent=90., runtime_seconds=4., candidate_evaluations=200),
        dict(case="E", method="Original GFS", status="failed",
             final_RMSE=None, runtime_seconds=6., candidate_evaluations=300),
        dict(case="U", method="Sinha", status="NA — theoretical restrictions not satisfied"),
    ]
    result = summarize(rows)
    assert len(result) == 14
    row = result[(result.case == "E") & (result.method == "Original GFS")].iloc[0]
    assert row.runs == 3 and row.valid_results == 2
    assert row.final_RMSE_mean == pytest.approx(.006)
    assert row.final_RMSE_median == pytest.approx(.006)
    assert row.final_RMSE_IQR == pytest.approx(.004)
    assert row.final_RMSE_SD == pytest.approx(.005656854249492381)
    assert row.recovery_percent_median == 70
    assert row.runtime_seconds_median == 4
    assert row.candidate_evaluations_mean == 200
    assert row.threshold_1e_02_count == 1
    assert row.threshold_1e_02_proportion == pytest.approx(1 / 3)
    assert row.threshold_1e_03_count == 0
    sinha = result[(result.case == "U") & (result.method == "Sinha")].iloc[0]
    assert np.isnan(sinha.final_RMSE_median)
    assert "theoretical restrictions" in sinha.status_counts


def test_convergence_incumbents_do_not_extend_past_shortest_run():
    from prescribed_sip.experiments.n12_artifacts import convergence_summary

    rows = [dict(case="E", method="Original GFS", replication=0, trace=[
        dict(candidates=0, RMSE=.4, elapsed_seconds=0),
        dict(candidates=10, RMSE=.2, elapsed_seconds=1),
        dict(candidates=20, RMSE=.3, elapsed_seconds=2)]),
        dict(case="E", method="Original GFS", replication=1, trace=[
        dict(candidates=0, RMSE=.6, elapsed_seconds=0),
        dict(candidates=5, RMSE=.1, elapsed_seconds=.5),
        dict(candidates=10, RMSE=.05, elapsed_seconds=1)])]
    result = convergence_summary(rows, "candidates")
    assert list(result.candidates) == [0, 5, 10]
    assert list(result.runs_represented) == [2, 2, 2]
    assert list(result.RMSE_median) == pytest.approx([.5, .25, .125])
    assert result.iloc[-1].RMSE_q25 == pytest.approx(.0875)
    assert result.iloc[-1].RMSE_q75 == pytest.approx(.1625)


def test_dsd_order_fitting_is_a_stochastic_twenty_run_comparison():
    from prescribed_sip.experiments.n12_artifacts import summarize, convergence_summary

    rows = [dict(case="E", method="DSD", replication=0, status="completed",
                 final_RMSE=.01, trace=[dict(candidates=0, RMSE=.02),
                                       dict(candidates=10, RMSE=.01)])]
    summary = summarize(rows)
    dsd = summary[(summary.case == "E") & (summary.method == "DSD")].iloc[0]
    assert dsd.expected_runs == 20
    curves = convergence_summary(rows)
    assert list(curves.method) == ["DSD", "DSD"]


def test_saved_support_audit_detects_a_corrupted_sip_without_normalizing(tmp_path):
    from prescribed_sip.experiments.n12_artifacts import _audit_support

    (tmp_path / "design.json").write_text(json.dumps(dict(
        N=12, n=4, samples=[list(range(4)), list(range(4, 8)), list(range(8, 12))],
        probabilities=[1/3, 1/3, 1/3])))
    sip = np.zeros((12, 12))
    sip[:4, :4] = 1/3
    sip[4:8, 4:8] = 1/3
    sip[8:, 8:] = 1/3
    sip[0, 1] = .25
    np.save(tmp_path / "sip.npy", sip)
    audit = _audit_support(tmp_path, "fixture", "design.json", np.full(12, 1/3), "sip.npy")
    assert audit["audit_status"] == "failed"
    assert audit["max_saved_SIP_discrepancy"] == pytest.approx(1/12)
    assert audit["max_FIP_error"] == 0
    assert audit["probability_sum_error"] == 0


def test_full_trace_plot_preserves_each_gfs_endpoint_without_extrapolating():
    import matplotlib.pyplot as plt
    from prescribed_sip.experiments.n12_artifacts import plot_full_traces

    rows = [dict(case="E", method="Original GFS", trace=[
        dict(candidates=0, RMSE=.4, elapsed_seconds=0),
        dict(candidates=868, RMSE=.2, elapsed_seconds=2)]),
        dict(case="E", method="Original GFS high precision", trace=[
        dict(candidates=0, RMSE=.6, elapsed_seconds=0),
        dict(candidates=2500, RMSE=.1, elapsed_seconds=5),
        dict(candidates=5000, RMSE=.05, elapsed_seconds=10)])]
    fig, ax = plt.subplots()
    plot_full_traces(ax, rows, "E", "candidates")
    assert [list(line.get_xdata()) for line in ax.lines] == [[0, 868], [0, 2500, 5000]]
    assert list(ax.get_xlim()) == [0, 5000]
    assert [list(line.get_ydata()) for line in ax.lines] == [[.4, .2], [.6, .1, .05]]
    plt.close(fig)
    fig, ax = plt.subplots()
    plot_full_traces(ax, rows, "E", "elapsed_seconds")
    assert [line.get_xdata()[-1] for line in ax.lines] == [2, 10]
    plt.close(fig)


def test_plateau_does_not_treat_roundoff_at_budget_end_as_progress():
    from prescribed_sip.experiments.n12_artifacts import _plateau_table

    rows = [dict(case="E", method="Original GFS", replication=0,
                 candidate_evaluations=5000, status="budget_exhausted",
                 details={"last_improvement_candidate": 100}, trace=[
                     dict(candidates=0, RMSE=.4),
                     dict(candidates=100, RMSE=.1),
                     dict(candidates=1000, RMSE=.10000000000000002),
                     dict(candidates=5000, RMSE=.09999999999999998)])]
    result = _plateau_table(rows).iloc[0]
    assert result.last_recorded_improvement_candidates == 100
    assert result.actual_incumbent_last_improvement_candidates == 100
    assert result.late_relative_RMSE_improvement == 0


def test_generate_exports_aggregate_numbers_and_all_figure_families(tmp_path):
    from prescribed_sip.experiments.n12_artifacts import generate

    for directory in ("target_equal", "target_unequal"):
        folder = tmp_path / directory
        folder.mkdir()
        pi = np.full(12, 1 / 3)
        initial = np.full((12, 12), 1 / 11)
        np.fill_diagonal(initial, pi)
        for name, matrix in (("Pi0", initial), ("Pistar", initial),
                             ("pi", pi), ("x", np.ones(12))):
            np.save(folder / (name + ".npy"), matrix)
        (folder / "metadata.json").write_text(json.dumps(dict(
            H0=1., Hstar=2., support0=3, supportstar=6,
            selected_exchanges=50, initial_RMSE=.02, d_rho=.1,
            validation={"max_FIP_error": 1e-15})))
        pd.DataFrame([dict(exchanges=0, entropy=1., support=3,
                          effective_support=2.7, RMSE_from_initial=0, d_rho=0),
                      dict(exchanges=50, entropy=2., support=6,
                           effective_support=7.4, RMSE_from_initial=.02, d_rho=.1)]
                     ).to_csv(folder / "trajectory.csv", index=False)
    (tmp_path / "raw_runs").mkdir()
    (tmp_path / "raw_runs/one.json").write_text(json.dumps(dict(
        case="E", method="Original GFS", replication=0, seed=1,
        status="completed", final_RMSE=.002, final_D2=.000264,
        final_Dinf=.004, recovery_percent=90., runtime_seconds=4.,
        candidate_evaluations=200, validation={}, details={}, trace=[])))
    generate(tmp_path)
    main = pd.read_csv(tmp_path / "tables/main_results.csv")
    assert len(main) == 7
    assert main.loc[main.method == "Original GFS", "E_median_RMSE"].iloc[0] == .002
    for family in "ABCDEF":
        for suffix in ("pdf", "png"):
            assert (tmp_path / f"figures/figure_{family}.{suffix}").stat().st_size > 1000
    for name in ("main_results", "supplementary_results", "threshold_counts"):
        for suffix in ("csv", "md", "tex"):
            assert (tmp_path / f"tables/{name}.{suffix}").exists()
    assert (tmp_path / "reports/N12_n4_prescribed_SIP_report.md").exists()
    assert (tmp_path / "reports/N12_n4_paper_ready_results.md").exists()
