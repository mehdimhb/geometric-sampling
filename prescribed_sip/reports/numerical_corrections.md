# Numerical defects found before extensive execution

## Certainty-unit graphical initialization

The pre-fix small-N pilot stopped two Advanced runs when a randomized graphical
start contained samples of size n-1. Independent inspection traced this to units
with FIP exactly1. Their split interval endpoints can differ at floating-point
roundoff, e.g. `.1` versus `1.1-1`. The original event sort then processes their
start before end and removes the certainty unit from a material interval.

Minimal regression: FIP `[.1,1.,.2,.3,.4]`, n=2, line order. The original builder
produces an invalid support; the regression test failed before correction.

The simulation adapter now conditions out exact certainty units, calls the original
builder on the remaining population, and adds those units to every sample. It
restores the full population/order afterward, so existing GBFS order/exchange logic
continues to operate. This is a numerical integration correction, not replacement
of the package optimizer. Original package files are unchanged.

The pre-fix pilot, failure reasons and report are retained in
`results/small_validation_pre_certainty_fix`. Algorithm source snapshots are in
`reports/source_snapshots/pre_certainty_fix` as `.py.txt` files. These data are
validation-history artifacts, not silently pooled into corrected scientific results.

## Exact seed export

Pandas can promote an optional integer seed column with missing values to float64.
For seeds above2^53 this loses digits in CSV. Per-run JSON and target metadata
retained the original exact integers. The regression test demonstrated a CSV value
9007199254740992 instead of9007199254740993 before correction. CSV collection now
converts seed columns to exact decimal strings before dataframe construction.
When reading optional seed columns with pandas, specify `dtype='string'` for
`seed`, `target_seed` and `initial_seed`.

## Other test-discovered corrections

- Sinha inversion coefficient corrected from ambiguous PDF text extraction using
  the independent pair-removal moment identities; a deterministic removed-pair
  example failed before the fix.
- Package heap-copy RNGs isolated in the serial precision adapter; a repeated run
  on the same input design failed reproducibility before the fix.

See `reports/validation.txt` for the most recently executed full test command output.

## Independent review corrections

The QP linear oracle now uses equality dual marginals with a simplex-based correction for negative reduced costs and a conservative floating-point allowance. The oracle primal objective is not used as a lower-bound certificate. Regression tests exercise a deliberately infeasible dual vector.

Convergence curves keep method variants separate and select the evaluation regime; tables retain density, pair structure, budget regime and budget values. Sparse search retains only chosen pairs, while independent full-pair diagnostics use bounded row workspaces after the search. Their runtime is recorded separately. `weighted_RMSE2` and full `RMSE2` have distinct meanings.
