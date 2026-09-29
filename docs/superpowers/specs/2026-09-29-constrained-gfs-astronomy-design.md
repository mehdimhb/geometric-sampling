# Constrained GFS for Astronomical Target Selection — Design Specification

**Status:** Proposed for user review  
**Date:** 2026-09-29  
**Repository:** `graphical-sampling`

## 1. Purpose and scientific question

Add a small research prototype that tests whether fixed-size Graphical
Finite-Population Sampling (GFS) segment exchanges can navigate and improve a
sampling design after hard support constraints have been imposed.

The experiment keeps three questions separate:

1. **Feasibility:** does any fixed-size design exist with the prescribed
   first-order inclusion probabilities (FIPs) and hard constraints?
2. **Navigation:** starting from one such design, can local GFS segment
   exchanges move through other feasible designs while preserving the FIPs?
3. **Optimization:** can feasible GFS exchanges improve design entropy or
   second-order inclusion probability (SIP) coverage among allowed pairs?

The central methodological distinction is:

> **The LP establishes existence; GFS provides navigation.**

Linear programming is used only to verify compatibility and obtain one initial
feasible point. It is not presented as the proposed navigation or optimization
method. The reusable code is generic constrained-GFS functionality; astronomy
terminology, artificial coordinates, and scientific interpretation remain in
the example notebook.

This prototype does not claim that all compatible designs are connected by
GFS moves, that hill climbing reaches a global optimum, or that the method
scales to survey-sized astronomical populations.

## 2. Relationship to the existing GFS implementation

The existing `Design._switch` implements the fixed-size Algorithm-4 four-piece
split on the package's graphical heap/order representation. It selects two
support samples, exchanges one unit from each sample's set difference, removes
equal mass from both parents, and assigns that mass to the two exchanged
children. Existing `Design`, `Order`, heap construction, and exchange code will
not be rewritten.

An LP solution is an arbitrary finite distribution over feasible samples and
does not naturally have the package's graphical heap/order provenance. The new
prototype therefore uses an algebraic support representation rather than
manufacturing a misleading `Design` hierarchy. Its elementary transition is
still explicitly a **GFS segment exchange**, with the same four-sample algebra:

\[
A'=(A\setminus\{k\})\cup\{\ell\},\qquad
B'=(B\setminus\{\ell\})\cup\{k\},
\]

where `k in A \ B`, `l in B \ A`, and
`0 < epsilon <= min(p_A, p_B)`. The mass update is

\[
p_A\leftarrow p_A-\varepsilon,\quad
p_B\leftarrow p_B-\varepsilon,\quad
p_{A'}\leftarrow p_{A'}+\varepsilon,\quad
p_{B'}\leftarrow p_{B'}+\varepsilon.
\]

The constrained method adds exactly one scientific rule to this move:

> Reject the proposed GFS exchange whenever either child sample is infeasible.

No projection, marginal repair, resampling, or post-hoc renormalization is
allowed. Identical support samples are merged algebraically, and numerical
dust at or below a declared tolerance is removed only after checking that the
result still satisfies all invariants.

## 3. Package architecture and public concepts

Create `graphical_sampling/constrained.py` as a focused module. It will not
depend on astronomy packages or modify the root package API. Callers import its
supported functions and immutable result records directly from
`graphical_sampling.constrained`.

### 3.1 Constraint representation

- A forbidden pair is normalized to an unordered two-element `frozenset` of
  valid population indices.
- `GroupCapacity` is an immutable record containing a nonempty unit set and a
  nonnegative integer capacity.
- `is_feasible(sample, *, population_size, sample_size, forbidden_pairs,
  group_constraints)` validates index range, uniqueness, fixed size, absence
  of forbidden pairs, and every group capacity.
- Malformed constraints, repeated endpoints, out-of-range units, impossible
  capacities, and invalid samples raise informative `ValueError`s rather than
  being silently repaired.

The implementation accepts standard Python iterables at its boundary and
normalizes samples internally to sorted tuples or `frozenset`s. This makes
duplicate support keys impossible while keeping notebook output readable.

### 3.2 Enumerated feasible space and LP initialization

`enumerate_feasible_samples(...)` enumerates all `n`-subsets of
`range(N)`, retains only feasible samples, and returns a deterministic
lexicographic tuple. `incidence_matrix(...)` returns an `N x M` float matrix
with columns corresponding exactly to that tuple.

`solve_feasible_design(...)` validates that `pi` is finite, lies in `[0, 1]`,
and sums to `n` within tolerance. It solves

\[
p\geq0,\quad \mathbf{1}^{\mathsf T}p=1,\quad Xp=\pi
\]

with `scipy.optimize.linprog(method="highs")`. Its result record contains the
enumerated samples, incidence matrix, solver status/message, and an aggregated
positive-support probability mapping. Solver success is never trusted without
independent checks of nonnegativity, normalization, feasibility, fixed size,
and FIP residuals.

`solve_feasible_design(...)` always returns an immutable `FeasibilityResult`.
For mathematical infeasibility, `success` is false, `design` is `None`, and the
solver status/message are retained. An unexpected solver failure raises
`RuntimeError`. The infeasible path never emits a candidate design. The notebook
reports this outcome and stops the navigation experiments. It does not change
the requested FIPs. A separate diagnostic cell may investigate which
artificial constraint relaxation restores feasibility, preferring constraint
changes over FIP changes, but such a relaxation is clearly labeled and does
not replace the requested experiment silently.

### 3.3 Algebraic constrained design and diagnostics

A constrained design is represented by a copy-on-write
`dict[frozenset[int], float]` from a unique fixed-size sample key to a
nonnegative `float64` probability. Public functions never mutate an input
mapping; a successful exchange returns a new mapping.
The module supplies:

- aggregation and validation of support mass;
- `compute_fip(...)`;
- `compute_sip(...)`;
- `design_entropy(...)` using `-sum(p * log(p))` over positive mass;
- support comparison helpers for shared, added, and removed samples;
- allowed-pair classification and summary statistics.

The SIP matrix is exact with respect to the finite support calculation:

\[
\pi_{k\ell}=\sum_{s:\{k,\ell\}\subseteq s}p_s.
\]

Its diagonal equals the computed FIP vector. Pair summaries use unordered
off-diagonal pairs only and divide them into exactly three categories:

1. physically forbidden pairs, whose SIP must be zero;
2. physically allowed pairs with SIP at or below the numerical zero tolerance;
3. physically allowed pairs with positive SIP.

Group capacity constraints can make additional pairs logically impossible even
when not listed as forbidden pairs. For this experiment, “physically
forbidden pair” means a pair excluded by any supplied hard constraint when
considered within at least one size-`n` completion. Pair classification will
therefore be derived from the enumerated feasible sample space: a pair is
physically allowed if it occurs in at least one feasible sample, and forbidden
otherwise. The notebook will separately identify the three explicitly supplied
forbidden pairs so the terminology is transparent.

## 4. Elementary constrained GFS proposal

`constrained_exchange(...)` performs one auditable Algorithm-4 move:

1. Select two distinct positive-probability support samples uniformly.
2. Select `k` uniformly from `A \ B` and `l` uniformly from `B \ A`. If either
   difference is empty, classify the proposal as structurally rejected.
3. Draw `alpha ~ Uniform(0.1, 0.9)` and set
   `epsilon = alpha * min(p_A, p_B)`.
4. Construct `A'` and `B'` by exchanging `k` and `l`.
5. Reject without changing the design unless both children pass the complete
   feasibility predicate.
6. Otherwise construct the four-term mass update, merge identical keys, and
   canonicalize values whose absolute mass is no larger than the declared
   numerical tolerance to zero.
7. Validate the canonical candidate's total mass, nonnegativity, fixed sample size,
   uniqueness, hard constraints, and FIPs before it can be accepted by a
   search objective.

The proposal result records the two parents, exchanged units, children,
`alpha`, `epsilon`, feasibility status, rejection reason, and candidate design.
This makes it possible to test the move as GFS algebra rather than infer its
identity from an optimizer's behavior.

All random choices come from an explicit `numpy.random.Generator`; no hidden
global generator is used.

## 5. Search and accounting

`run_constrained_gfs(...)` proposes a fixed number of GFS exchanges from the
current design. A proposal is committed only when it is feasible and strictly
improves the selected objective beyond a small objective tolerance. Rejected
proposals leave the current design value-equivalent and unmodified.

The search result records:

- requested and actually proposed exchanges;
- structurally valid proposals;
- feasible proposals;
- infeasible proposals;
- accepted improving proposals;
- rejected proposals, equal to `proposed - accepted`;
- rejection counts by structural, infeasible, and non-improving reason;
- checkpoint objective, entropy, allowed-pair metrics, support size, maximum
  FIP error, and elapsed runtime.

Checkpoints are indexed by proposal count and also record cumulative accepted
moves. This avoids conflating attempted navigation with successful objective
improvements.

The accounting identity is
`proposed = structural_rejected + infeasible + feasible`. Accepted moves are a
subset of feasible proposals; `non_improving = feasible - accepted`, and
`rejected = proposed - accepted`.

### 5.1 Entropy objective

The first run starts from the LP design and accepts a feasible move only when

\[
H(p_{new}) > H(p_{current}) + \tau_H.
\]

It uses seed `12345`, 20,000 proposals, `alpha` in `[0.1, 0.9]`, and regular
checkpoints. Its report compares initial entropy, final GFS entropy, support
change, and an independently computed constrained maximum-entropy benchmark.

### 5.2 Allowed-pair coverage objective

The second run starts from the same LP design, with a fresh generator using
seed `12345`, so it is a separate experiment rather than a continuation of the
entropy run. Its score is

\[
\sum_{(k,\ell)\in\mathcal A}\log(\pi_{k\ell}+10^{-12}),
\]

where `A` contains only physically allowed unordered pairs. Physically
forbidden pairs never enter the score and must remain zero. The result reports
before/after zero allowed SIP count, minimum, median, and mean allowed SIP,
entropy, support overlap/change, maximum FIP error, and the size of the SIP
matrix change.

Strict hill climbing can become trapped at a local optimum and does not imply
that every feasible state is reachable. That behavior is a result to report,
not a condition to conceal with restarts or LP re-optimization.

## 6. Exact constrained maximum-entropy benchmark

For this 210-candidate problem only, solve the concave program over all
feasible samples:

\[
\max_p -\sum_s p_s\log p_s
\quad\text{subject to}\quad
p\geq0,\ \mathbf{1}^{\mathsf T}p=1,\ Xp=\pi.
\]

Use SciPy's installed optimization facilities with stable entropy evaluation
and analytic derivatives where appropriate. Initialize from the verified LP
solution or a feasible interior mixture, remove linearly redundant equality
rows if the solver requires it, and independently validate the returned
probabilities and FIPs. Solver failure is reported explicitly; it does not
invalidate the GFS experiment or trigger a new dependency.

This result is labeled **Exact constrained MaxEnt benchmark**, not “GFS”. The
entropy gap is `H_exact - H_gfs_final`. The benchmark may characterize what is
achievable, but it does not supply navigation moves or replace the GFS final
state.

## 7. Required small experiment

Use exactly:

- `N = 10`, `n = 4`;
- `pi = [0.55, 0.45, 0.50, 0.40, 0.35, 0.35, 0.40, 0.35, 0.35, 0.30]`;
- forbidden pairs `(0, 1)`, `(2, 3)`, `(4, 5)`;
- group `{6, 7, 8}` with capacity `2`;
- GFS seed `12345` and 20,000 proposals per objective.

The notebook explicitly verifies that `sum(pi)` is four within floating-point
tolerance. A read-only design probe performed while writing this specification
found 122 feasible size-four samples, a successful HiGHS solution with nine
positive support samples, and maximum equality residual about
`2.8e-16`. These are preliminary observations only; the executed notebook and
tests recompute and validate them from the production implementation.

The initial report includes support size, achieved FIPs, maximum FIP error,
entropy, exact SIP matrix, number of physically forbidden pairs, number of
allowed zero-SIP pairs, and minimum/mean allowed SIP. Final reports add median
allowed SIP, support intersection/additions/removals, GFS proposal accounting,
and a direct assertion that the FIP vector is unchanged while the SIP matrix
has changed whenever at least one nontrivial move was accepted.

## 8. Notebook and figures

Create and execute
`simulations/jupyters/constrained_gfs_astronomy_demo.ipynb`. It is written for
a statistician and contains these concise sections:

1. Motivation
2. Population and prescribed FIPs
3. Artificial fibre constraints
4. Exact feasibility check
5. Initial feasible design
6. Constrained GFS exchange
7. Entropy optimization
8. Pairwise-inclusion optimization
9. Results
10. Short conclusions

The notebook imports reusable package code rather than redefining the exchange
or solver. Its outputs include:

- a labeled 2-D artificial target field with the three explicit forbidden
  pairs connected by dashed lines;
- entropy versus accepted GFS moves, with proposal checkpoints available in
  the underlying data;
- SIP heatmaps before and after pairwise optimization, using a common color
  scale and visibly distinguishing hard-forbidden cells;
- a compact comparison table for the initial LP design, final entropy-GFS
  design, final pair-GFS design, and exact MaxEnt benchmark where applicable;
- proposal/feasibility/acceptance/rejection counts and elapsed runtime;
- initial-versus-final support overlap and FIP/SIP change diagnostics.

The committed notebook is the fully executed default run with outputs. It must
execute top-to-bottom from a fresh kernel. Tests may execute a temporary copy
with reduced proposal counts to keep the suite fast; the production notebook
retains 20,000 proposals in each experiment.

## 9. Tests and invariant validation

Implementation follows test-driven development. Focused tests will cover:

- constraint normalization and `is_feasible` for forbidden pairs and group
  capacities;
- deterministic enumeration and incidence-matrix alignment;
- the requested LP problem's feasibility and an intentionally incompatible
  FIP/constraint problem's explicit infeasibility;
- a hand-constructed Algorithm-4 exchange whose exact four mass deltas are
  asserted;
- rejection when either exchanged child is infeasible;
- merging when a child already exists and safe removal of numerical dust;
- reproducible replay with an explicit generator;
- every accepted exchange preserving total mass, target FIPs, fixed sample
  size, unit uniqueness, nonnegativity, forbidden pairs, and group capacities
  to `1e-10` or tighter;
- SIP diagonals matching FIPs and hard-forbidden pair SIPs remaining zero;
- pair taxonomy excluding physically forbidden pairs from the coverage score;
- exact accounting identities among proposed, feasible, accepted, and rejected
  exchanges;
- entropy and pair objectives accepting only strict improvements;
- maximum-entropy benchmark feasibility and entropy dominance over the initial
  and GFS hill-climbing results within solver tolerance;
- notebook structure and a reduced-cost clean-kernel execution.

Before completion, run the focused tests, the entire repository test suite, and
the full default notebook. Any pre-existing unrelated failure is named in the
final report rather than hidden.

## 10. Numerical policy and error handling

- Use `float64` throughout probability calculations.
- Default mass/invariant tolerance is `1e-12`; public validation accepts an
  explicit positive finite tolerance.
- Report zero SIP pairs using a declared `1e-12` threshold.
- Compare invariant residuals against absolute tolerances with zero relative
  tolerance where practical.
- Never clip materially negative probabilities, renormalize a broken design,
  or repair FIPs. Values within numerical dust tolerance may be canonicalized
  to zero before validation; the candidate is rejected if canonicalization
  causes any invariant to exceed tolerance.
- Invalid input raises `TypeError` or `ValueError`; mathematical infeasibility
  is distinguished from solver/runtime failure.
- Objective values that are nonfinite cause rejection with a diagnostic reason.

## 11. Deliverables and repository boundaries

Expected changed files are limited to:

- `graphical_sampling/constrained.py`;
- `tests/test_constrained_gfs.py`;
- `tests/test_constrained_gfs_notebook.py` if notebook execution checks warrant
  a separate file;
- `simulations/jupyters/constrained_gfs_astronomy_demo.ipynb`;
- this specification and the later implementation plan.

No astronomy dependency is added. Existing `Design`, GFS search, max-entropy
modules, and unrelated simulations are left unchanged unless implementation
reveals a small necessary export or compatibility fix, which must be justified
before inclusion. Existing dirty and untracked user files are preserved.

The final handoff reports the changed-file list, exact test commands/results,
executed notebook path, feasibility and numerical simulation results, runtime,
limitations, surprising findings, and a short assessment of whether the
observed navigation and SIP improvement justify pursuing a methodological
paper.

## 12. Interpretation standard

Evidence supporting further work would consist of all invariants remaining at
numerical precision, a nontrivial feasible-proposal rate, accepted moves that
change the support and SIP matrix, and material improvement in entropy and/or
allowed-pair coverage. Failure to reach the MaxEnt benchmark, low acceptance,
or persistent allowed zero-SIP pairs are scientifically useful negative
results and will be reported directly.

The conclusions must say that LP verified existence and supplied the starting
point, while GFS supplied the local, marginal-preserving moves. They must not
claim positivity for physically forbidden pairs, universal reachability,
connectivity, global optimality, or large-scale practicality.
