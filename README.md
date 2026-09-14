# TSAlignment.jl

Goal Programming time-series alignment for wearable physiological signals (e.g. heart
rate from multiple devices). It provides three Goal Programming alignment models
(EGP, RGP, ERMCGP), a hierarchical recursive solver for long series, and
multi-series weighted-consensus fusion — plus the metrics and plots used in the
accompanying paper.

## Requirements

- **Julia** ≥ 1.9
- A **JuMP-compatible MILP optimizer**. This package was built and validated
  **exclusively with [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)**
  (via `CPLEX.jl`), which is a licensed solver. Any JuMP MILP optimizer *may* work,
  but only CPLEX has been tested and is the solver the reported results were obtained
  with. The fusion/consensus methods (uniform and weighted mean) need **no** optimizer.

The package deliberately does **not** depend on CPLEX in `Project.toml` (so it stays
installable without a license); you register the optimizer yourself at runtime.

## Installation

Dependencies are declared in `Project.toml`, so it installs like any Julia package.
From the package root:

```julia
] activate .
] instantiate
# ] add CPLEX     # if you have a CPLEX license
```

(Or run `julia install.jl`, which does the same non-interactively.) To use it from
another project without registering it, `] dev /path/to/TSAlignment.jl` or
`] add https://github.com/USER/TSAlignment.jl`. For `] add TSAlignment` by name, the
package must be registered in a Julia registry (the General registry, or a private
one via `LocalRegistry.jl`).

## Quick start

```julia
using TSAlignment
using CPLEX
TSAlignment.set_optimizer!(CPLEX.Optimizer)     # required before any GP solve

# align B onto A with one GP variant (hierarchical solver, for long series)
res = solve_recursive(A, B; model = :egp)       # :egp | :rgp | :ermcgp
res.mae, res.r2, res.dw, res.pairs

# all variants in one call, as a table
results = compare_variants(A, B; models = [:egp, :rgp, :ermcgp])
metrics_table(results; N = length(A))
```

## Multi-series consensus (fusion)

```julia
series, names = read_many(["A_common.csv", "B_rad7_2.csv", "B_rad7_3.csv", "B_b2b.csv"];
                          names = ["rad7_1","rad7_2","rad7_3","b2b"])

# Iterative Weighted Consensus (GP-based): learns per-series reliability weights,
# so a corrupt device is downweighted automatically.
w = weighted_align(series; names = names, model = :egp, weight = :robust, iters = 4)
w.consensus, w.weights, w.fit

# One-shot consensus builders (no optimizer):
cu = cons_uniform(series)
cw = cons_weighted(series, rel_weights(series))

# Leave-one-out fusion evaluation (uniform mean vs weighted inverse-variance):
X = Dict(zip(names, series))
lo = fusion_loo(X; folds = ["rad7_1","rad7_2","b2b"])   # rad7_3 stays in training
lo.summary
```

The weighting is an adaptation of classical **inverse-variance (information) weighting**
(weight ∝ 1/variance); what is specific here is that the per-series scale σ_k is the
dispersion of that series' residual to the consensus, plus a small regularizer so no
series is divided by zero or fully dropped.

## Package layout

```
src/
  config.jl        parameters (Refs) + set_optimizer!   ← solver selection lives here
  models.jl        egp_solve / rgp_solve / ermcgp_solve + MSOLVE
  hierarchical.jl  BandedAlignment + solve_recursive
  metrics.jl       metrics_from_pairs, count_crossings, monotone_repair, fusion_metrics
  fusion.jl        rel_weights, cons_*, weighted_align, fusion_loo
  synthetic.jl     base_signal, gen_gate_case_meta (robustness/sensitivity cases)
  pipeline.jl      compare_variants, metrics_table
  viz.jl           overlay_pairs/aligned, alignment_plot, plot_weights, plot_consensus_overlay
  io.jl            read_series, read_many
examples/          runnable scripts (real pipeline, weighted consensus, synthetic)
test/              runtests.jl (everything that does not need CPLEX)
```

## Reproducibility & parameters

All model parameters are `Ref`s whose defaults are the exact configuration used to
produce the paper's results, so at the defaults the package reproduces the reported
numbers. A sensitivity sweep is just a reassignment:

```julia
TSAlignment.W1[]  = 0.8       # EGP/ERMCGP deviation weight
TSAlignment.PSI[] = 0.4       # RGP restriction strength
TSAlignment.GAMMA[] = 0.15    # ERMCGP excess-deviation weight (γ)
# ... then re-run; call TSAlignment.reset_params!() to restore paper defaults.
```

Tunable knobs: `ALPHA, W1, W2, RHO_A, RHO_B, PSI, M_BIG, GAMMA, E_BOUND, EPS_GUARD`
(model) and `MAX_N, N_SPLIT, P_ANGLE, WMIN, WMAX, KFRAC` (hierarchy).

Preprocessing (clock synchronization, interval filtering, NaN handling) is **out of
scope** for this package — it is device/dataset specific and lives in the experiment
notebooks. TSAlignment assumes series are already co-registered on a common grid.

## Testing

```julia
] activate .
] test
```

The test suite covers everything that does not require an optimizer (metrics,
monotonicity, consensus builders, synthetic generators). GP-solver tests run
only if an optimizer has been registered via `set_optimizer!`.

## Citation

See `CITATION.bib`. 
## License

MIT — see `LICENSE`.
