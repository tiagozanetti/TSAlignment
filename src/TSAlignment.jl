module TSAlignment

# ----------------------------------------------------------------------------------
# TSAlignment.jl — Goal Programming time-series alignment (EGP / RGP / ERMCGP), a
# hierarchical recursive solver, multi-series weighted-consensus fusion, and the
# metrics / plots used in the accompanying paper.
#
# SOLVER: the Goal Programming models need a JuMP MILP optimizer. This package was
# built and validated EXCLUSIVELY with CPLEX; register it once per session:
#     using CPLEX
#     TSAlignment.set_optimizer!(CPLEX.Optimizer)
# ----------------------------------------------------------------------------------

using JuMP
using Statistics, Printf, Random
using Plots
using CSV, DataFrames

include("config.jl")        # parameters (Refs) + set_optimizer!
include("models.jl")        # egp_solve / rgp_solve / ermcgp_solve + MSOLVE
include("metrics.jl")       # metrics_from_pairs, count_crossings, monotone_repair, fusion_metrics
include("hierarchical.jl")  # BandedAlignment + solve_recursive
include("fusion.jl")        # rel_weights, cons_*, weighted_align, fusion_loo
include("synthetic.jl")     # base_signal, gen_gate_case_meta
include("pipeline.jl")      # compare_variants, metrics_table
include("viz.jl")           # overlay_pairs, alignment_plot, plot_weights, plot_consensus_overlay
include("io.jl")            # read_series, read_many

# ---- solver / config ----
export set_optimizer!, reset_params!

# ---- models & pipeline ----
export egp_solve, rgp_solve, ermcgp_solve, MSOLVE,
       solve_recursive, compare_variants, metrics_table

# ---- metrics ----
export metrics_from_pairs, count_crossings, count_crossings_pairs,
       monotone_repair, fusion_metrics

# ---- fusion / consensus ----
export rel_weights, cons_uniform, cons_weighted,
       weighted_align, fusion_loo, align_to_consensus

# ---- synthetic ----
export base_signal, gen_gate_case_meta

# ---- visualization ----
export overlay_pairs, alignment_plot, plot_weights, plot_consensus_overlay

# ---- io ----
export read_series, read_many

end # module
