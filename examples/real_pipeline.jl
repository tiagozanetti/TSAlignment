# examples/real_pipeline.jl
# ----------------------------------------------------------------------------------
# Align one device series B onto a reference A with each GP variant (hierarchical
# solver), tabulate the metrics, and plot the overlays.
#
# Adjust DATA to point at your co-registered CSVs (one `hr` column each). Preprocessing
# / synchronization is assumed already done (see the experiment notebooks).
# ----------------------------------------------------------------------------------

using TSAlignment
using CPLEX                                   # the tested optimizer
using Plots

TSAlignment.set_optimizer!(CPLEX.Optimizer)  # required for the GP models

DATA = joinpath(@__DIR__, "..", "..", "experiment_pipeline_vs_dtw", "data_full")
A = read_series(joinpath(DATA, "A_common.csv"))       # reference (rad7_1)
B = read_series(joinpath(DATA, "B_rad7_2.csv"))       # device to align

# --- all GP variants in one call ---
results = compare_variants(A, B; models = [:egp, :rgp, :ermcgp])
tbl = metrics_table(results; N = length(A))
show(tbl, allrows = true, allcols = true); println()

# --- overlays per variant (B remapped onto A via matched pairs) ---
mkpath(joinpath(@__DIR__, "plots"))
panels = [overlay_pairs(A, B, results[m].pairs; subtitle = "$m (hierarchical)")
          for m in ("EGP", "RGP", "ERMCGP")]
fig = plot(panels..., layout = (3, 1), size = (1200, 1100), margin = 6Plots.mm,
           plot_title = "reference A vs aligned B", plot_titlefontsize = 11)
savefig(fig, joinpath(@__DIR__, "plots", "overlays_example.png"))
println("saved examples/plots/overlays_example.png")
