# examples/weighted_consensus.jl
# ----------------------------------------------------------------------------------
# Multi-series weighted consensus: learn per-series reliability weights (a corrupt
# device is downweighted automatically), build the consensus, and run the
# leave-one-out fusion evaluation (uniform mean vs weighted inverse-variance).
# ----------------------------------------------------------------------------------

using TSAlignment
using CPLEX                                   # needed by weighted_align (GP inner solver)
using CSV, DataFrames, Plots

TSAlignment.set_optimizer!(CPLEX.Optimizer)

DATA  = joinpath(@__DIR__, "..", "..", "experiment_pipeline_vs_dtw", "data_full")
NAMES = ["rad7_1", "rad7_2", "rad7_3", "b2b"]
FILES = ["A_common.csv", "B_rad7_2.csv", "B_rad7_3.csv", "B_b2b.csv"]
series = [read_series(joinpath(DATA, f)) for f in FILES]

# --- Iterative Weighted Consensus (GP) vs uniform ---
res_w = weighted_align(series; names = NAMES, model = :egp, weight = :robust, iters = 4)
res_u = weighted_align(series; names = NAMES, model = :egp, weight = :uniform, iters = 4)

mkpath(joinpath(@__DIR__, "out"))
CSV.write(joinpath(@__DIR__, "out", "consensus_weighted.csv"), DataFrame(hr = res_w.consensus))
CSV.write(joinpath(@__DIR__, "out", "weights.csv"),
          DataFrame(series = res_w.names, weight = res_w.weights,
                    mae_to_consensus = [res_w.fit[n].mae for n in res_w.names],
                    robust_scale = [res_w.fit[n].scale for n in res_w.names]))

p1 = plot_consensus_overlay(series, NAMES,
        [("uniform", res_u.consensus), ("weighted", res_w.consensus)];
        title = "Weighted multi-series consensus")
savefig(p1, joinpath(@__DIR__, "out", "consensus_overlay.png"))
savefig(plot_weights(NAMES, res_w.weights), joinpath(@__DIR__, "out", "weights_bar.png"))

# --- leave-one-out fusion evaluation (no optimizer needed for this part) ---
X = Dict(zip(NAMES, series))
lo = fusion_loo(X; folds = ["rad7_1", "rad7_2", "b2b"])   # rad7_3 stays in training
println("\n--- per-method mean over folds (MAE lower better; R2 higher better) ---")
show(lo.summary, allrows = true, allcols = true); println()
CSV.write(joinpath(@__DIR__, "out", "fusion_loo_summary.csv"), lo.summary)
println("saved examples/out/*.csv and *.png")
