# examples/synthetic_scenarios.jl
# ----------------------------------------------------------------------------------
# Robustness cases on controlled synthetic signals: inject a challenge (outlier,
# interruption, ragged edges), align with a GP variant, and draw the two-track
# alignment plot with the ground-truth challenge highlighted.
# ----------------------------------------------------------------------------------

using TSAlignment
using CPLEX
using Plots

TSAlignment.set_optimizer!(CPLEX.Optimizer)

N = 120
kinds = (:short_outlier, :long_outlier, :interruption, :ragged_edges)

mkpath(joinpath(@__DIR__, "plots"))
for kind in kinds
    A, B, meta = gen_gate_case_meta(kind, N; seed = 1)
    r = egp_solve(A, B)                       # short case: leaf solver directly (no hierarchy)
    m = metrics_from_pairs(A, B, r.pairs)
    @info "$kind" mae=round(m.mae, digits=3) coverage=round(m.coverage, digits=3) crossings=count_crossings(r.pairs)
    p = alignment_plot(A, B, r.pairs; subtitle = "EGP — $kind", a_idx = meta.a_idx)
    savefig(p, joinpath(@__DIR__, "plots", "synthetic_$(kind).png"))
end
println("saved examples/plots/synthetic_*.png")

# sensitivity sweep example: vary the EGP deviation weight and watch MAE move
A, B, _ = gen_gate_case_meta(:short_outlier, N; seed = 3)
for w1 in (0.5, 0.6, 0.7, 0.8)
    TSAlignment.W1[] = w1; TSAlignment.W2[] = 1 - w1
    r = egp_solve(A, B)
    println("W1=$w1  MAE=$(round(metrics_from_pairs(A,B,r.pairs).mae, digits=3))")
end
TSAlignment.reset_params!()                   # restore paper defaults
