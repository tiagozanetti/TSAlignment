# models.jl
# ----------------------------------------------------------------------------------
# The three Goal Programming alignment models — EGP, RGP, ERMCGP — as single-shot
# MILP solvers on a pair (A, B). Each returns matched pairs plus quality metrics,
# or `nothing` if the optimizer did not reach an accepted status.
#
# These are the leaf solvers the hierarchical pipeline calls per segment; you may
# also call them directly on a short pair. The formulations are faithful to the
# validated research code (bloco3 / tsa_models_hierarchical.jl); the only changes
# are that the optimizer is injected (config.jl) and scalar parameters are read
# from Refs so sweeps are trivial.
# ----------------------------------------------------------------------------------

# ---- shared, model-agnostic constraint helpers ----
"Forbid matches outside a band of half-width W around the diagonal (shifted by `offset`)."
function add_window!(model, x, m, n, W; offset=0)
    for i in 1:m, j in 1:n
        abs(i - (j - offset)) > W && @constraint(model, x[i, j] == 0)
    end
end

"Carry-forward monotonicity (the adopted form): matched columns never decrease in row order."
function add_mono!(model, x, a, m, n)
    @expression(model, pos[i=1:m], sum(j * x[i, j] for j in 1:n))
    @variable(model, g[1:m] >= 0)
    @constraint(model, g[1] >= pos[1])
    @constraint(model, [i = 2:m], g[i] >= g[i-1])
    @constraint(model, [i = 1:m], g[i] >= pos[i])
    @constraint(model, [i = 2:m], pos[i] >= g[i-1] - (n + 1) * a[i])
end

"Pin a set of (i, j) matches (used to stitch segment boundaries in the hierarchy)."
function add_fixed!(model, x, fixed_pairs)
    fixed_pairs === nothing && return
    for (i, j) in fixed_pairs
        @constraint(model, x[i, j] == 1)
    end
end

"Read the solution: matched pairs + MAE / R² / DW over A's timeline."
function extract_result(model, x, a, A, B, t)
    m, n = length(A), length(B)
    xv = value.(x)
    pairs = [(i, j) for i in 1:m, j in 1:n if xv[i, j] > 0.5]
    lk = Dict(i => j for (i, j) in pairs)
    errs = [haskey(lk, i) ? abs(A[i] - B[lk[i]]) : abs(A[i]) for i in 1:m]
    Bal = [haskey(lk, i) ? B[lk[i]] : 0.0 for i in 1:m]
    sr = sum(e^2 for e in errs)
    st = sum((v - mean(A))^2 for v in A)
    (mae = mean(errs), r2 = (st == 0.0 ? NaN : 1.0 - sr / st),
     dw = mean(abs.(sort(A) .- sort(Bal))), pairs = pairs, tempo = t)
end

# ---- EGP : Extended Goal Programming ----
function egp_solve(A, B; W=select_W(length(A)), use_window=true, use_mono=true,
                   offset=0, fixed_pairs=nothing)
    m, n = length(A), length(B)
    model = _new_model()
    @variable(model, x[1:m, 1:n], Bin); @variable(model, a[1:m] >= 0); @variable(model, b[1:n] >= 0)
    @variable(model, dm[1:m] >= 0); @variable(model, dp[1:m] >= 0)
    @variable(model, Λ >= 0); @variable(model, δp >= 0); @variable(model, δn >= 0)
    @objective(model, Min, ALPHA[] * Λ + (1 - ALPHA[]) * (W1[] * sum(dm[i] + dp[i] for i in 1:m) + W2[] * δp))
    @constraint(model, [i=1:m], ALPHA[] * (W1[] * (dm[i] + dp[i]) + W2[] * δp) <= Λ)
    @constraint(model, sum(a) + sum(b) + δn - δp == 0)
    @constraint(model, [i=1:m], sum(x[i, j] for j in 1:n) + a[i] == 1)
    @constraint(model, [j=1:n], sum(x[i, j] for i in 1:m) + b[j] == 1)
    @constraint(model, [i=1:m], sum(B[j] * x[i, j] for j in 1:n) + dm[i] - dp[i] == A[i])
    use_window && add_window!(model, x, m, n, W; offset=offset)
    use_mono   && add_mono!(model, x, a, m, n)
    add_fixed!(model, x, fixed_pairs)
    t = @elapsed optimize!(model)
    _accepted(model) || return nothing
    extract_result(model, x, a, A, B, t)
end

# ---- RGP : Restricted Goal Programming ----
function rgp_solve(A, B; W=select_W(length(A)), use_window=true, use_mono=true,
                   offset=0, fixed_pairs=nothing)
    m, n = length(A), length(B)
    σ_A = std(A); σ_A = σ_A == 0 ? 1.0 : σ_A
    σ_B = std(B); σ_B = σ_B == 0 ? 1.0 : σ_B
    Â = RHO_A[] * σ_A; B̂ = RHO_B[] * σ_B
    model = _new_model()
    @variable(model, x[1:m, 1:n], Bin); @variable(model, a[1:m] >= 0); @variable(model, b[1:n] >= 0)
    @variable(model, dm[1:m] >= 0); @variable(model, dp[1:m] >= 0)
    @variable(model, Λ >= 0); @variable(model, δp >= 0); @variable(model, δn >= 0)
    @variable(model, u[1:m], Bin); @variable(model, ρ1[1:m] >= 0); @variable(model, ρ2[1:m] >= 0)
    @variable(model, Sd_m[1:m] >= 0); @variable(model, Sd_p[1:m] >= 0)
    @objective(model, Min, ALPHA[] * Λ + (1 - ALPHA[]) * (W1[] * sum(dm[i] + dp[i] for i in 1:m) + W2[] * δp))
    @constraint(model, [i=1:m], ALPHA[] * (W1[] * (dm[i] + dp[i]) + W2[] * δp) <= Λ)
    @constraint(model, sum(a) + sum(b) + δn - δp == 0)
    @constraint(model, [i=1:m], sum(x[i, j] for j in 1:n) + a[i] == 1)
    @constraint(model, [j=1:n], sum(x[i, j] for i in 1:m) + b[j] == 1)
    use_window && add_window!(model, x, m, n, W; offset=offset)
    use_mono   && add_mono!(model, x, a, m, n)
    add_fixed!(model, x, fixed_pairs)
    β = [@expression(model, PSI[] * (B̂ * sum(x[i, j] for j in 1:n) + Â)) for i in 1:m]
    @constraint(model, [i=1:m], sum(B[j] * x[i, j] for j in 1:n) + M_BIG[] * u[i] >= A[i])
    @constraint(model, [i=1:m], sum(B[j] * x[i, j] for j in 1:n) - M_BIG[] * (1 - u[i]) <= A[i] - 1e-6)
    @constraint(model, [i=1:m], β[i] <= ρ1[i] + M_BIG[] * u[i])
    @constraint(model, [i=1:m], β[i] <= ρ2[i] + M_BIG[] * (1 - u[i]))
    @constraint(model, [i=1:m], sum(B[j] * x[i, j] for j in 1:n) + ρ1[i] + Sd_m[i] - dp[i] == A[i])
    @constraint(model, [i=1:m], sum(B[j] * x[i, j] for j in 1:n) - ρ2[i] + dm[i] - Sd_p[i] == A[i])
    t = @elapsed optimize!(model)
    _accepted(model) || return nothing
    extract_result(model, x, a, A, B, t)
end

# ---- ERMCGP : Extended Reference-point / Multi-Choice Goal Programming ----
function ermcgp_solve(A, B; W=select_W(length(A)), use_window=true, use_mono=true,
                      offset=0, fixed_pairs=nothing)
    m, n = length(A), length(B)
    A_abs = max.(abs.(A), EPS_GUARD[]); e_ub = E_BOUND[] .* A_abs
    model = _new_model()
    @variable(model, x[1:m, 1:n], Bin); @variable(model, a[1:m] >= 0); @variable(model, b[1:n] >= 0)
    @variable(model, dm[1:m] >= 0); @variable(model, dp[1:m] >= 0)
    @variable(model, em[1:m] >= 0); @variable(model, ep[1:m] >= 0)
    @variable(model, Λ >= 0); @variable(model, δp >= 0); @variable(model, δn >= 0)
    @objective(model, Min, ALPHA[] * Λ + (1 - ALPHA[]) *
        (W1[] * sum(dm[i] + dp[i] for i in 1:m) + W2[] * δp + GAMMA[] * sum(ep[i] for i in 1:m)))
    @constraint(model, [i=1:m], sum(B[j] * x[i, j] for j in 1:n) + dm[i] + em[i] - dp[i] - ep[i] == A[i])
    @constraint(model, [i=1:m], em[i] <= e_ub[i])
    @constraint(model, [i=1:m], ep[i] <= e_ub[i])
    @constraint(model, [i=1:m], ALPHA[] * (W1[] * (dm[i] + dp[i]) + W2[] * δp + GAMMA[] * ep[i]) <= Λ)
    @constraint(model, sum(a) + sum(b) + δn - δp == 0)
    @constraint(model, [i=1:m], sum(x[i, j] for j in 1:n) + a[i] == 1)
    @constraint(model, [j=1:n], sum(x[i, j] for i in 1:m) + b[j] == 1)
    use_window && add_window!(model, x, m, n, W; offset=offset)
    use_mono   && add_mono!(model, x, a, m, n)
    add_fixed!(model, x, fixed_pairs)
    t = @elapsed optimize!(model)
    _accepted(model) || return nothing
    extract_result(model, x, a, A, B, t)
end

"Dispatch table for the GP variants: `:egp`, `:rgp`, `:ermcgp`."
const MSOLVE = Dict(:egp => egp_solve, :rgp => rgp_solve, :ermcgp => ermcgp_solve)
