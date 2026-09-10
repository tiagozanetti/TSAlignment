# config.jl
# ----------------------------------------------------------------------------------
# Tunable parameters and solver injection.
#
# Every scalar parameter is a `Ref` (the same idiom the original research scripts
# used for WMIN/WMAX/KFRAC). Defaults are the EXACT configuration that produced the
# paper's results, so at the default values the package reproduces the article's
# numbers; a sensitivity sweep is just `TSAlignment.W1[] = 0.8`, etc.
# ----------------------------------------------------------------------------------

# ---- optimizer injection -----------------------------------------------------------
# This package was built and validated EXCLUSIVELY with CPLEX. It deliberately does
# NOT depend on CPLEX (a licensed solver) so it stays installable; you register an
# optimizer once per session before calling any GP solver:
#
#     using CPLEX
#     TSAlignment.set_optimizer!(CPLEX.Optimizer)
#
# Any JuMP-compatible MILP optimizer *may* work, but only CPLEX has been tested and
# is the solver the reported results were obtained with.
const _OPTIMIZER  = Ref{Any}(nothing)
const _TIME_LIMIT = Ref{Union{Nothing,Float64}}(nothing)   # per-solve wall-clock cap (s); nothing = no cap
const _SILENT     = Ref{Bool}(true)

"""
    set_optimizer!(factory; time_limit=nothing, silent=true)

Register the JuMP optimizer used to build and solve every Goal Programming model.
`factory` is an optimizer constructor, e.g. `CPLEX.Optimizer`. `time_limit` (seconds)
is applied per solve when the optimizer supports it. CPLEX is the only tested solver.
"""
function set_optimizer!(factory; time_limit=nothing, silent=true)
    _OPTIMIZER[]  = factory
    _TIME_LIMIT[] = time_limit === nothing ? nothing : Float64(time_limit)
    _SILENT[]     = silent
    return nothing
end

"Build a fresh JuMP model on the registered optimizer (with silent/time-limit applied)."
function _new_model()
    _OPTIMIZER[] === nothing && error(
        "No optimizer registered. Call `TSAlignment.set_optimizer!(CPLEX.Optimizer)` " *
        "first — CPLEX is the only tested solver.")
    model = Model(_OPTIMIZER[])
    _SILENT[] && set_silent(model)
    if _TIME_LIMIT[] !== nothing
        try
            set_time_limit_sec(model, _TIME_LIMIT[])
        catch
            @warn "optimizer does not support set_time_limit_sec; time_limit ignored" maxlog=1
        end
    end
    return model
end

"Accepted termination statuses (mirrors the research harness)."
_accepted(model) = string(termination_status(model)) in ("OPTIMAL", "ALMOST_FEASIBLE")

# ---- Goal Programming model parameters (paper defaults) ----------------------------
const ALPHA     = Ref(0.10)   # α  : min–max vs weighted-sum trade-off
const W1        = Ref(0.70)   # w1 : weight on per-point deviations (Σ dm+dp)
const W2        = Ref(0.30)   # w2 : weight on cardinality slack (δp)
const RHO_A     = Ref(0.10)   # ρ_A: RGP radius fraction on A (× std(A))
const RHO_B     = Ref(0.30)   # ρ_B: RGP radius fraction on B (× std(B))
const PSI       = Ref(0.50)   # Ψ  : RGP restriction strength
const M_BIG     = Ref(1e4)    # big-M
const GAMMA     = Ref(0.10)   # γ  : ERMCGP weight on excess deviation (ep)
const E_BOUND   = Ref(0.20)   # ERMCGP tolerated-deviation band (fraction of |A|)
const EPS_GUARD = Ref(1e-6)   # ERMCGP floor for |A|

# ---- hierarchical structure parameters --------------------------------------------
const MAX_N      = Ref(2700)  # segments longer than this recurse before solving
const N_SPLIT    = Ref(10)    # recursion fan-out
const P_ANGLE    = Ref(5)     # points used to estimate the boundary slope
const WMIN       = Ref(3)     # band half-width floor
const WMAX       = Ref(20)    # band half-width cap
const KFRAC      = Ref(1.0)   # scales the segment count (KFRAC<1 => fewer, larger segments)

select_K(N)   = clamp(round(Int, KFRAC[] * 0.189 * N^0.859), 2, N ÷ 10)
select_W(seg) = clamp(round(Int, 0.1 * seg), WMIN[], WMAX[])
function boundary_window(seg)
    raw = seg <= 100 ? seg ÷ 4 :
          seg <= 300 ? seg ÷ 6 :
          seg <= 600 ? seg ÷ 8 : seg ÷ 10
    return clamp(raw, 2, 20)
end

"""
    reset_params!()

Restore every parameter to the paper's default configuration.
"""
function reset_params!()
    ALPHA[]=0.10; W1[]=0.70; W2[]=0.30; RHO_A[]=0.10; RHO_B[]=0.30; PSI[]=0.50
    M_BIG[]=1e4; GAMMA[]=0.10; E_BOUND[]=0.20; EPS_GUARD[]=1e-6
    MAX_N[]=2700; N_SPLIT[]=10; P_ANGLE[]=5; WMIN[]=3; WMAX[]=20; KFRAC[]=1.0
    return nothing
end
