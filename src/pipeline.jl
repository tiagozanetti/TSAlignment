# pipeline.jl
# ----------------------------------------------------------------------------------
# Convenience entry point: run several GP variants on the same pair and collect
# their metrics side by side.
# ----------------------------------------------------------------------------------

"""
    compare_variants(A, B; models=[:egp, :rgp, :ermcgp], hierarchical=true) -> Dict

Align B onto A with each GP model in `models`. With `hierarchical=true` each model
runs through `solve_recursive` (for long, real series); with `false` the leaf solver
is called directly (for short cases). Returns a Dict keyed by method name; each value
is that model's result NamedTuple (carrying `pairs`).
"""
function compare_variants(A, B; models=[:egp, :rgp, :ermcgp], hierarchical=true)
    out = Dict{String,Any}()
    for mdl in models
        r = hierarchical ? solve_recursive(A, B; model = mdl) : MSOLVE[mdl](A, B)
        out[uppercase(String(mdl))] = r
    end
    return out
end

"""
    metrics_table(results; N=nothing) -> DataFrame

Flatten the Dict from `compare_variants` into a tidy table (method, mae, r2, dw,
crossings, coverage, tempo). Pass `N = length(A)` to fill the `coverage` column
(fraction of A rows matched). `crossings` counts non-monotone matches.
"""
function metrics_table(results; N=nothing)
    rows = NamedTuple[]
    for (name, r) in results
        if r === nothing
            push!(rows, (method = name, mae = NaN, r2 = NaN, dw = NaN,
                         crossings = -1, coverage = NaN, tempo = NaN))
            continue
        end
        cross = count_crossings(r.pairs)
        cov = N === nothing ? NaN : round(length(r.pairs) / N, digits = 4)
        push!(rows, (method = name,
                     mae = round(r.mae, digits = 4),
                     r2 = round(r.r2, digits = 4),
                     dw = round(r.dw, digits = 4),
                     crossings = cross,
                     coverage = cov,
                     tempo = round(get(r, :tempo, NaN), digits = 4)))
    end
    DataFrame(rows)
end
