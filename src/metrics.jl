# metrics.jl
# ----------------------------------------------------------------------------------
# Alignment-quality metrics over matched pairs, monotonicity repair/counting, and
# the fusion (held-out prediction) metric. All pure functions — no optimizer needed.
# ----------------------------------------------------------------------------------

"""
    metrics_from_pairs(A, B, pairs) -> (; mae, r2, dw, coverage)

Quality of an alignment given as matched `(i, j)` pairs, scored over the matched
subset of A's timeline. `coverage` is the fraction of A rows that were matched.
Following the paper's guidance, R² is returned as-is (callers label a case
*unalignable* rather than reporting a negative R²).
"""
function metrics_from_pairs(A, B, pairs)
    m = length(A)
    lk = Dict(i => j for (i, j) in pairs)
    midx = sort(collect(keys(lk)))
    isempty(midx) && return (mae=NaN, r2=NaN, dw=NaN, coverage=0.0)
    Am = [A[i] for i in midx]
    Bm = [B[lk[i]] for i in midx]
    err = abs.(Am .- Bm)
    st = sum((Am .- mean(Am)) .^ 2)
    (mae = mean(err),
     r2 = (st == 0 ? NaN : 1 - sum(err .^ 2) / st),
     dw = mean(abs.(sort(Am) .- sort(Bm))),
     coverage = length(midx) / m)
end

"Count matched columns that decrease in row order (non-monotone crossings)."
function count_crossings_pairs(pairs)
    p = sort(collect(pairs), by = x -> x[1])
    cols = [j for (_, j) in p]
    cr = 0
    for k in 1:length(cols)-1
        cols[k+1] < cols[k] && (cr += 1)
    end
    cr
end
const count_crossings = count_crossings_pairs

"""
    monotone_repair(pairs) -> Vector{Tuple{Int,Int}}

Greedy left-to-right repair: keep a matched pair only if its column is ≥ the last
kept column, dropping the crossings. Returns a monotone subset of `pairs`.
"""
function monotone_repair(pairs)
    p = sort(collect(pairs), by = x -> x[1])
    kept = Tuple{Int,Int}[]
    rm = 0
    for (i, j) in p
        j >= rm && (push!(kept, (i, j)); rm = j)
    end
    kept
end

"""
    fusion_metrics(pred, truth) -> (; mae, r2, dw)

Metric for a consensus's prediction of a held-out device. There is no ground truth
for the consensus itself, so quality is measured against the left-out series.
"""
function fusion_metrics(pred, truth)
    err = abs.(pred .- truth)
    st = sum((truth .- mean(truth)) .^ 2)
    (mae = mean(err), r2 = 1 - sum(err .^ 2) / st, dw = mean(abs.(sort(pred) .- sort(truth))))
end
