# fusion.jl
# ----------------------------------------------------------------------------------
# Multi-series fusion / consensus construction.
#
# Two families:
#   (a) one-shot consensus builders over already co-registered series
#       — cons_uniform, cons_weighted (weighted inverse-variance);
#   (b) Iterative Weighted Consensus (IWC): weighted_align, which reuses the GP
#       pairwise solver to warp each series onto an evolving consensus while
#       learning per-series reliability weights.
#
# The weighting is an adaptation of the classical inverse-variance (information)
# weighting principle (weight ∝ 1/variance); what is specific here is the per-series
# scale σ_k = dispersion of the residual to the (median/weighted) consensus, plus a
# small regularizer so a series is never divided by zero or fully dropped.
# ----------------------------------------------------------------------------------

"""
    reliability_stats(train; c=1e-3) -> (sigma, omega)

Over co-registered series: build a provisional median consensus, take each series'
RMS residual to it as sigma_k, and set weight proportional to 1/(sigma_k^2 + c),
normalized. Returns both the per-series scale sigma_k and the weights omega_k.
"""
function reliability_stats(train; c=1e-3)
    C   = [median(s[i] for s in train) for i in 1:length(train[1])]
    sig = [sqrt(mean((s .- C) .^ 2)) for s in train]
    w   = [1.0 / (x^2 + c) for x in sig]
    return sig, w ./ sum(w)
end

"Reliability weights only (see `reliability_stats` for the (sigma_k, omega_k) pair)."
rel_weights(train; c=1e-3) = reliability_stats(train; c=c)[2]

"Plain across-series mean (uniform consensus)."
cons_uniform(train) = vec(mean(hcat(train...), dims = 2))

"Weighted across-series mean with weights `w` (final consensus Ĉ)."
cons_weighted(train, w) = vec(sum(hcat(train...) .* reshape(w, 1, :), dims = 2)) ./ sum(w)

# ---- Iterative Weighted Consensus (IWC) ----
"Warp series V onto consensus C with the GP pairwise solver; return V on C's timeline."
function align_to_consensus(C::Vector{Float64}, V::Vector{Float64}; model=:egp)
    r = solve_recursive(C, V; model = model)
    p = monotone_repair(r.pairs)
    m = length(C); av = fill(NaN, m)
    for (i, j) in p
        (1 <= i <= m && 1 <= j <= length(V)) && (av[i] = V[j])
    end
    return av, p, r.tempo
end

"Robust dispersion of finite residuals (MAD × 1.4826), used for IWC weighting."
function robust_scale(res::Vector{Float64})
    r = filter(isfinite, abs.(res))
    isempty(r) && return Inf
    return 1.4826 * median(r) + 1e-9
end

"Weighted mean per timeline index over the series present there (NaN carried forward)."
function weighted_consensus(aligned::Vector{Vector{Float64}}, w::Vector{Float64})
    S = length(aligned); m = length(aligned[1]); c = fill(NaN, m)
    for i in 1:m
        num = 0.0; den = 0.0
        for s in 1:S
            v = aligned[s][i]
            if isfinite(v); num += w[s] * v; den += w[s]; end
        end
        den > 0 && (c[i] = num / den)
    end
    for i in 2:m; isfinite(c[i]) || (c[i] = c[i-1]); end
    isfinite(c[1]) || (c[1] = c[findfirst(isfinite, c)])
    return c
end

"""
    weighted_align(series; names, model=:egp, iters=4, weight=:robust,
                   w_fixed=nothing, wfloor=0.02)
        -> (; consensus, weights, w_history, aligned, names, fit, tempo)

Iterative Weighted Consensus alignment of several co-registered series. Each
iteration warps every series onto the current consensus with the GP solver, then
(for `weight=:robust`) re-learns inverse-variance weights from robust residual
dispersion. `weight` ∈ (`:robust`, `:uniform`, `:fixed`); `w_fixed` supplies manual
weights for `:fixed`. `fit[name]` is `(mae, rms, scale, weight)` vs the final consensus.
"""
function weighted_align(series::Vector{Vector{Float64}};
                        names=nothing, model=:egp, iters=4,
                        weight=:robust, w_fixed=nothing, wfloor=0.02, verbose=true)
    S = length(series)
    names === nothing && (names = ["s$(k)" for k in 1:S])
    m = maximum(length.(series))
    ser = [length(v) == m ? copy(v) : vcat(v, fill(v[end], m - length(v)))[1:m] for v in series]

    w = weight == :fixed && w_fixed !== nothing ? w_fixed ./ sum(w_fixed) : fill(1.0 / S, S)
    aligned = [copy(v) for v in ser]
    c = weighted_consensus(aligned, w)
    w_hist = Vector{Vector{Float64}}(); ttotal = 0.0

    for it in 1:iters
        for s in 1:S
            av, _, t = align_to_consensus(c, ser[s]; model = model)
            for i in 2:m; isfinite(av[i]) || (av[i] = av[i-1]); end
            isfinite(av[1]) || (av[1] = ser[s][1])
            aligned[s] = av; ttotal += t
        end
        if weight == :robust
            sc = [robust_scale(aligned[s] .- c) for s in 1:S]
            wnew = [1.0 / (sc[s]^2) for s in 1:S]; wnew ./= sum(wnew)
            wnew = max.(wnew, wfloor); wnew ./= sum(wnew)
            w = wnew
        end
        push!(w_hist, copy(w))
        cnew = weighted_consensus(aligned, w)
        Δ = maximum(abs.(cnew .- c)); c = cnew
        verbose && @printf("iter %d  Δconsensus=%.4f  weights=[%s]\n", it, Δ,
                           join([@sprintf("%s=%.3f", names[s], w[s]) for s in 1:S], ", "))
        Δ < 1e-3 && (verbose && println("converged"); break)
    end

    fit = Dict{String,NamedTuple}()
    for s in 1:S
        r = aligned[s] .- c
        err = abs.(filter(isfinite, r))
        fit[names[s]] = (mae = mean(err), rms = sqrt(mean(err .^ 2)),
                         scale = robust_scale(r), weight = w[s])
    end

    if verbose
        d = [mean(abs.(filter(isfinite, aligned[s] .- c))) for s in 1:S]
        @printf("\nmedoid (closest real series to consensus): %s\n", names[argmin(d)])
        println("per-series fit to consensus:")
        for s in 1:S
            f = fit[names[s]]
            @printf("  %-8s  weight=%.3f  MAE=%.3f  RMS=%.3f  robust-scale=%.3f\n",
                    names[s], f.weight, f.mae, f.rms, f.scale)
        end
    end

    return (consensus = c, weights = w, w_history = w_hist,
            aligned = aligned, names = names, fit = fit, tempo = ttotal)
end

"""
    fusion_loo(series::AbstractDict; folds) -> (; full, summary)

Leave-one-out fusion evaluation. For each held-out device in `folds`, build the
consensus from the remaining series with two methods — uniform mean and weighted
(inverse-variance) — and score each against the held-out series with `fusion_metrics`.
Returns the per-fold rows and the per-method mean over folds. Needs no optimizer.
"""
function fusion_loo(series::AbstractDict; folds, verbose=true)
    names = collect(keys(series))
    methods = ["uniform mean", "weighted (GP)"]
    rows = NamedTuple[]
    for h in folds
        trn = [k for k in names if k != h]
        tr = [series[k] for k in trn]
        w = rel_weights(tr)
        cons = Dict("uniform mean" => cons_uniform(tr),
                    "weighted (GP)" => cons_weighted(tr, w))
        for mm in methods
            mt = fusion_metrics(cons[mm], series[h])
            push!(rows, (held_out = h, method = mm,
                         mae = round(mt.mae, digits = 3),
                         r2 = round(mt.r2, digits = 3),
                         dw = round(mt.dw, digits = 3)))
        end
        verbose && @printf("held-out %-8s weights: %s\n", h,
                           join(["$(n)=$(round(wi,digits=2))" for (n, wi) in zip(trn, w)], ", "))
    end
    full = DataFrame(rows)
    summary = combine(groupby(full, :method),
                      :mae => mean => :MAE, :r2 => mean => :R2, :dw => mean => :DW)
    summary = summary[[findfirst(==(mm), summary.method) for mm in methods], :]
    for col in (:MAE, :R2, :DW)
        summary[!, col] = round.(summary[!, col], digits = 3)
    end
    return (full = full, summary = summary)
end


"""
    align_and_fuse(reference, devices; names=nothing, model=:rgp, c=1e-3,
                   include_reference=true, verbose=true)
        -> (; consensus, aligned, weights, sigma, names, fit, tempo)

GP consensus over co-registered streams. Each device stream is GP-pairwise-aligned
onto `reference` with `solve_recursive` (model in :egp/:rgp/:ermcgp), then a single
reliability-weighted consensus is built over the aligned streams. Assumes inputs
already share the reference grid (see `preprocess_streams`).
"""
function align_and_fuse(reference::Vector{Float64}, devices::Vector{Vector{Float64}};
                        names=nothing, model=:rgp, c=1e-3,
                        include_reference=true, verbose=true)
    S = length(devices)
    names === nothing && (names = ["dev$(k)" for k in 1:S])
    aligned = Vector{Vector{Float64}}(); ttotal = 0.0
    for (k, B) in enumerate(devices)                        # 1) GP pairwise align -> reference
        av, _, t = align_to_consensus(reference, B; model = model)
        for i in 2:length(av); isfinite(av[i]) || (av[i] = av[i-1]); end
        isfinite(av[1]) || (av[1] = B[1])
        push!(aligned, av); ttotal += t
        verbose && @printf("aligned %-10s -> reference (%.2fs)\n", names[k], t)
    end
    streams = include_reference ? vcat([reference], aligned) : aligned
    snames  = include_reference ? vcat(["reference"], names) : names
    sigma, w = reliability_stats(streams; c = c)            # 2) weights + weighted consensus
    consensus = weighted_consensus(streams, w)
    fit = Dict{String,NamedTuple}()                         # 3) per-stream fit
    for s in eachindex(streams)
        r = abs.(filter(isfinite, streams[s] .- consensus))
        fit[snames[s]] = (sigma = sigma[s], weight = w[s],
                          mae = mean(r), rms = sqrt(mean(r .^ 2)))
    end
    if verbose
        println("reliability weights (sigma_k, omega_k):")
        for s in eachindex(streams)
            @printf("  %-10s sigma=%.3f  omega=%.3f\n", snames[s], sigma[s], w[s])
        end
    end
    return (consensus = consensus, aligned = aligned, weights = w,
            sigma = sigma, names = snames, fit = fit, tempo = ttotal)
end
