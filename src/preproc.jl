# preproc.jl
# ----------------------------------------------------------------------------------
# Preprocessing for the GP consensus pipeline.
#
# Puts heterogeneous, time-stamped streams on the reference sampling grid, cancels a
# coarse clock offset (start anchor), synchronizes a residual lag by cross-correlation,
# drops long signal interruptions, and compacts everything to the window where all
# streams hold valid signal simultaneously. Ported verbatim from the paper's
# real-data preprocessing (Section: Real-data preprocessing).
#
# The heavy, DATASET-SPECIFIC choices (which stream needs the start anchor, which one
# needs the lag sync) are passed as per-stream flags, not hardcoded here, so the
# module stays reusable.
# ----------------------------------------------------------------------------------

"Nearest-neighbour resample of a time-stamped series `(tB, vB)` onto the grid `tA`."
function nn_resample(tA::Vector{Float64}, tB::Vector{Float64}, vB::Vector{Float64})
    n = length(tB); out = Vector{Float64}(undef, length(tA))
    @inbounds for (i, a) in enumerate(tA)
        j = clamp(searchsortedfirst(tB, a), 2, n)
        out[i] = abs(a - tB[j-1]) <= abs(a - tB[j]) ? vB[j-1] : vB[j]
    end
    out
end

"Forward-fill non-finite samples (run-enabling gap patch; only NaN is carried forward)."
function patch_nan(v0::Vector{Float64})
    v = copy(v0)
    for i in 2:length(v); isfinite(v[i]) || (v[i] = v[i-1]); end
    if !isfinite(v[1]); gi = findfirst(isfinite, v); v[1] = gi === nothing ? 0.0 : v[gi]; end
    v
end

"Inclusive `(start, end)` index runs where `mask` is `true`."
function bad_runs(mask::AbstractVector{Bool})
    runs = Tuple{Int,Int}[]; s = 0
    for (i, b) in enumerate(mask)
        if b && s == 0; s = i
        elseif !b && s != 0; push!(runs, (s, i - 1)); s = 0; end
    end
    s != 0 && push!(runs, (s, length(mask)))
    runs
end

"""
    valid_mask(v; L=5, lo=0.0, hi=220.0) -> BitVector

Scoring/validity mask: a sample is invalid only when it belongs to a run of length
`>= L` of out-of-range (`<= lo` or `> hi`) or non-finite values. Short glitches stay
valid (they are handled by the alignment), long interruptions are marked invalid.
"""
function valid_mask(v::Vector{Float64}; L::Int=5, lo::Float64=0.0, hi::Float64=220.0)
    bad = [!isfinite(x) || x <= lo || x > hi for x in v]
    valid = trues(length(v))
    for (s, e) in bad_runs(bad); (e - s + 1 >= L) && (valid[s:e] .= false); end
    valid
end

"""
    best_lag(a, b, valid; maxlag=800) -> (k, corr)

Integer lag `k` of `b` relative to `a` that maximizes the normalized cross-correlation
over the jointly valid samples, searched in `-maxlag:maxlag`. Positive `k` means `b`
lags `a`; shift `b` by `-k` to align it.
"""
function best_lag(a::Vector{Float64}, b::Vector{Float64}, valid::AbstractVector{Bool}; maxlag::Int=800)
    m = min(length(a), length(b)); a = a[1:m]; b = b[1:m]; vv = valid[1:m]
    va = a[vv]; vb = b[vv]
    an = (a .- mean(va)) ./ (std(va) + 1e-9)
    bn = (b .- mean(vb)) ./ (std(vb) + 1e-9)
    bk = 0; bc = -Inf
    for k in -maxlag:maxlag
        if k >= 0; ra = 1:(m - k); rb = (1 + k):m
        else; nk = -k; ra = (1 + nk):m; rb = 1:(m - nk); end
        length(ra) < 10 && continue
        s = 0; cc = 0.0
        @inbounds for (ia, ib) in zip(ra, rb)
            if vv[ia]; cc += an[ia] * bn[ib]; s += 1; end
        end
        s < 10 && continue
        cc /= s
        cc > bc && (bc = cc; bk = k)
    end
    bk, bc
end

"""
    preprocess_streams(streams; L=5, hr_lo=0.0, hr_hi=220.0, maxlag=800, verbose=true)
        -> (; names, series, valid, grid, N)

Full GP-pipeline preprocessing. `streams[1]` is the reference; every stream is a
NamedTuple `(; name, t, v, anchor=false, sync=false)` with raw (monotone) timestamps
`t` and values `v`. Steps:

  1. start-anchor of the `anchor` streams (pin first sample to the reference's first
     sample) to cancel a coarse clock offset;
  2. nearest-neighbour resample of every stream onto the reference grid;
  3. trim to the time window all streams span simultaneously;
  4. forward-fill gaps (`patch_nan`);
  5. best-lag synchronization of the `sync` streams against the reference;
  6. drop long interruptions (`valid_mask`) and compact to the window where every
     stream holds valid signal.

Returns the processed series (all the same length, reference first) ready for
`align_and_fuse`, the surviving-sample mask `valid`, and the reference `grid`.
"""
function preprocess_streams(streams; L::Int=5, hr_lo::Float64=0.0, hr_hi::Float64=220.0,
                            maxlag::Int=800, verbose::Bool=true)
    isempty(streams) && error("preprocess_streams: no streams given")
    tR0 = Float64.(streams[1].t)

    # 1) start-anchor + collect
    ts = Vector{Vector{Float64}}(); vs = Vector{Vector{Float64}}(); names = String[]
    for st in streams
        t = Float64.(st.t); v = Float64.(st.v)
        get(st, :anchor, false) && (t = t .- (t[1] - tR0[1]))
        push!(ts, t); push!(vs, v); push!(names, String(st.name))
    end

    # 2) resample onto reference grid, restricted to 3) the common overlap window
    lo = maximum(t[1] for t in ts); hi = minimum(t[end] for t in ts)
    mR = (ts[1] .>= lo) .& (ts[1] .<= hi); grid = ts[1][mR]
    G = [i == 1 ? vs[1][mR] : nn_resample(grid, ts[i], vs[i]) for i in eachindex(streams)]

    # 4) patch gaps, then per-stream validity
    G = [patch_nan(g) for g in G]
    valids = [valid_mask(g; L = L, lo = hr_lo, hi = hr_hi) for g in G]

    # 5) best-lag synchronization of the flagged streams
    for i in eachindex(streams)
        if get(streams[i], :sync, false)
            k, cc = best_lag(G[1], G[i], valids[1] .& valids[i]; maxlag = maxlag)
            sh = -k
            G[i] = circshift(G[i], sh)
            vi = circshift(valids[i], sh)
            if sh > 0; vi[1:sh] .= false
            elseif sh < 0; vi[end + sh + 1:end] .= false; end
            valids[i] = vi
            verbose && @printf("sync %-10s lag k=%d (%.1f min)  corr=%.3f\n",
                               names[i], k, k * 2 / 60, cc)
        end
    end

    # 6) interruption removal + compaction to the common valid window
    keep = reduce(.&, valids)
    idx = findall(keep); isempty(idx) && error("preprocess_streams: no jointly valid samples")
    sl = first(idx):last(idx); kc = keep[sl]
    series = [g[sl][kc] for g in G]
    verbose && @printf("preprocessed: N=%d  (kept %.1f%% of the common window)\n",
                       length(series[1]), 100 * mean(keep))
    return (; names = names, series = series, valid = kc, grid = grid[sl][kc], N = length(series[1]))
end
