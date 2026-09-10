# hierarchical.jl
# ----------------------------------------------------------------------------------
# Hierarchical recursive alignment. Long series are split into segments; each GP
# model drives its own segmentation, boundary stitching (angle-projected windows),
# and a final fixed-pair polish. A banded data structure keeps the alignment sparse.
#
# Entry point:  solve_recursive(A, B; model = :egp | :rgp | :ermcgp)
# Faithful to tsa_models_hierarchical.jl; only config.jl (params/optimizer) changed.
# ----------------------------------------------------------------------------------

# ---- BandedAlignment: sparse i→j alignment within a fixed half-width band ----
mutable struct BandedAlignment
    N::Int
    W::Int
    data::Matrix{Int8}
    matched_i::Vector{Bool}
    matched_j::Vector{Bool}
end
BandedAlignment(N::Int, W::Int) =
    BandedAlignment(N, W, zeros(Int8, N, 2W + 1), zeros(Bool, N), zeros(Bool, N))

function clear_row!(b::BandedAlignment, i::Int)
    for w in 1:2b.W+1
        if b.data[i, w] == 1
            j = i + w - b.W - 1
            1 <= j <= b.N && (b.matched_j[j] = false)
            b.data[i, w] = 0
        end
    end
    b.matched_i[i] = false
end

function set_pair!(b::BandedAlignment, i::Int, j::Int)
    w = j - i + b.W + 1
    (w < 1 || w > 2b.W + 1 || j < 1 || j > b.N) && return false
    clear_row!(b, i)
    b.data[i, w] = 1
    b.matched_i[i] = true
    b.matched_j[j] = true
    return true
end

function clear_window!(b::BandedAlignment, i0::Int, i1::Int)
    for i in max(1, i0):min(b.N, i1)
        clear_row!(b, i)
    end
end

function compute_metrics_banded(A, B, b::BandedAlignment)
    m = b.N; errors = zeros(m); Bal = zeros(m)
    for i in 1:m
        matched = false
        for w in 1:2b.W+1
            if b.data[i, w] == 1
                j = i + w - b.W - 1
                if 1 <= j <= m
                    errors[i] = abs(A[i] - B[j]); Bal[i] = B[j]; matched = true; break
                end
            end
        end
        matched || (errors[i] = abs(A[i]))
    end
    sr = sum(e^2 for e in errors); st = sum((v - mean(A))^2 for v in A)
    (mae = mean(errors), r2 = (st == 0.0 ? NaN : 1.0 - sr / st),
     dw = mean(abs.(sort(A) .- sort(Bal))))
end

function extract_pairs_from_band(b::BandedAlignment)
    p = Tuple{Int,Int}[]
    for i in 1:b.N, w in 1:2b.W+1
        if b.data[i, w] == 1
            j = i + w - b.W - 1
            1 <= j <= b.N && push!(p, (i, j))
        end
    end
    p
end

# ---- hierarchical machinery, parameterized by the model ----
"Keep the first pair claiming each i and each j (resolve many-to-one conflicts)."
function deconflict(pairs)
    ui, uj = Set{Int}(), Set{Int}(); cl = Tuple{Int,Int}[]
    for (i, j) in pairs
        if !(i in ui) && !(j in uj)
            push!(cl, (i, j)); push!(ui, i); push!(uj, j)
        end
    end
    cl
end

function solve_segments(A, B; K, model)
    N_ = length(A); seg = N_ ÷ K; W_s = select_W(seg)
    sp = Vector{Vector{Tuple{Int,Int}}}(undef, K); t = 0.0; junc = Int[]
    for k in 1:K
        i0 = (k - 1) * seg + 1; i1 = k == K ? N_ : k * seg
        r = MSOLVE[model](A[i0:i1], B[i0:i1]; W = W_s, use_window = true, use_mono = true)
        sp[k] = Tuple{Int,Int}[]
        if r !== nothing
            t += r.tempo
            for (il, jl) in r.pairs
                ig, jg = i0 + il - 1, i0 + jl - 1
                ig <= N_ && jg <= N_ && push!(sp[k], (ig, jg))
            end
        end
        k < K && push!(junc, i1)
    end
    deconflict(vcat(sp...)), sp, junc, t, seg
end

function estimate_angle(spk; P=5)
    p = sort(spk, by = x -> x[1]); nn = length(p); nn < 2 && return 1.0
    last = p[max(1, nn - P + 1):nn]
    Δi = last[end][1] - last[1][1]; Δj = last[end][2] - last[1][2]
    Δi == 0 && return 1.0
    Δj / Δi
end

function apply_angle_boundary!(band, A, B, sp, junc, t0; model, seg, P)
    N_ = length(A); W_b = boundary_window(seg); tb = 0.0
    for (ki, jc) in enumerate(junc)
        slope = estimate_angle(sp[ki]; P = P); sks = sort(sp[ki], by = x -> x[1])
        offset = if isempty(sks)
            0
        else
            li, lj = sks[end]
            round(Int, lj + slope * (jc + 1 - li)) - (jc + 1)
        end
        b0 = max(1, jc - W_b + 1); b1 = min(N_, jc + W_b); Ww = select_W(b1 - b0 + 1)
        clear_window!(band, b0, b1)
        r = MSOLVE[model](A[b0:b1], B[b0:b1]; W = Ww, use_window = true, use_mono = false, offset = offset)
        r === nothing && (r = MSOLVE[model](A[b0:b1], B[b0:b1]; use_window = false, use_mono = false))
        if r !== nothing
            tb += r.tempo
            for (il, jl) in r.pairs
                ig, jg = b0 + il - 1, b0 + jl - 1
                ig <= N_ && jg <= N_ && set_pair!(band, ig, jg)
            end
        end
    end
    tb
end

"""
    solve_recursive(A, B; model=:egp, P=P_ANGLE[], max_n=MAX_N[], n_split=N_SPLIT[])
        -> (; mae, r2, dw, pairs, tempo)

Hierarchical alignment of B onto A using the chosen GP `model`. Segments longer
than `max_n` recurse (fan-out `n_split`); shorter ones are solved per segment,
stitched at junctions with angle-projected boundary windows, and finished with a
fixed-pair full solve. Returns metrics plus the matched pairs and total solve time.
"""
function solve_recursive(A, B; model=:egp, P=P_ANGLE[], max_n=MAX_N[], n_split=N_SPLIT[], depth=0)
    N_ = length(A)
    if N_ <= max_n
        W_ = select_W(N_); K = select_K(N_)
        fixed, sp, junc, ts, seg = solve_segments(A, B; K = K, model = model)
        band = BandedAlignment(N_, W_)
        for (i, j) in fixed; set_pair!(band, i, j); end
        tb = apply_angle_boundary!(band, A, B, sp, junc, ts; model = model, seg = seg, P = P)
        ff = extract_pairs_from_band(band)
        rf = MSOLVE[model](A, B; use_window = false, use_mono = false, fixed_pairs = ff)
        rf !== nothing && return (mae = rf.mae, r2 = rf.r2, dw = rf.dw, pairs = rf.pairs, tempo = ts + tb + rf.tempo)
        m = compute_metrics_banded(A, B, band)
        return (mae = m.mae, r2 = m.r2, dw = m.dw, pairs = ff, tempo = ts + tb)
    end
    seg = N_ ÷ n_split
    spr = Vector{Vector{Tuple{Int,Int}}}(undef, n_split); jr = Int[]; tt = 0.0
    for k in 1:n_split
        i0 = (k - 1) * seg + 1; i1 = k == n_split ? N_ : k * seg
        r = solve_recursive(A[i0:i1], B[i0:i1]; model = model, P = P, max_n = max_n, n_split = n_split, depth = depth + 1)
        spr[k] = Tuple{Int,Int}[]
        if r !== nothing
            tt += r.tempo
            for (il, jl) in r.pairs
                ig, jg = i0 + il - 1, i0 + jl - 1
                ig <= N_ && jg <= N_ && push!(spr[k], (ig, jg))
            end
        end
        k < n_split && push!(jr, i1)
    end
    W_ = select_W(seg); band = BandedAlignment(N_, W_)
    for pk in spr, (i, j) in pk; set_pair!(band, i, j); end
    tb = apply_angle_boundary!(band, A, B, spr, jr, 0.0; model = model, seg = seg, P = P); tt += tb
    m = compute_metrics_banded(A, B, band)
    (mae = m.mae, r2 = m.r2, dw = m.dw, pairs = extract_pairs_from_band(band), tempo = tt)
end
