# synthetic.jl
# ----------------------------------------------------------------------------------
# Controlled synthetic scenarios used for the robustness / sensitivity studies:
# a base signal plus injected challenges (outliers, interruptions, ragged edges)
# with ground-truth indices for highlighting. Feed the (A, B) directly to a leaf
# solver (egp_solve / rgp_solve / ermcgp_solve) — these cases are short enough not
# to need the hierarchy.
# ----------------------------------------------------------------------------------

"Base signal of length M: `:sine`, `:multisine`, `:ar1`, or a Gaussian-bumps default."
function base_signal(sig, M)
    if sig == :ar1
        v = zeros(M)
        for i in 2:M; v[i] = 0.95 * v[i-1] + 0.3 * randn(); end
        return v
    end
    dx = 1 / 99; t = (0:M-1) .* dx
    sig == :sine      ? sin.(2π .* t .* 3) :
    sig == :multisine ? sin.(2π .* t .* 2) .+ 0.5 .* sin.(2π .* t .* 7) :
    begin
        vv = zeros(M); beats = round.(Int, range(10, 90, length = 6))
        for b in beats, i in 1:M
            vv[i] += exp(-((i - b)^2) / (2 * (100 * 0.012)^2))
        end
        vv .- mean(vv)
    end
end

"""
    gen_gate_case_meta(kind, N; seed=1, sig=:multisine) -> (A, B, meta)

Generate a synthetic alignment case. `kind` ∈
(`:no_rejection`, `:short_outlier`, `:long_outlier`, `:interruption`, `:ragged_edges`).
`meta = (label, a_idx)` carries the ground-truth indices of the injected challenge,
useful for highlighting in `alignment_plot`.
"""
function gen_gate_case_meta(kind, N; seed=1, sig=:multisine)
    Random.seed!(1000 + seed)
    if kind == :no_rejection
        v = base_signal(sig, N); A = collect(v); B = collect(v) .+ 0.05 .* randn(N)
        return A, B, (label = "", a_idx = Int[])
    elseif kind == :short_outlier
        v = base_signal(sig, N); A = collect(v); B = collect(v) .+ 0.05 .* randn(N)
        sA = std(A); cand = collect(10:N-10); oidx = Int[]
        while length(oidx) < 5 && !isempty(cand)
            j = rand(cand); push!(oidx, j); filter!(c -> abs(c - j) >= 6, cand)
        end
        for j in oidx; A[j] += sign(randn()) * (5.0 * sA + abs(randn()) * sA); end
        return A, B, (label = "outlier", a_idx = sort(oidx))
    elseif kind == :long_outlier
        v = base_signal(sig, N); A = collect(v); B = collect(v) .+ 0.05 .* randn(N)
        sA = std(A); L = 13; start = rand(15:N-15-L)
        for j in start:start+L-1; A[j] += sign(randn()) * (5.0 * sA + abs(randn()) * sA); end
        return A, B, (label = "outlier", a_idx = collect(start:start+L-1))
    elseif kind == :interruption
        G = 15; mid = N ÷ 2; u = base_signal(sig, N + G)
        A = collect(u[1:N]); B = collect(vcat(u[1:mid], u[mid+G+1:N+G])) .+ 0.05 .* randn(N)
        return A, B, (label = "gap (no B match)", a_idx = collect(mid+1:mid+G))
    elseif kind == :ragged_edges
        extra = 15; u = base_signal(sig, N + 2 * extra)
        A = collect(u); B = collect(u[extra+1:extra+N]) .+ 0.05 .* randn(N)
        return A, B, (label = "ragged edge (no B match)", a_idx = vcat(1:extra, N+extra+1:N+2*extra))
    end
    error("unknown kind $kind")
end
