# viz.jl
# ----------------------------------------------------------------------------------
# Plotting helpers (Plots.jl). All take matched pairs or aligned vectors, so they
# work with any solver's output. Default y-label is "Pulse rate (bpm)" (wearable HR),
# override via `ylabel=` for other signals.
# ----------------------------------------------------------------------------------

"""
    overlay_pairs(A, B, pairs; subtitle="", rng=nothing, ylabel="Pulse rate (bpm)")

Overlay A and B-remapped-onto-A (via matched `pairs`) on a shared index axis.
"""
function overlay_pairs(A, B, pairs; subtitle="", rng=nothing, ylabel="Pulse rate (bpm)")
    m = length(A); matched = zeros(Int, m)
    for (i, j) in pairs; 1 <= i <= m && (matched[i] = j); end
    Bal = [matched[i] == 0 ? NaN : B[matched[i]] for i in 1:m]
    idx = rng === nothing ? (1:m) : rng
    p = plot(title = subtitle, titlefontsize = 9, legend = :topright, xlabel = "index", ylabel = ylabel)
    plot!(p, idx, A[idx], color = :seagreen, lw = 1.2, label = "A")
    plot!(p, idx, Bal[idx], color = :firebrick, lw = 1.0, label = "B aligned to A")
    p
end

"""
    alignment_plot(A, B, pairs; subtitle="", a_idx=Int[])

Two-track view: A drawn on top, B below, a gray segment per matched pair (crossing
segments = non-monotone matches). Stars mark ground-truth `a_idx` (injected
challenges); red crosses mark unmatched A rows.
"""
function alignment_plot(A, B, pairs; subtitle="", a_idx=Int[])
    m, n = length(A), length(B)
    matched = zeros(Int, m)
    for (i, j) in pairs; 1 <= i <= m && (matched[i] = j); end
    off = (maximum(A) - minimum(B)) + 0.25 * (maximum(A) - minimum(A) + 1e-9)
    Aup = A .+ off
    p = plot(title = subtitle, titlefontsize = 9, legend = false, xlabel = "index",
             ylabel = "A (top) / B (bottom)", xlims = (0, max(m, n) + 1))
    for i in 1:m
        j = matched[i]; j == 0 && continue
        plot!(p, [i, j], [Aup[i], B[j]], color = :gray55, alpha = 0.40, lw = 0.7)
    end
    plot!(p, 1:m, Aup, color = :seagreen, lw = 1.6, marker = :circle, ms = 1.6, markerstrokewidth = 0)
    plot!(p, 1:n, B, color = :firebrick, lw = 1.6, marker = :circle, ms = 1.6, markerstrokewidth = 0)
    isempty(a_idx) || scatter!(p, a_idx, Aup[a_idx], marker = :star5, ms = 6, color = :darkorange,
                               markerstrokecolor = :black, markerstrokewidth = 0.4)
    arej = [i for i in 1:m if matched[i] == 0]
    isempty(arej) || scatter!(p, arej, Aup[arej], marker = :xcross, ms = 5, color = :crimson, markerstrokewidth = 1.3)
    p
end

"Bar chart of learned per-series weights."
function plot_weights(names, weights; title="Learned weights")
    bar(names, weights, title = title, ylabel = "weight", legend = false, size = (700, 320))
end

"""
    plot_consensus_overlay(series, names, consensuses; title="Multi-series consensus",
                           ylabel="Pulse rate (bpm)")

Faint per-series lines with one or more consensus curves overlaid. `consensuses` is a
Vector of `(label, vector)` pairs (e.g. `[("uniform", cu), ("weighted", cw)]`).
"""
function plot_consensus_overlay(series, names, consensuses;
                                title="Multi-series consensus", ylabel="Pulse rate (bpm)")
    p = plot(title = title, xlabel = "index", ylabel = ylabel, legend = :topright, size = (1200, 420))
    for (nm, v) in zip(names, series)
        plot!(p, v, lw = 0.5, alpha = 0.5, label = nm)
    end
    styles = [:dash, :solid, :dot, :dashdot]
    for (k, (lab, c)) in enumerate(consensuses)
        plot!(p, c, lw = 1.8, ls = styles[mod1(k, length(styles))],
              color = k == length(consensuses) ? :black : :gray, label = "consensus ($lab)")
    end
    p
end
