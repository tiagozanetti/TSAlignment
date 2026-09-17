using TSAlignment
using Test
using Statistics
using Random
using DataFrames   # for nrow on fusion_loo output

@testset "TSAlignment.jl" begin

    @testset "monotonicity & crossings" begin
        pairs = [(1, 1), (2, 3), (3, 2), (4, 4)]          # (3,2) is a crossing
        @test count_crossings(pairs) == 1
        rep = monotone_repair(pairs)
        cols = [j for (_, j) in rep]
        @test issorted(cols)                               # repaired => monotone
        @test count_crossings(rep) == 0
        @test (3, 2) ∉ rep                                 # the crossing was dropped
    end

    @testset "metrics_from_pairs" begin
        A = [1.0, 2.0, 3.0, 4.0]; B = copy(A)
        id = [(i, i) for i in 1:4]
        m = metrics_from_pairs(A, B, id)
        @test m.mae ≈ 0 atol = 1e-12
        @test m.coverage ≈ 1.0
        @test m.r2 ≈ 1.0
        # partial coverage
        m2 = metrics_from_pairs(A, B, [(1, 1), (2, 2)])
        @test m2.coverage ≈ 0.5
    end

    @testset "fusion_metrics" begin
        truth = collect(1.0:10.0)
        @test fusion_metrics(truth, truth).mae ≈ 0 atol = 1e-12
        @test fusion_metrics(truth, truth).r2 ≈ 1.0
    end

    @testset "consensus builders & weights" begin
        clean = collect(1.0:20.0)
        noisy = clean .+ 5.0 .* randn(20)
        series = [clean, clean, noisy]
        w = rel_weights(series)
        @test isapprox(sum(w), 1.0; atol = 1e-9)
        @test w[3] < w[1]                                  # noisy series downweighted
        @test length(cons_uniform(series)) == 20
        @test length(cons_weighted(series, w)) == 20
    end

    @testset "fusion_loo (no optimizer)" begin
        base = collect(1.0:30.0)
        X = Dict("a" => base .+ 0.1randn(30),
                 "b" => base .+ 0.1randn(30),
                 "c" => base .+ 0.1randn(30),
                 "d" => base .+ 3.0randn(30))
        lo = fusion_loo(X; folds = ["a", "b", "c"], verbose = false)
        @test nrow(lo.summary) == 2
        @test Set(lo.summary.method) == Set(["uniform mean", "weighted (GP)"])
    end

    @testset "synthetic generators" begin
        v = base_signal(:multisine, 120)
        @test length(v) == 120
        for kind in (:no_rejection, :short_outlier, :long_outlier, :interruption, :ragged_edges)
            A, B, meta = gen_gate_case_meta(kind, 100; seed = 1)
            @test length(B) == 100 && length(A) >= 100   # :ragged_edges makes A longer than B by design
            @test haskey(meta, :a_idx)
        end
    end

    @testset "preprocessing (no optimizer)" begin
        Random.seed!(11)
        # nn_resample on the identity grid returns the series unchanged
        t = collect(0.0:2.0:20.0); v = collect(1.0:length(t))
        @test nn_resample(t, t, v) == v
        # valid_mask: a single glitch stays valid; a run >= L is invalidated
        x = fill(100.0, 30); x[10] = 0.0
        @test all(valid_mask(x; L = 5))
        x2 = copy(x); x2[10:16] .= 0.0
        vm = valid_mask(x2; L = 5)
        @test !any(vm[10:16]) && all(vm[1:9])
        # best_lag recovers a known integer shift
        bump = exp.(-((collect(1:200) .- 100.0) .^ 2) ./ (2 * 3.0^2)); lag = 7
        k, _ = best_lag(bump, circshift(bump, lag), trues(200); maxlag = 50)
        @test k == lag
        # preprocess_streams end-to-end: coarse-offset + lagged b2b are anchored + synced
        tg  = collect(0.0:2.0:2.0 * 399)
        ref = 100 .+ 0.02 .* (1:400) .+ 5 .* exp.(-((collect(1:400) .- 200.0) .^ 2) ./ (2 * 40.0^2))
        d2  = ref .+ 0.2 .* randn(400)
        d3  = ref .+ 6.0 .* randn(400)                       # noisy stream
        db  = circshift(ref, 5) .+ 0.2 .* randn(400)         # lagged, on a shifted clock
        streams = [(; name = "ref", t = tg,               v = ref),
                   (; name = "d2",  t = tg,               v = d2),
                   (; name = "d3",  t = tg,               v = d3),
                   (; name = "b2b", t = tg .+ 1.0e6,      v = db, anchor = true, sync = true)]
        pp = preprocess_streams(streams; verbose = false)
        @test length(pp.series) == 4
        @test all(length.(pp.series) .== pp.N) && pp.N > 0
        @test pp.names == ["ref", "d2", "d3", "b2b"]
        sig, w = reliability_stats(pp.series)
        @test isapprox(sum(w), 1.0; atol = 1e-9)
        @test argmax(sig) == 3                               # the noisy stream has the largest scale
    end

    # --- GP solver tests: only if an optimizer has been registered ---------------
    if TSAlignment._OPTIMIZER[] !== nothing
        @testset "GP leaf solvers (optimizer present)" begin
            A, B, _ = gen_gate_case_meta(:no_rejection, 40; seed = 2)
            for solver in (egp_solve, rgp_solve, ermcgp_solve)
                r = solver(A, B)
                @test r !== nothing
                @test count_crossings(r.pairs) == 0        # monotone by construction
            end
        end
        @testset "align_and_fuse & run_pipeline (optimizer present)" begin
            Random.seed!(3)
            n = 60
            ref = 100 .+ 5 .* sin.(0.1 .* (1:n))
            d2  = ref .+ 0.2 .* randn(n)
            d3  = ref .+ 5.0 .* randn(n)
            af  = align_and_fuse(ref, [d2, d3]; names = ["d2", "d3"], model = :egp, verbose = false)
            @test length(af.consensus) == n
            @test isapprox(sum(af.weights), 1.0; atol = 1e-9)
            @test af.names[1] == "reference"
            idx = Dict(af.names[i] => i for i in eachindex(af.names))
            @test af.weights[idx["d3"]] < af.weights[idx["d2"]]     # noisy stream downweighted
            # full pipeline from timestamped streams
            tg = collect(0.0:2.0:2.0 * (n - 1))
            streams = [(; name = "ref", t = tg, v = ref),
                       (; name = "d2",  t = tg, v = d2),
                       (; name = "d3",  t = tg, v = d3)]
            rp = run_pipeline(streams; model = :egp, verbose = false)
            @test haskey(rp, :preproc)
            @test length(rp.consensus) == rp.preproc.N
        end

    else
        @info "No optimizer registered — skipping GP solver tests. " *
              "Call TSAlignment.set_optimizer!(CPLEX.Optimizer) before `test` to include them."
    end

end
