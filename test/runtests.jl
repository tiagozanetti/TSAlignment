using TSAlignment
using Test
using Statistics
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
    else
        @info "No optimizer registered — skipping GP solver tests. " *
              "Call TSAlignment.set_optimizer!(CPLEX.Optimizer) before `test` to include them."
    end

end
