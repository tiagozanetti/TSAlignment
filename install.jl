# install.jl — optional convenience.
# Dependencies are declared in Project.toml, so the standard Pkg flow also works:
#     julia> ] activate .
#     julia> ] instantiate
# This script does the same non-interactively and prints the optimizer hint.

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# ---- Optimizer -------------------------------------------------------------------
# The Goal Programming models require a JuMP MILP optimizer. This package was built
# and validated EXCLUSIVELY with CPLEX (a licensed solver). If you have CPLEX
# installed and licensed, add it to this environment:
#
#   Pkg.add("CPLEX")
#
# Any JuMP-compatible MILP optimizer *may* work, but only CPLEX has been tested.

println("""
TSAlignment.jl environment ready.

Next:
    using TSAlignment
    using CPLEX                                  # or your tested optimizer
    TSAlignment.set_optimizer!(CPLEX.Optimizer)
""")
