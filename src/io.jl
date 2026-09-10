# io.jl
# ----------------------------------------------------------------------------------
# Small IO helpers for the CSV layout used in the experiments (one column per series,
# default column name `hr`).
# ----------------------------------------------------------------------------------

"""
    read_series(path; col=:hr) -> Vector{Float64}

Read one column (default `:hr`) from a CSV as a Float64 vector.
"""
read_series(path; col=:hr) = Float64.(CSV.read(path, DataFrame)[!, col])

"""
    read_many(paths; names=nothing, col=:hr) -> (series, names)

Read several CSVs into a Vector of series (for the fusion / weighted-consensus API).
"""
function read_many(paths; names=nothing, col=:hr)
    series = [read_series(p; col = col) for p in paths]
    names === nothing && (names = [splitext(basename(p))[1] for p in paths])
    return series, names
end
