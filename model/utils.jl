"""
    Utility functions for the Flood Mortgage Risk ABM (Julia port).
    Corresponds to preparation/utils.py in the Python version.
"""

using Images, FileIO

"""
    read_map(folder::String, path::String) -> Matrix{Float64}

Read an image file, convert to grayscale, invert, and normalize.
Equivalent to Python's `read_map()`.
"""
function read_map(folder::String, path::String)
    filepath = joinpath(folder, path)
    img = load(filepath)
    gray = Gray.(img)
    mat = Float64.(gray)           # values in [0,1]
    mat = 1.0 .- mat               # invert (same as 255 - pixel / 255)
    mx = maximum(mat)
    if mx > 0
        mat ./= mx                 # normalize to [0,1]
    end
    return mat
end

"""
    density_preprocess(density_map::Matrix{Float64}) -> (Vector{Float64}, Vector{Tuple{Int,Int}})

Compute cumulative density slots and coordinates for agent placement.
Returns (density_slots, coordinates) where coordinates are (row, col) tuples.
"""
function density_preprocess(density_map::Matrix{Float64})
    coords = Tuple{Int,Int}[]
    values = Float64[]
    for i in 1:size(density_map, 1)        # rows outer — matches Python numpy row-major order
        for j in 1:size(density_map, 2)    # columns
            if density_map[i, j] > 0
                push!(coords, (i, j))
                push!(values, density_map[i, j])
            end
        end
    end
    cumvals = cumsum(values)
    cumvals ./= cumvals[end]    # normalize to [0,1]
    return cumvals, coords
end

"""
    sigmoid(x::Float64) -> Float64

Numerically stable sigmoid. Matches Python utils.sigmoid.
"""
function sigmoid(x::Float64)
    if x >= 0
        z = exp(-x)
        return 1.0 / (1.0 + z)
    else
        ez = exp(x)
        return ez / (1.0 + ez)
    end
end

"""
    calc_share_expenses(income, m, varepsilon, s_ub)

Share of income used for living expenses (scalar or vector).
Matches Python `calc_share_expenses`.
"""
function calc_share_expenses(income::Float64, m::Float64, varepsilon::Float64, s_ub::Float64)
    share = m / (1.0 + varepsilon * income)
    return min(share, s_ub)
end

function calc_share_expenses(income::Vector{Float64}, m::Float64, varepsilon::Float64, s_ub::Float64)
    share = m ./ (1.0 .+ varepsilon .* income)
    return min.(share, s_ub)
end

"""
    scorecard(r_cap, income, seniority, ltv, install, scoring_intensity) -> Vector{Float64}

Vectorized scorecard for new joiners.
Matches Python `scorecard` from utils.py.
"""
function scorecard(r_cap::Vector{Float64}, income::Vector{Float64},
                   seniority::Vector{Float64}, ltv::Vector{Float64},
                   install::Vector{Float64}, scoring_intensity::Float64=1.0)
    n = length(r_cap)
    scores = zeros(Float64, n)
    for i in 1:n
        # Capital score (max 80)
        cap = r_cap[i] <= 0.4 ? 20.0 :
              r_cap[i] <= 0.6 ? 40.0 :
              r_cap[i] <= 0.8 ? 60.0 : 80.0

        # Income score (max 65)
        inc = income[i] <= 1500 ? 13.0 :
              income[i] <= 2000 ? 26.0 :
              income[i] <= 2500 ? 39.0 :
              income[i] <= 3000 ? 52.0 : 65.0

        # Seniority score (max 65)
        sen = seniority[i] <= 100 ? 13.0 :
              seniority[i] <= 200 ? 26.0 :
              seniority[i] <= 300 ? 39.0 :
              seniority[i] <= 400 ? 52.0 : 65.0

        # LTV score (max 50)
        ltv_s = ltv[i] <= 0.4 ? 50.0 :
                ltv[i] <= 0.6 ? 37.5 :
                ltv[i] <= 0.8 ? 25.0 : 12.5

        # Installment score (max 40)
        inst = install[i] <= 800 ? 40.0 :
               install[i] <= 1000 ? 30.0 :
               install[i] <= 1200 ? 20.0 : 10.0

        scores[i] = (cap + inc + sen + ltv_s + inst) * scoring_intensity
    end
    return scores
end
