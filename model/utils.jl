"""
    Utility functions of the flood mortgage risk agent-based model: map reading,
    household placement, the expenditure share, and the admission scorecard.
"""

using Images, FileIO

"""
    read_map(folder::String, path::String) -> Matrix{Float64}

Read an image file, convert it to grayscale, invert it, and normalise it to [0, 1].
"""
function read_map(folder::String, path::String)
    filepath = joinpath(folder, path)
    img = load(filepath)
    gray = Gray.(img)
    mat = Float64.(gray)           # values in [0, 1]
    mat = 1.0 .- mat               # invert: dark pixels are high values
    mx = maximum(mat)
    if mx > 0
        mat ./= mx                 # normalise to [0, 1]
    end
    return mat
end

"""
    density_preprocess(density_map::Matrix{Float64}) -> (Vector{Float64}, Vector{Tuple{Int,Int}})

Cumulative density slots and coordinates for placing new households.
Returns (density_slots, coordinates), with coordinates as (row, col) tuples.
"""
function density_preprocess(density_map::Matrix{Float64})
    coords = Tuple{Int,Int}[]
    values = Float64[]
    for i in 1:size(density_map, 1)        # rows outer
        for j in 1:size(density_map, 2)    # columns inner
            if density_map[i, j] > 0
                push!(coords, (i, j))
                push!(values, density_map[i, j])
            end
        end
    end
    cumvals = cumsum(values)
    cumvals ./= cumvals[end]    # normalise to [0, 1]
    return cumvals, coords
end

"""
    calc_share_expenses(income, m, varepsilon, s_ub)

Share of income spent on non-housing living expenses, s = min(m / (1 + varepsilon income), s_ub),
for a scalar or a vector of incomes.
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
    scorecard(r_cap, income, seniority, ltv, install) -> Vector{Float64}

Admission scorecard for new applicants (maximum 300).
"""
function scorecard(r_cap::Vector{Float64}, income::Vector{Float64},
                   seniority::Vector{Float64}, ltv::Vector{Float64},
                   install::Vector{Float64})
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

        scores[i] = cap + inc + sen + ltv_s + inst
    end
    return scores
end
