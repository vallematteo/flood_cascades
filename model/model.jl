"""
    Model definition of the flood mortgage risk agent-based model.

    Households live on a non-periodic grid of the province of Limburg. Each month
    (one model step):

      1. the flood severity of the month is read from the scenario's series, which sets
         the flood map, the damage D of every household and the unemployment rate;
      2. new households apply for a mortgage and are admitted by a credit scorecard;
      3. every household's collateral c and neighbourhood utility U are updated
         (`cascade_phase!`), including the value discount from last month's
         foreclosures within radius kappa;
      4. households step in order of creation (`agent_step!` in agents.jl): they may
         default through the financial or the strategic channel, be forborne, pay,
         and change employment;
      5. foreclosed and matured mortgages leave the model.

    Implementation notes:
      - Burr Type III draws use the inverse CDF (`rand_burr`).
      - Neighbourhoods use the Euclidean radius kappa and are clipped at the map edges.
      - Model properties are a typed mutable struct (`ModelProps`).
      - Output is collected in column vectors and written to Parquet at the end of a run.
"""

using Agents
using Distributions
using Random
using CSV, DataFrames, Parquet2

include("utils.jl")
include("agents.jl")

# The fitted distribution of initial funds relative to the loan (zeta) is rescaled by this factor.
const ZETA_SCALE = 0.015

# ─── Burr Type III sampling via the inverse CDF

"""
    rand_burr(rng, c, d, loc, scale) -> Float64

Draw from a Burr Type III distribution, F(x) = (1 + ((x - loc)/scale)^(-c))^(-d), the
parameterisation of scipy.stats.burr(c, d, loc, scale). With U ~ Uniform(0, 1),
X = scale * (U^(-1/d) - 1)^(-1/c) + loc.
"""
function rand_burr(rng::AbstractRNG, c::Float64, d::Float64, loc::Float64, scale::Float64)
    u = rand(rng, Uniform(0.0, 1.0))
    return scale * ((u^(-1.0/d) - 1.0)^(-1.0/c)) + loc
end

function rand_burr_vec(rng::AbstractRNG, c::Float64, d::Float64, loc::Float64, scale::Float64, n::Int)
    return [rand_burr(rng, c, d, loc, scale) for _ in 1:n]
end


# ─── Column-based data collectors

mutable struct ModelDataCollector
    gev::Vector{Float64}
    set::Vector{Int}
    step::Vector{Int}
    u::Vector{Float64}
    tot_num_agents_active::Vector{Int}
    num_default_current::Vector{Int}
    default_rate::Vector{Float64}
    num_new_joiners::Vector{Int}
    acceptance_rate::Vector{Float64}
end
ModelDataCollector() = ModelDataCollector(
    Float64[], Int[], Int[], Float64[], Int[], Int[],
    Float64[], Int[], Float64[]
)

"""
One row per foreclosure, with the household's state at the moment of foreclosure.
The last six columns attribute the foreclosure to the spillover: U and wtp with and
without the foreclosure discount (u, u0, p_wtp, p_wtp0), the discounted foreclosure
count n_kappa, the flood damage D, and the number of households in the kappa-disk.
"""
mutable struct FailureDataCollector
    step::Vector{Int}
    t::Vector{Int}
    id::Vector{Int}
    s_d::Vector{Int}
    s_e::Vector{Int}
    x::Vector{Int}
    y::Vector{Int}
    ltv::Vector{Float64}
    pti::Vector{Float64}
    psi::Vector{Float64}
    p_wtp::Vector{Float64}
    fin_threshold::Vector{Float64}
    p::Vector{Float64}
    default_reason::Vector{Int}
    u::Vector{Float64}
    u0::Vector{Float64}
    p_wtp0::Vector{Float64}
    n_kappa::Vector{Float64}
    D::Vector{Float64}
    nbox::Vector{Int}
end
FailureDataCollector() = FailureDataCollector(
    Int[], Int[], Int[], Int[], Int[], Int[], Int[],
    Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Int[],
    Float64[], Float64[], Float64[], Float64[], Float64[], Int[]
)

"""
One row per (step, household) every time the household's default trigger fires
(s_d flips 0 -> 1), whether the household is then forborne or forecloses. Grouping by
`id` gives the number of distress months per household. The state columns are the
same as in FailureDataCollector, recorded at the same point of the step, so forborne
and foreclosed households can be compared on identical covariates.
"""
mutable struct DistressDataCollector
    step::Vector{Int}
    t::Vector{Int}
    id::Vector{Int}
    s_e::Vector{Int}
    x::Vector{Int}
    y::Vector{Int}
    ltv::Vector{Float64}
    pti::Vector{Float64}
    psi::Vector{Float64}
    p_wtp::Vector{Float64}
    fin_threshold::Vector{Float64}
    p::Vector{Float64}
    default_reason::Vector{Int}
end
DistressDataCollector() = DistressDataCollector(
    Int[], Int[], Int[], Int[], Int[], Int[],
    Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Int[]
)


# ─── Typed model properties

mutable struct ModelProps
    # configuration (constant after initialisation)
    run_id::Int
    kappa::Int                 # neighbourhood radius (grid cells)
    acpt_score::Float64        # minimum scorecard value for admission
    settings::Int              # flood scenario, 0 = no floods
    add_agent_ctrl::Bool       # admit new households every month
    rho::Float64               # households per unit of housing density
    save_path::String
    tot_step::Int
    beta1::Float64
    beta2::Float64
    beta3::Float64
    exp_lb::Float64            # subsistence floor of non-housing expenditure
    eta::Float64               # baseline unemployment rate
    vartheta::Float64          # flood-induced unemployment
    alpha_param::Float64       # seniority cap (alpha)
    m_param::Float64           # expenditure-share intercept (m)
    varepsilon::Float64        # expenditure-share decay with income
    s_ub::Float64              # upper bound of the expenditure share
    N_nj::Int                  # applicants per month, before scaling by rho
    file_suffix::String
    max_agents::Int
    # maps
    density_map::Matrix{Float64}
    m_fgrote::Matrix{Float64}
    m_fmiddel::Matrix{Float64}
    m_fklein::Matrix{Float64}
    m_feklein::Matrix{Float64}
    density_slots::Vector{Float64}
    coordinates::Vector{Tuple{Int,Int}}
    gev_list::Vector{Float64}
    neighbor_offsets::Vector{Tuple{Int,Int}}   # (dx, dy) offsets of the Euclidean kappa-disk
    # dynamic state
    gev::Float64
    u_unemployment::Float64
    tot_num_agents_active::Int
    tot_num_agents_his::Int
    num_default_cumul::Int
    num_default_current::Int
    num_mature_cumul::Int
    num_mature_current::Int
    current_step::Int
    acceptance_rate::Float64
    num_created::Int
    num_joined::Int
    num_revival_current::Int   # forborne households this step (reset each step)
    num_revival_financial_current::Int   # forborne financial defaulters this step
    num_revival_strategic_current::Int   # forborne strategic defaulters this step
    save_data_enabled::Bool    # write the Parquet output at the end of the run
    # foreclosure spillover: c = v (1 - lambda D_t) (1 - delta)^n_kappa
    lambda::Float64            # share of house value lost to a flood at D = 1
    delta::Float64             # value discount per foreclosure within kappa in month t-1
    fcl_grid::Matrix{Float64}  # foreclosures of month t-1 per cell
    fcl_last_tick::Vector{Tuple{Int,Int}}   # (x, y) of this step's foreclosures, consumed next step
    # data collectors
    model_data::ModelDataCollector
    failures_data::FailureDataCollector
    distress_data::DistressDataCollector
end


# ─── Model properties constructor

function build_model_properties(;
        run_id::Int,
        kappa::Int,
        acpt_score::Float64,
        settings::Int,
        add_agents::Bool,
        rho::Float64,
        save_path::String,
        tot_step::Int,
        beta1::Float64,
        beta2::Float64,
        beta3::Float64,
        exp_lb::Float64,
        eta::Float64,
        vartheta::Float64,
        alpha::Float64,
        m::Float64,
        varepsilon::Float64,
        s_ub::Float64,
        N_nj::Int,
        file_suffix::String = "",
        density_map::Matrix{Float64},
        m_fgrote::Matrix{Float64},
        m_fmiddel::Matrix{Float64},
        m_fklein::Matrix{Float64},
        m_feklein::Matrix{Float64},
        gev_list::Vector{Float64},
        save_data_enabled::Bool = true,
        lambda::Float64,
        delta::Float64,
    )
    kappa >= 0 || throw(ArgumentError("kappa must be nonnegative"))

    density_slots, coordinates = density_preprocess(density_map)
    neighbor_offsets = Tuple{Int,Int}[
        (dx, dy) for dy in -kappa:kappa for dx in -kappa:kappa
        if 0 < dx^2 + dy^2 <= kappa^2
    ]

    return ModelProps(
        run_id, kappa, acpt_score, settings, add_agents, rho, save_path,
        tot_step, beta1, beta2, beta3, exp_lb,
        eta, vartheta, alpha, m, varepsilon, s_ub,
        N_nj, file_suffix, 6000,   # max_agents
        # maps
        density_map, m_fgrote, m_fmiddel, m_fklein, m_feklein,
        density_slots, coordinates, gev_list, neighbor_offsets,
        # dynamic state
        0.0,       # gev
        eta,       # u_unemployment
        0, 0,      # tot_num_agents_active, tot_num_agents_his
        0, 0,      # num_default_cumul, num_default_current
        0, 0,      # num_mature_cumul, num_mature_current
        0,         # current_step
        0.0,       # acceptance_rate
        0, 0,      # num_created, num_joined
        0,         # num_revival_current
        0, 0,      # num_revival_financial_current, num_revival_strategic_current
        save_data_enabled,
        lambda, delta,
        zeros(size(density_map)), Tuple{Int,Int}[],
        # data collectors
        ModelDataCollector(),
        FailureDataCollector(),
        DistressDataCollector(),
    )
end

# ─── Model constructor

"""
    create_model(; kwargs...) -> AgentBasedModel

Create and return the ABM with agents populated on the grid.
"""
function create_model(; rng_seed::Int = 42, kwargs...)
    props = build_model_properties(; kwargs...)

    density_map = props.density_map
    height, width = size(density_map)

    space = GridSpace((width, height); periodic = false, metric = :euclidean)

    model = StandardABM(
        Household,
        space;
        properties = props,
        rng = Xoshiro(rng_seed),
        model_step! = model_step!,
        warn = false,
    )

    populate_agents!(model)
    return model
end


# ─── Populate agents at t = 0

function populate_agents!(model)
    props = abmproperties(model)
    density_map = props.density_map
    height, width = size(density_map)
    rng = abmrng(model)

    num_per_cell = round.(Int, density_map .* props.rho)
    total_agents = sum(num_per_cell)

    # rescale the initial population so it does not exceed max_agents
    if total_agents > props.max_agents
        scale_factor = props.max_agents / total_agents
        num_per_cell = round.(Int, num_per_cell .* scale_factor)
        total_agents = sum(num_per_cell)
    end

    xs = Int[]; ys = Int[]
    for i in 1:height        # rows outer
        for j in 1:width     # columns inner
            n = num_per_cell[i, j]
            for _ in 1:n
                push!(xs, j)
                push!(ys, i)
            end
        end
    end

    N = length(xs)

    fgrote  = [props.m_fgrote[ys[k], xs[k]]  for k in 1:N]
    fmiddel = [props.m_fmiddel[ys[k], xs[k]] for k in 1:N]
    fklein  = [props.m_fklein[ys[k], xs[k]]  for k in 1:N]
    feklein = [props.m_feklein[ys[k], xs[k]]  for k in 1:N]

    ltv_arr    = abs.(rand(rng, Normal(0.57, 0.1), N))
    income_arr = rand_burr_vec(rng, 3.30, 0.45, -12.76, 3101.46, N)
    sen_arr    = rand(rng, Beta(0.88, 2.79), N) .* 106.34 .+ 1.0
    zeta_arr   = rand_burr_vec(rng, 9.42, 0.14, -0.11, 73.40, N) .* ZETA_SCALE
    s_i_arr    = calc_share_expenses(income_arr, props.m_param, props.varepsilon, props.s_ub)
    living_arr = s_i_arr .* income_arr
    sp_arr     = rand_burr_vec(rng, 2.20, 3.00, -0.63, 146.68, N)
    n_arr      = floor.(rand(rng, Uniform(1.0, 121.0), N))
    # house values are fitted in thousands of euros and converted to euros
    v_arr      = rand_burr_vec(rng, 3.38, 0.92, -1.34, 195.25, N) .* 1000.0
    p_arr      = ltv_arr .* v_arr ./ n_arr
    l_arr      = p_arr .* n_arr
    w_arr      = zeta_arr .* l_arr

    for k in 1:N
        create_household(model, xs[k], ys[k],
                         fgrote[k], fmiddel[k], fklein[k], feklein[k],
                         zeta_arr[k], income_arr[k], sen_arr[k], living_arr[k],
                         w_arr[k], ltv_arr[k], p_arr[k],
                         v_arr[k], sp_arr[k], n_arr[k], l_arr[k])
        props.tot_num_agents_active += 1
        props.tot_num_agents_his += 1
    end
end


# ─── Model step

"""
    model_step!(model)

Single month of the simulation: flood severity and unemployment, admission of new
households, the damage phase, household decisions, removal, and data collection.
"""
function model_step!(model)
    props = abmproperties(model)
    step = props.current_step

    # reset the per-step counters at the start, so they can be read after the step
    props.num_default_current = 0
    props.num_mature_current = 0
    props.num_revival_current = 0
    props.num_revival_financial_current = 0
    props.num_revival_strategic_current = 0

    # ── flood severity and unemployment of the month ──
    if step < length(props.gev_list)
        props.gev = props.gev_list[step + 1]
    end

    if props.gev > 0
        props.u_unemployment = props.eta + tanh(props.gev / 100.0) * props.vartheta
    else
        props.u_unemployment = props.eta
    end

    # ── new households ──
    if props.add_agent_ctrl
        add_agents!(model)
    end

    # ── damage phase ──
    # D, c and U are set for every household before any household steps, so every
    # decision reads this month's c of all neighbours. Runs after add_agents! so new
    # households are included. Uses no random numbers.
    cascade_phase!(model)

    # ── household decisions, in order of creation (sequential IDs) ──
    for id in sort!(collect(allids(model)))
        agent_step!(model[id], model)
    end

    # ── foreclosures and matured mortgages leave ──
    remove_agents!(model)

    # data collection skips the first 100 warm-up steps
    if step >= 100
        collect_model_data!(model)
    end

    if step % 50 == 0
        println("run_id: $(props.run_id), epoch: $(step), num_active: $(props.tot_num_agents_active)")
    end

    if step == props.tot_step - 1 && props.save_data_enabled
        save_data(model)
    end

    props.current_step += 1
end


# ─── Damage phase and neighbourhood utility on the grid

"""
    radius_sum(A, k) -> Matrix

Sum of `A` within Euclidean radius `k` of each cell, including the cell itself.
The disk is clipped at the map edges. Row prefixes give each horizontal disk
slice in constant time, for O(min(k, height) * length(A)) work without wrapping.
"""
function radius_sum(A::AbstractMatrix{Float64}, k::Int)
    k >= 0 || throw(ArgumentError("radius must be nonnegative"))
    H, W = size(A)
    prefix = zeros(Float64, H, W + 1)
    @inbounds for x in 1:W
        for y in 1:H
            prefix[y, x + 1] = prefix[y, x] + A[y, x]
        end
    end
    B = zeros(Float64, H, W)
    max_dy = min(k, H - 1)
    @inbounds for dy in -max_dy:max_dy
        half_width = isqrt(k^2 - dy^2)
        first_y, last_y = max(1, 1 - dy), min(H, H - dy)
        for x in 1:W
            left, right = max(1, x - half_width), min(W, x + half_width)
            for y in first_y:last_y
                B[y, x] += prefix[y + dy, right + 1] - prefix[y + dy, left]
            end
        end
    end
    return B
end

"""
    cascade_phase!(model)

Damage phase, run for every household before any household steps.

1. fcl_grid holds last step's foreclosures per cell.
2. n_kappa_i = radius(fcl_grid), the number of last step's foreclosures within kappa of i.
3. c0_i = v_i (1 - lambda D_t), the collateral without the discount, and
   c_i = c0_i (1 - delta)^n_kappa_i (`update_damage!`).
4. U_i = radius(c) / radius(v) and U0_i = radius(c0) / radius(v), as in `neighbor_utility`.

Only households in the model enter the sums, including this step's new households.
"""
function cascade_phase!(model)
    props = abmproperties(model)
    k = props.kappa

    props.fcl_grid .= 0.0
    for (x, y) in props.fcl_last_tick
        props.fcl_grid[y, x] += 1.0
    end
    empty!(props.fcl_last_tick)

    n_cell = zeros(size(props.fcl_grid))
    for a in allagents(model)
        n_cell[a.y, a.x] += 1.0
    end
    box_n = radius_sum(n_cell, k)
    box_F = radius_sum(props.fcl_grid, k)

    c_cell  = zeros(size(props.fcl_grid))
    c0_cell = zeros(size(props.fcl_grid))
    v_cell  = zeros(size(props.fcl_grid))
    for a in allagents(model)
        a.nbox = Int(box_n[a.y, a.x])
        a.n_kappa = box_F[a.y, a.x]
        update_damage!(a, props)
        c_cell[a.y, a.x]  += a.c
        c0_cell[a.y, a.x] += a.c0
        v_cell[a.y, a.x]  += a.v
    end
    box_c  = radius_sum(c_cell, k)
    box_c0 = radius_sum(c0_cell, k)
    box_v  = radius_sum(v_cell, k)
    for a in allagents(model)
        bv = box_v[a.y, a.x]
        a.u  = bv != 0 ? clamp(box_c[a.y, a.x]  / bv, 0.0, 1.0) : 0.0
        a.u0 = bv != 0 ? clamp(box_c0[a.y, a.x] / bv, 0.0, 1.0) : 0.0
    end
end


# ─── New households

function find_positions(model, num_joiners::Int)
    props = abmproperties(model)
    rng = abmrng(model)
    rvs = rand(rng, Uniform(0.0, 1.0), num_joiners)
    slots = props.density_slots
    coords = props.coordinates

    ys = Int[]
    xs = Int[]
    for rv in rvs
        idx = searchsortedfirst(slots, rv)
        idx = min(idx, length(coords))
        push!(ys, coords[idx][1])
        push!(xs, coords[idx][2])
    end
    return ys, xs
end

function add_agents!(model)
    props = abmproperties(model)
    if props.tot_num_agents_active >= props.max_agents
        props.num_created = 0
        props.num_joined = 0
        return
    end

    num_created = Int(trunc(props.rho * props.N_nj))
    if num_created <= 0
        props.num_created = 0
        return
    end
    num_created = min(num_created, props.max_agents - props.tot_num_agents_active)
    props.num_created = num_created

    ys, xs = find_positions(model, num_created)
    rng = abmrng(model)

    fgrote_v  = [props.m_fgrote[ys[k], xs[k]]  for k in 1:num_created]
    fmiddel_v = [props.m_fmiddel[ys[k], xs[k]] for k in 1:num_created]
    fklein_v  = [props.m_fklein[ys[k], xs[k]]  for k in 1:num_created]
    feklein_v = [props.m_feklein[ys[k], xs[k]]  for k in 1:num_created]

    income_v = rand_burr_vec(rng, 3.30, 0.45, -12.76, 3101.46, num_created)
    sen_v    = rand(rng, Beta(0.88, 2.79), num_created) .* 106.34 .+ 1.0
    zeta_v   = rand_burr_vec(rng, 9.42, 0.14, -0.11, 73.40, num_created) .* ZETA_SCALE
    ltv_v    = abs.(rand(rng, Normal(0.57, 0.1), num_created))
    sp_v     = rand_burr_vec(rng, 2.20, 3.00, -0.63, 146.68, num_created)
    v_v      = rand_burr_vec(rng, 3.38, 0.92, -1.34, 195.25, num_created) .* 1000.0  # thousands of euros to euros
    n_val    = 120.0
    p_v      = ltv_v .* v_v ./ n_val
    l_v      = p_v .* n_val
    w_v      = zeta_v .* l_v

    s_i_v     = calc_share_expenses(income_v, props.m_param, props.varepsilon, props.s_ub)
    living_v  = s_i_v .* income_v

    scores = scorecard(zeta_v, income_v, sen_v, ltv_v, p_v)

    props.num_joined = 0
    for i in 1:num_created
        if scores[i] >= props.acpt_score
            create_household(model, xs[i], ys[i],
                             fgrote_v[i], fmiddel_v[i], fklein_v[i], feklein_v[i],
                             zeta_v[i], income_v[i], sen_v[i], living_v[i],
                             w_v[i], ltv_v[i], p_v[i],
                             v_v[i], sp_v[i], n_val, l_v[i])
            props.tot_num_agents_active += 1
            props.tot_num_agents_his += 1
            props.num_joined += 1
        end
    end
    props.acceptance_rate = props.num_created > 0 ? props.num_joined / props.num_created : 0.0
end


# ─── Remove foreclosed and matured mortgages

function remove_agents!(model)
    props = abmproperties(model)
    fc = props.failures_data
    to_remove = Int[]
    for id in allids(model)
        a = model[id]
        if a.s_d == 1
            props.tot_num_agents_active -= 1
            props.num_default_cumul += 1
            props.num_default_current += 1
            push!(fc.step, props.current_step)
            push!(fc.t, a.t)
            push!(fc.id, a.id)
            push!(fc.s_d, a.s_d)
            push!(fc.s_e, a.s_e)
            push!(fc.x, a.x)
            push!(fc.y, a.y)
            push!(fc.ltv, a.ltv)
            push!(fc.pti, a.pti)
            push!(fc.psi, a.psi_1)
            push!(fc.p_wtp, a.p_wtp)
            push!(fc.fin_threshold, a.fin_threshold)
            push!(fc.p, a.p)
            push!(fc.default_reason, a.default_reason)
            push!(fc.u, a.u)
            push!(fc.u0, a.u0)
            push!(fc.p_wtp0, a.p_wtp0)
            push!(fc.n_kappa, a.n_kappa)
            push!(fc.D, a.D)
            push!(fc.nbox, a.nbox)
            push!(props.fcl_last_tick, (a.x, a.y))
            push!(to_remove, id)
        elseif floor(a.n) <= 0
            props.tot_num_agents_active -= 1
            props.num_mature_cumul += 1
            props.num_mature_current += 1
            push!(to_remove, id)
        end
    end
    for id in to_remove
        remove_agent!(id, model)
    end
end


# ─── Data collection (push! into column vectors)

function collect_model_data!(model)
    props = abmproperties(model)
    mc = props.model_data
    def_rate = props.tot_num_agents_active > 0 ?
               props.num_default_current / props.tot_num_agents_active : 0.0

    push!(mc.gev, props.gev)
    push!(mc.set, props.settings)
    push!(mc.step, props.current_step)
    push!(mc.u, props.u_unemployment)
    push!(mc.tot_num_agents_active, props.tot_num_agents_active)
    push!(mc.num_default_current, props.num_default_current)
    push!(mc.default_rate, def_rate)
    push!(mc.num_new_joiners, props.num_joined)
    push!(mc.acceptance_rate, props.acceptance_rate)
end

# ─── Save data

function collector_to_dataframe(c)
    d = Dict{Symbol, Any}()
    for fname in fieldnames(typeof(c))
        d[fname] = getfield(c, fname)
    end
    return DataFrame(d)
end

"""
    zstd_codec(df) -> Dict{String, Symbol}

Per-column zstd compression for Parquet2.writefile.
"""
function zstd_codec(df::DataFrame)
    return Dict(col => :zstd for col in names(df))
end

# Sort vals in place, then return (gini, median).
function gini_and_median!(vals::Vector{Float64})
    n = length(vals)
    n == 0 && return 0.0, 0.0
    sort!(vals)
    med = n % 2 == 1 ? vals[div(n, 2) + 1] :
                       0.5 * (vals[div(n, 2)] + vals[div(n, 2) + 1])
    s = sum(vals)
    s == 0.0 && return 0.0, med
    weighted = sum(i * vals[i] for i in 1:n)
    return (2.0 * weighted) / (n * s) - (n + 1.0) / n, med
end

function save_data(model)
    props = abmproperties(model)
    mkpath(props.save_path)
    suffix = props.file_suffix

    # ── model data ──
    model_df = collector_to_dataframe(props.model_data)
    Parquet2.writefile(
        joinpath(props.save_path, "model_data$(suffix).parquet"),
        model_df;
        compression_codec = zstd_codec(model_df),
    )

    # ── wealth distribution of the households alive at the final step ──
    n_ag = props.tot_num_agents_active
    w_vals  = Vector{Float64}(undef, n_ag)
    wc_vals = Vector{Float64}(undef, n_ag)
    i = 1
    for id in allids(model)
        a = model[id]
        w_vals[i]  = a.w
        wc_vals[i] = a.w + a.c
        i += 1
    end
    gini_w,  median_w  = gini_and_median!(w_vals)
    gini_wc, median_wc = gini_and_median!(wc_vals)
    final_df = DataFrame(
        run_id    = [props.run_id],
        setting   = [props.settings],
        gini_w    = [gini_w],
        median_w  = [median_w],
        gini_wc   = [gini_wc],
        median_wc = [median_wc],
    )
    Parquet2.writefile(
        joinpath(props.save_path, "agent_data$(suffix).parquet"),
        final_df;
        compression_codec = zstd_codec(final_df),
    )

    # ── distress data ──
    # Every trigger, i.e. every step at which s_d flipped 0 -> 1, whether or not the
    # household was then forborne. default_data below holds only the triggers that ended
    # in foreclosure, so forbearance is the set difference on (id, step): a household
    # triggers and resolves within the same step, so that pair is unique.
    distress_df = collector_to_dataframe(props.distress_data)
    Parquet2.writefile(
        joinpath(props.save_path, "distress_data$(suffix).parquet"),
        distress_df;
        compression_codec = zstd_codec(distress_df),
    )

    # ── default (foreclosure) data ──
    fc = props.failures_data
    if !isempty(fc.step)
        fail_df = collector_to_dataframe(fc)
        Parquet2.writefile(
            joinpath(props.save_path, "default_data$(suffix).parquet"),
            fail_df;
            compression_codec = zstd_codec(fail_df),
        )
    else
        empty_df = DataFrame(
            step=Int[], t=Int[], id=Int[], s_d=Int[], s_e=Int[],
            x=Int[], y=Int[], ltv=Float64[], pti=Float64[],
            psi=Float64[], p_wtp=Float64[], fin_threshold=Float64[],
            p=Float64[], default_reason=Int[],
            u=Float64[], u0=Float64[], p_wtp0=Float64[], n_kappa=Float64[], D=Float64[], nbox=Int[]
        )
        Parquet2.writefile(
            joinpath(props.save_path, "default_data$(suffix).parquet"),
            empty_df;
            compression_codec = zstd_codec(empty_df),
        )
    end
    println("Data saved to $(props.save_path)")
end
