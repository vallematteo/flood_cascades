"""
    MortgageModel definition for the Flood Mortgage Risk ABM (Julia port).
    Corresponds to model_3.py in the Python version.

    ## KEY DIFFERENCES FROM PYTHON VERSION

    1. **Burr distribution**: SciPy has `scipy.stats.burr` (Burr Type III/XII).
       Julia's Distributions.jl does not ship a Burr distribution out of the box.
       We implement sampling via the inverse-CDF method:
         X = scale * ((U^(-1/d) - 1)^(-1/c)) + loc
       where U ~ Uniform(0,1). This matches SciPy's Burr XII parameterisation
       (c, d, loc, scale). See `rand_burr()` below.

    2. **Spatial distance**: This variant uses a non-periodic grid and Euclidean
       radius kappa. Neighbourhoods are clipped at the map edges, without wrapping.

    3. **Scheduler**: Mesa BaseScheduler steps agents in insertion order.
       Agents.jl by default iterates `allids(model)` in ID order, which is
       functionally the same since we insert agents sequentially.

    4. **Data collection**: We collect into pre-allocated column vectors (push!),
       then build DataFrames only at save time. This avoids O(n^2) vcat.

    5. **Saving**: We save to Parquet via Parquet2.jl for efficient storage.

    6. **`alpha` parameter renamed**: Python model uses `self.alpha` for seniority
       cap. In Julia we rename to `alpha_param` to avoid conflict with Agents.jl
       internals. Similarly `m` → `m_param`, `a`/`b` → `a_flood`/`b_flood`.

    7. **Neighbor lookup**: Both models now compute neighbors dynamically each step.
       Python caches cell positions via `get_neighborhood()` and queries live occupants
       via `get_cell_list_contents()` each step. Julia calls `nearby_ids()` fresh each step.
       Dead agents are excluded and new joiners are included automatically.

    8. **Typed properties**: Model properties use a mutable struct (`ModelProps`)
       instead of Dict{Symbol,Any} for type stability and performance.
"""

using Agents
using Distributions
using Random
using CSV, DataFrames, Parquet2

include("utils.jl")
include("agents.jl")

# ─── Burr XII sampling via inverse CDF 
"""
    rand_burr(rng, c, d, loc, scale) -> Float64

Sample from a Burr Type XII distribution with parameters matching
scipy.stats.burr(c, d, loc, scale).
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
    install_scaling::Vector{Float64}
end
ModelDataCollector() = ModelDataCollector(
    Float64[], Int[], Int[], Float64[], Int[], Int[],
    Float64[], Int[], Float64[], Float64[]
)

mutable struct AgentDataCollector
    step::Vector{Int}
    unique_id::Vector{Int}
    score::Vector{Float64}
    s_d::Vector{Int}
    s_e::Vector{Int}
    x::Vector{Int}
    y::Vector{Int}
    t::Vector{Int}
    income::Vector{Float64}
    w::Vector{Float64}
    ltv::Vector{Float64}
    sp::Vector{Float64}
    pti::Vector{Float64}
    v::Vector{Float64}
    c::Vector{Float64}
    delta_p::Vector{Float64}
    s_i::Vector{Float64}
    tau::Vector{Int}
    p::Vector{Float64}
    psi::Vector{Float64}
    p_wtp::Vector{Float64}
    u::Vector{Float64}
end
AgentDataCollector() = AgentDataCollector(
    Int[], Int[], Float64[], Int[], Int[], Int[], Int[], Int[],
    Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
    Float64[], Float64[], Float64[], Int[], Float64[], Float64[], Float64[], Float64[]
)

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
    # fcl_cascade: state at foreclosure, for offspring attribution
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
One row per (step, agent) every time the agent's default trigger fires
(s_d flips 0->1), regardless of whether it cures (revives same-tick) or
forecloses. Unlike FailureDataCollector (terminal event only) or
num_revival_current (step-level aggregate, no agent identity), this gives
a per-agent event log: grouping by `id` and summing rows gives total
delinquent months per agent; sorting by `step` within an `id` and diffing
reveals whether episodes are adjacent (one continuous run) or separated.

State columns mirror FailureDataCollector (same fields, same tick), so the
forborne population (present here, absent there) can be compared to the
foreclosed population on identical covariates at the moment of trigger,
instead of only being observable conditional on having foreclosed.
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
    # configuration (const after init)
    run_id::Int
    kappa::Int
    acpt_score::Float64
    settings::Int
    add_agent_ctrl::Bool
    collect_agent_data::Bool
    rho::Float64
    save_path::String
    tot_step::Int
    beta1::Float64
    beta2::Float64
    beta3::Float64
    lambda_sen::Float64
    low_level_ir::Float64
    exp_lb::Float64
    eta::Float64
    vartheta::Float64
    a_flood::Float64
    b_flood::Float64
    g::Float64
    alpha_param::Float64
    m_param::Float64
    varepsilon::Float64
    s_ub::Float64
    gu_max::Float64
    gu_min::Float64
    N_nj::Int
    use_ir_scaling::Bool
    use_score_scaling::Bool
    use_principal_reduction::Bool
    use_damage_subsidy::Bool
    alpha_s::Float64
    alpha_r::Float64
    alpha_l::Float64
    alpha_g::Float64
    alpha_n::Float64
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
    neighbor_offsets::Vector{Tuple{Int,Int}}   # precomputed (dx,dy) for Euclidean kappa neighbourhood
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
    gu::Float64
    std_nj::Float64
    install_scaling::Float64
    low_ir_period::Int
    scoring_intensity::Float64
    num_revival_current::Int   # revivals this step (reset each step)
    arrears_threshold::Int     # months in arrears before financial default fires (0 = immediate)
    revival_all::Bool          # true = original behaviour (all defaulters can revive), false = strategic only
    revival_lag::Int           # minimum steps in default before revival is eligible (0 = same-tick allowed)
    num_revival_financial_current::Int   # financial-reason (default_reason==1) revivals this step (reset each step)
    num_revival_strategic_current::Int   # strategic-reason (default_reason==-1) revivals this step (reset each step)
    up_psi::Float64            # upper bound of psi ~ Uniform(0, up_psi); 0.5 = canonical model
    psi1_time_dependent::Bool  # if true, redraw psi_1 (strategic threshold) each tick; psi_2 (employment) stays fixed at birth
    zeta_scale::Float64        # multiplies the drawn zeta (=w_0/l ratio). 1.0 = as-drawn (median ~43); 0.01 = unit-fix (median ~0.43)
    save_data_enabled::Bool    # if false, skip the end-of-run parquet dump (for large sweeps that only need in-memory metrics)
    # fcl_cascade: c = v(1 - lambda*D_t)(1 - delta)^n_kappa
    lambda::Float64            # share of house value lost to a flood at D = 1
    delta::Float64             # value discount per foreclosure within kappa in month t-1
    rho_mem::Float64           # monthly memory of F, rho = 2^(-1/h) for half-life h (0 = only last month's foreclosures)
    F_grid::Matrix{Float64}    # weighted foreclosure count per cell, F = rho_mem*F + foreclosures of t-1
    fcl_last_tick::Vector{Tuple{Int,Int}}   # (x, y) of foreclosures booked by remove_agents!, consumed next tick
    # data collectors
    model_data::ModelDataCollector
    agent_data::AgentDataCollector
    failures_data::FailureDataCollector
    distress_data::DistressDataCollector
    micro_observer::Union{Nothing,Function} # read-only birth/decision observer; no RNG
    # ghost cache disabled: removed agents are ignored in neighborhood utility
    # ghost_cv::Dict{Int, Tuple{Float64, Float64}}
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
        lambda_sen::Float64,
        level_low_ir::Float64,
        exp_lb::Float64,
        eta::Float64,
        vartheta::Float64,
        a::Float64 = 0.3,     # unused: kept for backward-compat with existing runner call sites
        b::Float64 = 1.0,     # unused: kept for backward-compat with existing runner call sites
        g::Float64,
        alpha::Float64,
        m::Float64,
        varepsilon::Float64,
        s_ub::Float64,
        gu_max::Float64,
        gu_min::Float64,
        N_nj::Int,
        use_ir_scaling::Bool = false,
        use_score_scaling::Bool = false,
        use_principal_reduction::Bool = false,
        use_damage_subsidy::Bool = false,
        alpha_s::Float64 = 0.0,
        alpha_r::Float64 = 0.0,
        alpha_l::Float64 = 0.0,
        alpha_g::Float64 = 0.0,
        alpha_n::Float64 = 0.0,
        collect_agent_data::Bool = true,
        file_suffix::String = "",
        density_map::Matrix{Float64},
        m_fgrote::Matrix{Float64},
        m_fmiddel::Matrix{Float64},
        m_fklein::Matrix{Float64},
        m_feklein::Matrix{Float64},
        gev_list::Vector{Float64},
        arrears_threshold::Int = 0,
        revival_all::Bool = false,
        revival_lag::Int = 0,
        up_psi::Float64 = 0.5,
        psi1_time_dependent::Bool = false,
        zeta_scale::Float64 = 1.0,
        save_data_enabled::Bool = true,
        lambda::Float64,
        delta::Float64,
        rho_mem::Float64 = 0.0,
        micro_observer::Union{Nothing,Function} = nothing,
    )
    kappa >= 0 || throw(ArgumentError("kappa must be nonnegative"))

    density_slots, coordinates = density_preprocess(density_map)
    neighbor_offsets = Tuple{Int,Int}[
        (dx, dy) for dy in -kappa:kappa for dx in -kappa:kappa
        if 0 < dx^2 + dy^2 <= kappa^2
    ]

    return ModelProps(
        run_id, kappa, acpt_score, settings, add_agents, collect_agent_data, rho, save_path,
        tot_step, beta1, beta2, beta3, lambda_sen, level_low_ir, exp_lb,
        eta, vartheta, a, b, g, alpha, m, varepsilon, s_ub,
        gu_max, gu_min, N_nj,
        use_ir_scaling, use_score_scaling, use_principal_reduction, use_damage_subsidy,
        alpha_s, alpha_r, alpha_l, alpha_g, alpha_n, file_suffix, 6000,
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
        1.0,       # gu
        5.0,       # std_nj
        1.0,       # install_scaling
        0,         # low_ir_period
        1.0,       # scoring_intensity
        0,         # num_revival_current
        arrears_threshold,
        revival_all,
        revival_lag,
        0, 0,      # num_revival_financial_current, num_revival_strategic_current
        up_psi,
        psi1_time_dependent,
        zeta_scale,
        save_data_enabled,
        lambda, delta, rho_mem,
        zeros(size(density_map)), Tuple{Int,Int}[],
        # data collectors
        ModelDataCollector(),
        AgentDataCollector(),
        FailureDataCollector(),
        DistressDataCollector(),
        micro_observer,
        # Dict{Int, Tuple{Float64, Float64}}(),  # ghost_cv disabled
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


# ─── Populate agents at t=0 

function populate_agents!(model)
    props = abmproperties(model)
    density_map = props.density_map
    height, width = size(density_map)
    rng = abmrng(model)

    num_per_cell = round.(Int, density_map .* props.rho)
    total_agents = sum(num_per_cell)
    
    # initialize reate less than max_agents agents at T=0
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
    zeta_arr   = rand_burr_vec(rng, 9.42, 0.14, -0.11, 73.40, N) .* props.zeta_scale
    s_i_arr    = calc_share_expenses(income_arr, props.m_param, props.varepsilon, props.s_ub)
    living_arr = s_i_arr .* income_arr
    sp_arr     = rand_burr_vec(rng, 2.20, 3.00, -0.63, 146.68, N)
    n_arr      = floor.(rand(rng, Uniform(1.0, 121.0), N))
    # SCALE FIX: v is fit (MLE, real ING data) to be in thousands of euros, see
    # old_versions/model/model_old.py's "v *= 1000 # unit: euro" (dropped when the
    # model was refactored into model_3.py / julia_model/model.jl). Without this,
    # v (~150-300) is ~1000x smaller than income (~3000-4000, already euro-scale),
    # and ~1000x smaller than delta_p (sp*D*gamma, genuinely euro-scale), causing a
    # cascade of scale-mismatch bugs (collateral instantly wiped on any flood
    # contact, wtp_fin/sigmoid stuck near its flat midpoint for ~everyone). See
    # diag_v_scale_fix/ for the diagnostic this fix was built to test.
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


# ─── Model step ──────────────────────────────────────────────────────

"""
    model_step!(model)

Single tick of the simulation. Handles pre-agent setup, agent stepping,
removal, and data collection in the correct order.
"""
function model_step!(model)
    props = abmproperties(model)
    step = props.current_step

    # Reset per-step counters at start so they're readable after step returns
    props.num_default_current = 0
    props.num_mature_current = 0
    props.num_revival_current = 0
    props.num_revival_financial_current = 0
    props.num_revival_strategic_current = 0

    # ── Pre-agent-step logic ──
    if step < length(props.gev_list)
        props.gev = props.gev_list[step + 1]
    end

    if props.gev > 0
        props.u_unemployment = props.eta + tanh(props.gev / 100.0) * props.vartheta
    else
        props.u_unemployment = props.eta
    end

    if props.use_ir_scaling
        ir_scaling!(props)
    end
    if props.use_score_scaling
        score_scaling!(props)
    end
    # principal_reduction and damage_subsidy scaling factors are computed
    # per-agent on demand via principal_reduction_scaling / damage_subsidy_scaling

    # add new joiners
    if props.add_agent_ctrl
        add_agents!(model)
    end

    # ── Damage phase ──
    # Every agent's D, c and U are set before any agent steps, so each decision reads
    # c_{j,t} for every neighbour. Runs after add_agents! so joiners are damaged too.
    # Uses no RNG, so the random stream is unchanged.
    cascade_phase!(model)

    # ── Agent stepping ──
    # Sort by ID to reproduce Mesa BaseScheduler insertion-order activation.
    # Agents get sequential IDs, so sort(allids) == insertion order.
    # second step 
    for id in sort!(collect(allids(model)))
        agent_step!(model[id], model)
    end

    # ── Post-agent-step logic ──
    remove_agents!(model)

    # data collection skip first 100 warmup steps
    if step >= 100
        collect_model_data!(model)
        if props.collect_agent_data
            collect_agent_data!(model)
        end
    end

    if step % 50 == 0
        println("run_id: $(props.run_id), epoch: $(step), num_active: $(props.tot_num_agents_active)")
    end

    if step == props.tot_step - 1 && props.save_data_enabled
        save_data(model)
    end

    props.current_step += 1
end


# ─── fcl_cascade: memory grids and grid-based neighbourhood utility 

"""
    rho_from_half_life(h) -> Float64

Monthly memory rate with half-life `h` months, rho = 2^(-1/h). `h = 0` gives rho = 0,
so only last month's foreclosures count.
"""
rho_from_half_life(h::Real) = h == 0 ? 0.0 : 0.5^(1 / h)

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

Damage phase, run for every agent before any agent steps.

1. F_grid decays by rho_mem and takes last tick's foreclosures, so
   F_t = sum_{k>=1} rho^(k-1) X_{t-k}: an exponential kernel of infinite length kept in one grid.
2. n_kappa_i = radius(F), the weighted count of foreclosures within kappa of i (radius_sum is
   linear, so decaying the cell grid decays the disk sums).
3. c0_i = v_i(1 - lambda*D_t), the collateral without the discount, and
   c_i = c0_i * (1 - delta)^n_kappa_i (`update_damage!`).
4. U_i = radius(c) / radius(v) and U0_i = radius(c0) / radius(v), matching `neighbor_utility`.

Only alive agents enter the sums, and this tick's joiners are included.
"""
function cascade_phase!(model)
    props = abmproperties(model)
    k = props.kappa

    props.F_grid .*= props.rho_mem
    for (x, y) in props.fcl_last_tick
        props.F_grid[y, x] += 1.0
    end
    empty!(props.fcl_last_tick)

    n_cell = zeros(size(props.F_grid))
    for a in allagents(model)
        n_cell[a.y, a.x] += 1.0
    end
    box_n = radius_sum(n_cell, k)
    box_F = radius_sum(props.F_grid, k)

    c_cell  = zeros(size(props.F_grid))
    c0_cell = zeros(size(props.F_grid))
    v_cell  = zeros(size(props.F_grid))
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


# ─── Add new joiner agents 

function global_utility(model)
    props = abmproperties(model)
    tot_c = 0.0
    tot_v = 0.0
    for id in allids(model)
        a = model[id]
        tot_c += a.c
        tot_v += a.v
    end
    tot_v == 0 && return 0.0
    utility = tot_c / tot_v
    # gu = (utility - props.gu_min) / (props.gu_max - props.gu_min)
    gu = utility
    return max(0.0, gu)
end

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
    zeta_v   = rand_burr_vec(rng, 9.42, 0.14, -0.11, 73.40, num_created) .* props.zeta_scale
    ltv_v    = abs.(rand(rng, Normal(0.57, 0.1), num_created))
    sp_v     = rand_burr_vec(rng, 2.20, 3.00, -0.63, 146.68, num_created)
    v_v      = rand_burr_vec(rng, 3.38, 0.92, -1.34, 195.25, num_created) .* 1000.0  # SCALE FIX, see populate_agents!
    n_val    = 120.0
    p_v      = ltv_v .* v_v ./ n_val
    l_v      = p_v .* n_val
    w_v      = zeta_v .* l_v

    s_i_v     = calc_share_expenses(income_v, props.m_param, props.varepsilon, props.s_ub)
    living_v  = s_i_v .* income_v

    scores = scorecard(zeta_v, income_v, sen_v, ltv_v, p_v, props.scoring_intensity)

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


# ─── Remove defaulted / matured agents 

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
            # record failure into column vectors
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
        a = model[id]
        # props.ghost_cv[id] = (a.c, a.v)  # ghost cache disabled
        remove_agent!(id, model)
    end
end


# ─── Adaptive Model scaling 

function score_scaling!(props::ModelProps)
    if props.use_score_scaling
        props.scoring_intensity = exp(-props.alpha_s * max(props.gev, 0.0))
    else
        props.scoring_intensity = 1.0
    end
end

function ir_scaling!(props::ModelProps)
    if props.use_ir_scaling
        props.install_scaling = exp(-props.alpha_r * max(props.gev, 0.0))
    else
        props.install_scaling = 1.0
    end
end

function principal_reduction_scaling(props::ModelProps)
    # Returns the principal scaling factor for flood-exposed agents.
    # Called per-agent in agent_step! (not stored in props).
    return props.use_principal_reduction ? exp(-props.alpha_l * max(props.gev, 0.0)) : 1.0
end

function damage_subsidy_scaling(props::ModelProps)
    # Returns the damage scaling factor (reduces effective delta_p).
    # Called per-agent in agent_step! (not stored in props).
    return props.use_damage_subsidy ? exp(-props.alpha_g * max(props.gev, 0.0)) : 1.0
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
    push!(mc.install_scaling, props.install_scaling)
end

function collect_agent_data!(model)
    props = abmproperties(model)
    ac = props.agent_data
    for id in allids(model)
        a = model[id]
        push!(ac.step, props.current_step)
        push!(ac.unique_id, a.id)
        push!(ac.score, a.score)
        push!(ac.s_d, a.s_d)
        push!(ac.s_e, a.s_e)
        push!(ac.x, a.x)
        push!(ac.y, a.y)
        push!(ac.t, a.t)
        push!(ac.income, a.income)
        push!(ac.w, a.w)
        push!(ac.ltv, a.ltv)
        push!(ac.sp, a.sp)
        push!(ac.pti, a.pti)
        push!(ac.v, a.v)
        push!(ac.c, a.c)
        push!(ac.delta_p, a.delta_p)
        push!(ac.s_i, a.s_i)
        push!(ac.tau, a.tau)
        push!(ac.p, a.p)
        push!(ac.psi, a.psi_1)
        push!(ac.p_wtp, a.p_wtp)
        push!(ac.u, a.u)
    end
end


# ─── Save data ──────────────────────────────────────────────────────

function collector_to_dataframe(c)
    d = Dict{Symbol, Any}()
    for fname in fieldnames(typeof(c))
        d[fname] = getfield(c, fname)
    end
    return DataFrame(d)
end

"""
    optimize_dtypes(df) -> DataFrame

Downcast numeric columns to the smallest type that preserves the data range,
matching Python's `optimize_dtypes` in preparation/utils.py.
- Float64 → Float32  (Parquet does not support Float16 natively)
- Int64   → Int8 / Int16 / Int32 or UInt8 / UInt16 / UInt32
"""
function optimize_dtypes(df::DataFrame)
    result = copy(df)
    for col in names(result)
        v = result[!, col]
        T = eltype(v)
        isempty(v) && continue
        if T == Float64
            result[!, col] = Float32.(v)
        elseif T == Int64
            mn, mx = extrema(v)
            if mn >= 0
                if     mx <= typemax(UInt8);  result[!, col] = UInt8.(v)
                elseif mx <= typemax(UInt16); result[!, col] = UInt16.(v)
                elseif mx <= typemax(UInt32); result[!, col] = UInt32.(v)
                end
            else
                if     typemin(Int8)  <= mn && mx <= typemax(Int8);  result[!, col] = Int8.(v)
                elseif typemin(Int16) <= mn && mx <= typemax(Int16); result[!, col] = Int16.(v)
                elseif typemin(Int32) <= mn && mx <= typemax(Int32); result[!, col] = Int32.(v)
                end
            end
        end
    end
    return result
end

"""
    zstd_codec(df) -> Dict{String, Symbol}

Build a per-column zstd compression dict for Parquet2.writefile, matching the
`compression_codec = Dict("col" => :zstd, ...)` calling convention.
"""
function zstd_codec(df::DataFrame)
    return Dict(col => :zstd for col in names(df))
end

"""
    separate_static_columns(agent_df) -> DataFrame

Separate columns that are constant per agent (`x`, `y`, `c`, `sp`, `v`) from
the time-series, then merge them back — matching Python's `save_data` logic.

Benefits:
  1. Columnar Parquet stores long runs of the same value (one agent's constant x
     repeated across many steps) with near-zero overhead after compression.
  2. A `first_step` column is added recording when each agent first appeared.
"""
function separate_static_columns(agent_df::DataFrame)
    static_cols = ["x", "y", "sp", "v"]

    # Collect first-occurrence row index per unique_id.
    # Agents are stepped in ID order so this equals their creation order.
    seen       = Set{Int}()
    first_idxs = Int[]
    for (i, uid) in enumerate(agent_df.unique_id)
        if uid ∉ seen
            push!(seen, uid)
            push!(first_idxs, i)
        end
    end

    # Static table: one row per agent with unique_id, first_step, static cols
    static_df            = agent_df[first_idxs, vcat(["unique_id"], static_cols)]
    static_df.first_step = agent_df.step[first_idxs]

    # Time-varying table: all columns except the static ones
    time_df = select(agent_df, Not(static_cols))

    # Left-join preserves the row order of time_df (DataFrames.jl guarantee)
    combined = leftjoin(time_df, static_df, on = :unique_id)

    # Column order: step, unique_id, first_step, static..., everything else
    fixed = vcat(["step", "unique_id", "first_step"], static_cols)
    rest  = [c for c in names(combined) if c ∉ fixed]
    return combined[:, vcat(fixed, rest)]
end

# Sort vals in-place, then return (gini, median). Reuses the allocation.
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

    # ── model data ──────────────────────────────────────────────────
    model_df = collector_to_dataframe(props.model_data)
    Parquet2.writefile(
        joinpath(props.save_path, "model_data$(suffix).parquet"),
        model_df;
        compression_codec = zstd_codec(model_df),
    )

    # ── agent data or final wealth stats 
    if props.collect_agent_data
        raw_agent_df = collector_to_dataframe(props.agent_data)
        agent_df     = optimize_dtypes(separate_static_columns(raw_agent_df))
        Parquet2.writefile(
            joinpath(props.save_path, "agent_data$(suffix).parquet"),
            agent_df;
            compression_codec = zstd_codec(agent_df),
        )
    else
        # Compute wealth distribution stats from live agents at final step
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
    end

    # ── distress data (SA_4 addition; SA_3 does not write this) ──────────────
    # Every TRIGGER, i.e. every step at which s_d flipped 0->1, whether or not the
    # household was later forborne. failures_data below holds only the subset that was
    # NOT revived, so forbearance is the set difference, keyed on (id, step): a household
    # triggers and resolves within the same tick, so that pair is unique.
    #
    # Carries the same state columns as FailureDataCollector (added on top of the
    # original step/id/default_reason), so this is now wider than the 1.1-1.4 B/row,
    # 46-474 KB/run footprint originally measured for the 3-column version; expect it to
    # approach default_data's per-row size (176-568 KB/run) rather than stay near it.
    dc_out = props.distress_data
    distress_df = collector_to_dataframe(dc_out)
    Parquet2.writefile(
        joinpath(props.save_path, "distress_data$(suffix).parquet"),
        distress_df;
        compression_codec = zstd_codec(distress_df),
    )

    # ── default data 
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

