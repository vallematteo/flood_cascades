"""
    Household agent definition for the Flood Mortgage Risk ABM (Julia port).
    Corresponds to agents_3.py in the Python version.
"""

using Agents, Distributions, DataStructures

@agent struct Household(GridAgent{2})
    # location (redundant with pos, but kept for map indexing)
    x::Int          # column
    y::Int          # row
    # flood risk parameters per location
    fgrote::Float64
    fmiddel::Float64
    fklein::Float64
    feklein::Float64
    f::Float64              # accumulated flood impact
    D::Float64              # damage from current flood
    u::Float64              # neighborhood utility
    t::Int                  # agent age (months alive)
    # status
    s_e::Int                # employment status (1=employed, 0=unemployed)
    s_d::Int                # default status (0=ok, 1=defaulted)
    # financial
    l::Float64              # outstanding loan value
    income::Float64
    sen::Float64            # seniority
    tot_expenditures::Float64
    living_expenses::Float64
    w::Float64              # wealth / liquidity
    tau::Int                # months unemployed
    ltv::Float64
    p::Float64              # monthly installment (payment)
    original_install::Float64
    n::Float64              # remaining mortgage payments (kept Float for fractional subtraction)
    zeta::Float64           # initial funds / loan size
    pti::Float64            # payment-to-income ratio
    v::Float64              # house value
    c::Float64              # collateral
    sp::Float64             # area of the property
    v_arr::CircularBuffer{Int}   # rolling window of default flags (length 36)
    delta_p::Float64
    psi_1::Float64          # strategic-default threshold (redrawn each tick if psi1_time_dependent)
    psi_2::Float64          # employment-transition threshold (fixed at birth)
    default_reason::Int     # 1=hard/financial, -1=soft/strategic, 0=revived, 3=none
    p_wtp::Float64          # willingness to pay
    s_i::Float64            # share of income for living expenses
    revived::Int
    score::Float64
    fin_threshold::Float64
    financial_def_times::Int
    r_cap::Float64          # wealth / outstanding loan
    arrears::Int            # consecutive months below fin_threshold (for arrears rule)
    steps_defaulted::Int    # steps spent in default state (for revival_lag)
    # fcl_cascade: set for every agent in the damage phase (cascade_phase!)
    n_kappa::Float64        # n^kappa_{i,t-1}: foreclosures of t-1 within kappa
    nbox::Int               # households in the kappa-disk this tick
    c0::Float64             # collateral without the foreclosure discount, v(1 - lambda*D_t)
    u0::Float64             # neighbourhood utility from c0 (counterfactual for attribution)
    p_wtp0::Float64         # wtp from u0, same burden
end

"""
    create_household(model, x, y, ...) -> Household

Construct a Household and add it to the model at grid position (x, y).
"""
function create_household(model, x::Int, y::Int,
                          fgrote, fmiddel, fklein, feklein,
                          zeta, income, sen, living_exp,
                          w, ltv, p, v, sp, n, l)
    v_arr = CircularBuffer{Int}(36)
    for _ in 1:36
        push!(v_arr, 0)
    end
    # Two independent draws: psi_1 (strategic default) and psi_2 (employment).
    # In the single-psi model these were the same variable; splitting them lets
    # psi_1 be redrawn over time while psi_2 (employment type) stays fixed.
    up = abmproperties(model).up_psi
    psi_1_val = rand(abmrng(model), Uniform(0.0, up))
    psi_2_val = rand(abmrng(model), Uniform(0.0, up))
    pti_val = p / income

    agent = add_agent!((x, y), Household, model;
        x = x,
        y = y,
        fgrote = Float64(fgrote),
        fmiddel = Float64(fmiddel),
        fklein = Float64(fklein),
        feklein = Float64(feklein),
        f = 0.0,
        D = 0.0,
        u = 1.0,
        t = 0,
        s_e = 1,
        s_d = 0,
        l = Float64(l),
        income = Float64(income),
        sen = Float64(sen),
        tot_expenditures = 0.0,
        living_expenses = Float64(living_exp),
        w = Float64(w),
        tau = 0,
        ltv = Float64(ltv),
        p = Float64(p),
        original_install = Float64(p),
        n = Float64(n),
        zeta = Float64(zeta),
        pti = pti_val,
        v = Float64(v),
        c = Float64(v),
        sp = Float64(sp),
        v_arr = v_arr,
        delta_p = 0.0,
        psi_1 = psi_1_val,
        psi_2 = psi_2_val,
        default_reason = 3,
        p_wtp = 0.0,
        s_i = 0.0,
        revived = 0,
        score = 210.0,
        fin_threshold = 0.0,
        financial_def_times = 0,
        r_cap = 0.0,
        arrears = 0,
        steps_defaulted = 0,
        n_kappa = 0.0,
        nbox = 1,
        c0 = Float64(v),
        u0 = 1.0,
        p_wtp0 = 0.0,
    )
    observer = abmproperties(model).micro_observer
    observer === nothing || observer(:birth, agent, model, 0)
    return agent
end


# ─── Agent step functions (called each tick) ────────────────────────────

"""
    agent_step!(agent, model)

Main per-tick update for each Household. Matches `Household.step()` → `Household.update()`.
"""
function agent_step!(agent::Household, model)
    props = abmproperties(model)

    # Time-varying strategic threshold: redraw psi_1 each tick so an agent's
    # willingness-to-pay cutoff is no longer a fixed type. psi_2 (employment)
    # is left untouched, so unemployment spells stay persistent. Drawn at the
    # top of the tick so both the strategic trigger and the same-tick revival
    # check (update_s_d!) see the same fresh value.
    if props.psi1_time_dependent
        agent.psi_1 = rand(abmrng(model), Uniform(0.0, props.up_psi))
    end

    # rescale installment (if install_scaling != 1)
    agent.p = props.install_scaling * agent.original_install

    agent.t += 1

    # flood impact: D, c and u were set for every agent by cascade_phase! before any agent steps

    # seniority update
    agent.sen = min(props.alpha_param,
                    agent.s_e * props.lambda_sen * (agent.sen + 1) +
                    (1 - agent.s_e) * agent.sen)

    # unemployment duration
    agent.tau = agent.s_e == 1 ? 0 : (agent.tau + 1)

    # default status
    triggered_reason = update_s_d!(agent, model)
    # Record after the trigger/revival decision, before payments, wealth, loan
    # balance and employment change. Collateral/utility were fixed for all agents
    # before this tick's decisions. The observer must never draw from the model RNG.
    props.micro_observer === nothing || props.micro_observer(:decision, agent, model, triggered_reason)

    # living expenses
    living_exp = max(agent.s_i * agent.income, props.exp_lb)
    agent.living_expenses = living_exp
    agent.tot_expenditures = living_exp + (1 - agent.s_d) * agent.p

    # remaining payments
    agent.n -= (1 - agent.s_d)

    # rolling default window
    popfirst!(agent.v_arr)
    push!(agent.v_arr, agent.s_d)

    # wealth
    agent.w = agent.w + agent.income - agent.tot_expenditures
    agent.l = agent.n * agent.p

    # income update (includes employment status transition)
    update_income!(agent, model)

    # derived quantities
    agent.ltv = agent.l / agent.c
    if agent.n != 0
        agent.r_cap = agent.w / agent.l
    end
    agent.pti = agent.p / (agent.income + 0.01)

    # credit score
    compute_score!(agent, props)

    # record the trigger event (agent.s_d flipped 0->1 this tick), regardless of
    # whether it later cures (revival, above) or forecloses. Pushed here, after
    # ltv/pti/s_e are current for this tick, so distress_data's state columns land
    # on the same values FailureDataCollector would record for the same event --
    # see DistressDataCollector docstring in model.jl.
    if triggered_reason != 0
        dc = props.distress_data
        push!(dc.step, props.current_step)
        push!(dc.t, agent.t)
        push!(dc.id, agent.id)
        push!(dc.s_e, agent.s_e)
        push!(dc.x, agent.x)
        push!(dc.y, agent.y)
        push!(dc.ltv, agent.ltv)
        push!(dc.pti, agent.pti)
        push!(dc.psi, agent.psi_1)
        push!(dc.p_wtp, agent.p_wtp)
        push!(dc.fin_threshold, agent.fin_threshold)
        push!(dc.p, agent.p)
        push!(dc.default_reason, triggered_reason)
    end
end


# ─── Sub-routines ─────────────────────────────────────────────────────

function update_s_e!(agent::Household, props)
    u_val = props.u_unemployment
    if agent.s_e == 1 && agent.psi_2 < u_val
        agent.s_e = 0
    end
    if agent.s_e == 0 && agent.psi_2 >= u_val
        agent.s_e = 1
    end
end

"""
Set this tick's flood exposure D and the collateral c = v(1 - lambda*D_t)(1 - delta)^n_kappa.
n_kappa must already be set for this tick (cascade_phase!).
"""
function update_damage!(agent::Household, props)
    find_flood_map!(agent, props)
    agent.delta_p = props.lambda * agent.D * agent.v
    # Policy 2: damage subsidy — government partially offsets flood damage
    effective_delta_p = agent.delta_p * damage_subsidy_scaling(props)
    agent.c0 = max(agent.v - effective_delta_p, 0.001)
    agent.c = agent.c0 * (1 - props.delta)^agent.n_kappa
end

function update_s_d!(agent::Household, model)
    props = abmproperties(model)
    agent.s_i = calc_share_expenses(agent.income, props.m_param, props.varepsilon, props.s_ub)
    agent.fin_threshold = agent.w - max(agent.s_i * agent.income, props.exp_lb)

    # u and u0 were set for every agent by cascade_phase! (grid radius sums)
    # ANNUALIZED-INCOME FIX: wtp_fin compares a stock (loan balance l) to a flow
    # (income). Monthly income under-annualizes this leverage ratio by 12x relative
    # to standard LTI conventions. Annualizing brings origination-time LTI in line
    # with real benchmarks (~1-6x annual income). Only affects wtp_fin
    # (income_for_calc is used nowhere else).
    income_for_calc = agent.income > 0 ? agent.income * 12 : 1e-6
    # Policy 1: principal reduction — forbearance lowers effective DTI during flood steps only.
    # Applied to wtp computation, not to agent.n, so no compounding across steps.
    effective_l = agent.l * principal_reduction_scaling(props)
    #
    # SA_4 CHANGE (the ONLY difference vs SA_3): burden index and its mapping into wtp.
    #
    # SA_3:  wtp_fin = LTI * (beta3*(1-s_e)*tau + 1);  penalty = beta2 * sigmoid(wtp_fin)
    # SA_4:  X       = LTI * (tau + 1);                penalty = beta2 * X / (beta3 + X)
    #
    # Rationale. wtp_fin >= 0 always, so sigmoid was confined to [0.5, 1): half the
    # codomain was unreachable and sigmoid(0) = 0.5 charged a debt-free agent half the
    # maximum penalty. Because psi_1 ~ U(0, up_psi) is redrawn every tick, the rule
    # `p_wtp < psi_1` is a Bernoulli draw with P(default) = 1 - clamp(p_wtp, 0, 1),
    # so that floor was an irreducible beta2/2 of strategic-default hazard per tick
    # on every agent regardless of leverage.
    #
    # The Hill (Michaelis-Menten) form fixes both: it is 0 at zero burden, monotone,
    # spans [0, beta2), and approaches its asymptote as a power law (1 - beta3/X)
    # rather than exponentially, so it keeps discriminating in the high-burden tail
    # that flood shocks push agents into.
    #
    # beta3 changes meaning: it is no longer an unemployment sensitivity but the
    # HALF-SATURATION BURDEN, the value of X at which the penalty reaches beta2/2.
    # Its sampling range must be widened accordingly (~[0.5, 5]); the SA_3 range of
    # [0.01, 1] is degenerate here (at beta3 = 0.01 the penalty is >0.98 at the median).
    #
    # tau enters linearly inside X with unit coefficient, deliberately: the outer Hill
    # makes the composition concave in tau for free, and ties the tau at which an agent
    # reaches half penalty to its leverage (a levered agent is pushed over sooner).
    # The (1 - s_e) gate of SA_3 is dropped as redundant: agent.tau is already reset
    # to 0 while employed (see update_income!), so (tau + 1) == 1 for employed agents.
    wtp_fin = (effective_l / income_for_calc) * (agent.tau + 1)
    u_effective = (agent.s_d == 0 && agent.fin_threshold > agent.p) ?
        agent.u * exp(props.alpha_n * max(props.gev, 0.0)) :
        agent.u
    # clamp to [0,1]: a NO-OP on dynamics, made explicit. psi_1 ~ U(0, up_psi) with
    # up_psi = 1 governs both the trigger (wtp < psi_1) and the resolution
    # (forbearance iff wtp > 1 - psi_1), so wtp < 0 already behaves exactly like wtp = 0
    # and wtp > 1 exactly like wtp = 1. Clamping makes wtp a genuine probability, so
    # P(foreclosure) = max(0, 1 - 2*wtp) holds without a caveat, and it matters now that
    # SA_4 samples beta1 and beta2 without the beta1 > beta2 constraint.
    agent.p_wtp = clamp(props.beta1 * u_effective - props.beta2 * wtp_fin / (props.beta3 + wtp_fin), 0.0, 1.0)
    # counterfactual wtp with no foreclosure discount, for offspring attribution only
    agent.p_wtp0 = clamp(props.beta1 * agent.u0 - props.beta2 * wtp_fin / (props.beta3 + wtp_fin), 0.0, 1.0)

    # hard margin — with optional arrears rule (arrears_threshold=0 → current behaviour)
    if agent.s_d == 0 && agent.fin_threshold <= agent.p
        agent.arrears += 1
        if props.arrears_threshold == 0 || agent.arrears >= props.arrears_threshold
            agent.s_d = 1
            agent.default_reason = 1
            agent.financial_def_times += 1
            agent.arrears = 0
        end
    else
        agent.arrears = 0   # constraint no longer binding — reset counter
    end

    # soft margin — policy acts here via u_effective
    if agent.s_d == 0 && agent.p_wtp < agent.psi_1
        agent.s_d = 1
        agent.default_reason = -1
        agent.steps_defaulted = 0
    end

    # Trigger flag (agent.s_d flipped 0->1 this tick), regardless of whether it later
    # cures (revival, below) or forecloses. Since every living agent enters each tick
    # with s_d==0, s_d==1 here means "new trigger this tick". Captured as a return value
    # rather than pushed to distress_data here, because default_reason gets reset to 0
    # on same-tick revival (below), and because ltv/pti/s_e are not yet current for this
    # tick at this point in agent_step! (see the push site at the end of agent_step!).
    triggered_reason = agent.s_d == 1 ? agent.default_reason : 0

    # track how many steps this agent has been in default
    if agent.s_d == 1
        agent.steps_defaulted += 1
    else
        agent.steps_defaulted = 0
    end

    # Revival: original behaviour (revival_all=true) allows any defaulter to revive.
    # Fixed behaviour (revival_all=false) restricts revival to strategic defaulters only,
    # and revival_lag enforces a minimum number of steps in default before revival is eligible.
    p_wtp_base = clamp(props.beta1 * agent.u - props.beta2 * wtp_fin / (props.beta3 + wtp_fin), 0.0, 1.0)
    revival_eligible = props.revival_all ? agent.s_d == 1 : (
        agent.s_d == 1 &&
        agent.default_reason == -1 &&
        agent.steps_defaulted > props.revival_lag
    )
    if revival_eligible && p_wtp_base > (1 - agent.psi_1)
        if agent.default_reason == 1
            props.num_revival_financial_current += 1
        elseif agent.default_reason == -1
            props.num_revival_strategic_current += 1
        end
        agent.revived = 1
        agent.s_d = 0
        agent.default_reason = 0
        agent.steps_defaulted = 0
        props.num_revival_current += 1
    end
    return triggered_reason
end

function update_income!(agent::Household, model)
    props = abmproperties(model)
    old_s_e = agent.s_e
    update_s_e!(agent, props)
    new_s_e = agent.s_e

    if new_s_e == 1 && old_s_e == 1
        agent.income *= (1.0 + min(0.01 * (agent.sen / 12.0), props.g))
    elseif new_s_e == 0 && old_s_e == 1
        agent.income = min(0.7 * agent.income, 5000.0)
    elseif new_s_e == 0 && old_s_e == 0 && agent.tau > 12
        agent.income = 0.0
    elseif new_s_e == 1 && old_s_e == 0
        agent.income = rand_burr(abmrng(model), 3.30, 0.45, -12.76, 3101.46)
    end
end

function find_flood_map!(agent::Household, props)
    gev = props.gev
    if 0 < gev <= 30
        agent.D = agent.fgrote
    elseif 30 < gev <= 300
        agent.D = agent.fmiddel
    elseif 300 < gev <= 3000
        agent.D = agent.fklein
    elseif 3000 < gev
        agent.D = agent.feklein
    else  # gev <= 0
        agent.D = 0.0
    end
    # Persistence formula removed (a_flood, b_flood no longer used):
    # agent.f = props.a_flood * agent.f + props.b_flood * agent.D
end

"""
Agent-by-agent loop for U, the reference that `cascade_phase!`'s grid sums are tested
against. The model itself does not call it.
"""
function neighbor_utility(agent::Household, model)
    props = abmproperties(model)
    # Self always included
    tot_c = agent.c
    tot_v = agent.v
    ax, ay = agent.x, agent.y
    width  = size(props.density_map, 2)
    height = size(props.density_map, 1)

    # Same-cell: other agents (exclude self; ids_in_position returns only alive agents)
    for nid in ids_in_position((ax, ay), model)
        nid == agent.id && continue
        nb = model[nid]
        tot_c += nb.c
        tot_v += nb.v
    end

    # Euclidean disk offsets skip (0,0); clip at map edges without wrapping.
    for (dx, dy) in props.neighbor_offsets
        x_nb = ax + dx
        y_nb = ay + dy
        (1 <= x_nb <= width && 1 <= y_nb <= height) || continue
        for nid in ids_in_position((x_nb, y_nb), model)
            nb = model[nid]
            tot_c += nb.c
            tot_v += nb.v
        end
    end

    u_val = tot_v != 0 ? tot_c / tot_v : 0.0

    return clamp(u_val, 0.0, 1.0)
end

function compute_score!(agent::Household, props)
    score = 0.0
    # ltv
    if agent.ltv <= 0.4;       score += 20
    elseif agent.ltv <= 0.7;   score += 14
    elseif agent.ltv <= 0.9;   score += 7
    end
    # income
    if agent.income <= 300;       score += 12
    elseif agent.income <= 500;   score += 17
    elseif agent.income <= 800;   score += 24
    elseif agent.income <= 1700;  score += 34
    else;                         score += 38
    end
    # seniority
    if agent.sen <= 15;       score += 9
    elseif agent.sen <= 47;   score += 14
    else;                     score += 28
    end
    # zeta / r_cap
    if agent.zeta <= 0.05;       score += 9
    elseif agent.zeta <= 0.40;   score += 22
    elseif agent.zeta <= 0.50;   score += 24
    else;                        score += 33
    end
    # pti
    if agent.pti <= 0.40
        score += 20
    else
        score += 9
    end
    # credit score from rolling default window
    ratio_arr = sum(agent.v_arr) / 36.0
    score += 100.0 * (1.0 - ratio_arr)

    agent.score = score * props.scoring_intensity
end
