"""
    Household agent of the flood mortgage risk agent-based model.
"""

using Agents, Distributions

@agent struct Household(GridAgent{2})
    # location (same as pos, kept for map indexing)
    x::Int          # column
    y::Int          # row
    # flood damage of the location under each flood map
    fgrote::Float64
    fmiddel::Float64
    fklein::Float64
    feklein::Float64
    D::Float64              # damage from the current flood
    u::Float64              # neighbourhood utility U
    t::Int                  # age of the mortgage (months)
    # status
    s_e::Int                # employment status (1 = employed, 0 = unemployed)
    s_d::Int                # default status (0 = paying, 1 = defaulted)
    # financial
    l::Float64              # outstanding loan
    income::Float64
    sen::Float64            # seniority
    tot_expenditures::Float64
    living_expenses::Float64
    w::Float64              # wealth (liquidity)
    tau::Int                # months unemployed
    ltv::Float64
    p::Float64              # monthly installment
    n::Float64              # remaining installments
    zeta::Float64           # initial funds / loan size
    pti::Float64            # payment-to-income ratio
    v::Float64              # house value
    c::Float64              # collateral
    sp::Float64             # floor area of the property
    delta_p::Float64        # flood loss of house value, lambda D v
    psi_1::Float64          # strategic-default threshold (redrawn every month)
    psi_2::Float64          # employment threshold (fixed at entry)
    default_reason::Int     # 1 = financial, -1 = strategic, 0 = forborne, 3 = none
    p_wtp::Float64          # willingness to pay
    s_i::Float64            # share of income spent on living expenses
    fin_threshold::Float64  # wealth left after living expenses
    # foreclosure spillover, set for every household in the damage phase (cascade_phase!)
    n_kappa::Float64        # foreclosures of t-1 within kappa
    nbox::Int               # households in the kappa-disk this month
    c0::Float64             # collateral without the foreclosure discount, v (1 - lambda D_t)
    u0::Float64             # neighbourhood utility from c0
    p_wtp0::Float64         # wtp from u0, same debt pressure
end

"""
    create_household(model, x, y, ...) -> Household

Construct a Household and add it to the model at grid position (x, y).
"""
function create_household(model, x::Int, y::Int,
                          fgrote, fmiddel, fklein, feklein,
                          zeta, income, sen, living_exp,
                          w, ltv, p, v, sp, n, l)
    # two independent draws: psi_1 sets the strategic-default threshold and is redrawn
    # every month; psi_2 sets the employment type and stays fixed
    psi_1_val = rand(abmrng(model), Uniform(0.0, 1.0))
    psi_2_val = rand(abmrng(model), Uniform(0.0, 1.0))
    pti_val = p / income

    agent = add_agent!((x, y), Household, model;
        x = x,
        y = y,
        fgrote = Float64(fgrote),
        fmiddel = Float64(fmiddel),
        fklein = Float64(fklein),
        feklein = Float64(feklein),
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
        n = Float64(n),
        zeta = Float64(zeta),
        pti = pti_val,
        v = Float64(v),
        c = Float64(v),
        sp = Float64(sp),
        delta_p = 0.0,
        psi_1 = psi_1_val,
        psi_2 = psi_2_val,
        default_reason = 3,
        p_wtp = 0.0,
        s_i = 0.0,
        fin_threshold = 0.0,
        n_kappa = 0.0,
        nbox = 1,
        c0 = Float64(v),
        u0 = 1.0,
        p_wtp0 = 0.0,
    )
    return agent
end


# ─── Household step (called every month)

"""
    agent_step!(agent, model)

Monthly update of one household: strategic threshold, age, seniority, default and
forbearance, expenditure and payment, wealth, income and employment.
"""
function agent_step!(agent::Household, model)
    props = abmproperties(model)

    # redraw the strategic threshold at the top of the month, so the default trigger
    # and the forbearance check (update_s_d!) see the same value; psi_2 (employment)
    # is not redrawn, so unemployment spells persist
    agent.psi_1 = rand(abmrng(model), Uniform(0.0, 1.0))

    agent.t += 1

    # D, c and u were set for every household by cascade_phase! before any household steps

    # seniority grows by one month while employed, up to alpha, and is frozen otherwise
    if agent.s_e == 1
        agent.sen = min(props.alpha_param, agent.sen + 1)
    end

    # unemployment duration
    agent.tau = agent.s_e == 1 ? 0 : (agent.tau + 1)

    # default and forbearance
    triggered_reason = update_s_d!(agent, model)

    # living expenses
    living_exp = max(agent.s_i * agent.income, props.exp_lb)
    agent.living_expenses = living_exp
    agent.tot_expenditures = living_exp + (1 - agent.s_d) * agent.p

    # remaining installments
    agent.n -= (1 - agent.s_d)

    # wealth and loan
    agent.w = agent.w + agent.income - agent.tot_expenditures
    agent.l = agent.n * agent.p

    # income and employment
    update_income!(agent, model)

    # derived quantities
    agent.ltv = agent.l / agent.c
    agent.pti = agent.p / (agent.income + 0.01)

    # Record every trigger (s_d flipped 0 -> 1 this month), whether the household was
    # forborne or forecloses. Recorded here, once ltv, pti and s_e are current, so the
    # state columns match those recorded at foreclosure (see DistressDataCollector).
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


# ─── Sub-routines

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
Set this month's flood damage D and the collateral c = v (1 - lambda D_t) (1 - delta)^n_kappa.
n_kappa must already be set for this month (cascade_phase!).
"""
function update_damage!(agent::Household, props)
    find_flood_map!(agent, props)
    agent.delta_p = props.lambda * agent.D * agent.v
    agent.c0 = max(agent.v - agent.delta_p, 0.001)
    agent.c = agent.c0 * (1 - props.delta)^agent.n_kappa
end

function update_s_d!(agent::Household, model)
    props = abmproperties(model)
    agent.s_i = calc_share_expenses(agent.income, props.m_param, props.varepsilon, props.s_ub)
    agent.fin_threshold = agent.w - max(agent.s_i * agent.income, props.exp_lb)

    # u and u0 were set for every household by cascade_phase!

    # Debt pressure X = (loan / annual income) * (tau + 1). The loan is a stock, so it is
    # compared with annual income, as in a loan-to-income ratio. tau is 0 while
    # employed (update_income!), so X is the loan-to-income ratio for employed households
    # and grows linearly with the months of unemployment otherwise.
    income_for_calc = agent.income > 0 ? agent.income * 12 : 1e-6
    wtp_fin = (agent.l / income_for_calc) * (agent.tau + 1)

    # Willingness to pay, wtp = beta1 U - beta2 X / (beta3 + X). The penalty is zero
    # without debt, increases with X and approaches beta2; beta3 is the debt pressure at
    # which it reaches beta2 / 2. wtp is clamped to [0, 1]: under the trigger
    # (wtp < psi_1) and forbearance (wtp > 1 - psi_1) rules a wtp below 0 acts as 0 and
    # one above 1 acts as 1, so clamping only makes wtp a probability.
    agent.p_wtp = clamp(props.beta1 * agent.u - props.beta2 * wtp_fin / (props.beta3 + wtp_fin), 0.0, 1.0)
    # wtp without the foreclosure discount, used only to attribute foreclosures to the spillover
    agent.p_wtp0 = clamp(props.beta1 * agent.u0 - props.beta2 * wtp_fin / (props.beta3 + wtp_fin), 0.0, 1.0)

    # financial default: wealth after living expenses no longer covers the installment
    if agent.s_d == 0 && agent.fin_threshold <= agent.p
        agent.s_d = 1
        agent.default_reason = 1
    end

    # strategic default
    if agent.s_d == 0 && agent.p_wtp < agent.psi_1
        agent.s_d = 1
        agent.default_reason = -1
    end

    # Every household enters the month with s_d == 0, so s_d == 1 here is a new trigger.
    # Returned rather than recorded here, because forbearance (below) resets
    # default_reason and because ltv, pti and s_e are not yet current for this month.
    triggered_reason = agent.s_d == 1 ? agent.default_reason : 0

    # forbearance: a defaulter of either channel is forborne when wtp > 1 - psi_1
    if agent.s_d == 1 && agent.p_wtp > (1 - agent.psi_1)
        if agent.default_reason == 1
            props.num_revival_financial_current += 1
        elseif agent.default_reason == -1
            props.num_revival_strategic_current += 1
        end
        agent.s_d = 0
        agent.default_reason = 0
        props.num_revival_current += 1
    end
    return triggered_reason
end

function update_income!(agent::Household, model)
    props = abmproperties(model)
    old_s_e = agent.s_e
    update_s_e!(agent, props)
    new_s_e = agent.s_e

    # income is constant while employed
    if new_s_e == 0 && old_s_e == 1
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
    else  # gev <= 0: no flood
        agent.D = 0.0
    end
end

"""
Household-by-household computation of U, the reference against which the grid sums
of `cascade_phase!` are tested. The model itself does not call it.
"""
function neighbor_utility(agent::Household, model)
    props = abmproperties(model)
    # the household itself is always included
    tot_c = agent.c
    tot_v = agent.v
    ax, ay = agent.x, agent.y
    width  = size(props.density_map, 2)
    height = size(props.density_map, 1)

    # other households in the same cell
    for nid in ids_in_position((ax, ay), model)
        nid == agent.id && continue
        nb = model[nid]
        tot_c += nb.c
        tot_v += nb.v
    end

    # the Euclidean disk offsets skip (0, 0) and are clipped at the map edges
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
