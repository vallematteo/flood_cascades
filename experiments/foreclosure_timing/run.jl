"""
    Foreclosure timing: mortgage age and distress history at foreclosure.

    One record per foreclosure at fixed parameter vectors, at the calibrated parametrisation,
    for Fig. 5, SI Figs 11 and 12 and the numbers on foreclosure timing.

    Designs (beta1 = 1, kappa = 10, vartheta = 0.05, delta = 0; scenarios S0..S4;
    lambda in {0.25, 0.5, 0.75}; 10 seeds; run_id is the position in the loop order
    setting, beta2, beta3, lambda, seed):
      gap     beta2 in {0.55, 0.5, 0.45, 0.4, 0.3} (beta1 - beta2 from 0.45 to 0.7),
              beta3 = 1                                                      750 runs
      beta3   beta3 in {0.5, 2.0}, beta2 = 0.5                               300 runs

    Calibrated parametrisation (model/parameters.csv): eta, m, varepsilon from the calibration
    (the parameter combination inside both empirical bands), the other parameters at their
    baseline values.

    Output, in --out-dir, per design:
      <design>_runs.csv           one row per run: parameters, post-warm-up triggers and
                                  foreclosures by channel, mean active population
      <design>_foreclosures.csv   one row per post-warm-up foreclosure: run parameters,
                                  step, mortgage age t, household id, channel, ltv, pti,
                                  wtp, distress months (n_dis, by channel; post-warm-up,
                                  including the terminal one), mortgage age at the first
                                  distress month (t_dis_first) and dis_span = t - t_dis_first + 1
    reduce_foreclosure_timing.py reduces these to the files of data/foreclosure_timing/.

    Usage (from the repository root):
      both designs (about 13 minutes on 10 cores of a laptop):
        julia --project=. experiments/foreclosure_timing/run.jl --num-workers 10
      a few runs, for checks:
        julia --project=. experiments/foreclosure_timing/run.jl --design gap --run-ids 0,300,749 --out-dir output/timing_check
"""

using Distributed

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))

function parse_args(args)
    d = Dict{String,Any}(
        "num_workers" => min(10, Sys.CPU_THREADS),
        "design"      => "all",       # gap, beta3 or all
        "run_ids"     => "",          # only these runs of the design (for checks)
        "out_dir"     => joinpath(ROOT, "output", "foreclosure_timing", "raw"),
    )
    i = 1
    while i <= length(args)
        key = replace(args[i], "--" => "", "-" => "_")
        haskey(d, key) || error("Unknown argument: $(args[i])")
        d[key] = key == "num_workers" ? parse(Int, args[i+1]) : args[i+1]
        i += 2
    end
    isabspath(d["out_dir"]) || (d["out_dir"] = joinpath(ROOT, d["out_dir"]))
    return d
end

const A = parse_args(ARGS)

if A["num_workers"] > 1
    n_to_add = A["num_workers"] - (nprocs() - 1)
    n_to_add > 0 && addprocs(n_to_add; exeflags = ["--project=$(Base.active_project())"])
end
@everywhere const ROOT = $ROOT
@everywhere begin
    include(joinpath(ROOT, "model", "model.jl"))
    using CSV, DataFrames
end
using Printf

# ─── Parameters

const P = model_parameters("fixed", "baseline", "calibrated")
const TOT_STEP = Int(P[:tot_step])
const WARMUP = Int(P[:warmup])

const PARAMS = (
    kappa = Int(P[:kappa]), acpt_score = P[:acpt_score], add_agents = P[:add_agents] == 1,
    rho = P[:rho], alpha = P[:alpha], s_ub = P[:s_ub], N_nj = Int(P[:N_nj]), vartheta = P[:vartheta],
    exp_lb = P[:exp_lb], eta = P[:eta], m = P[:m], varepsilon = P[:varepsilon],
)

const SEEDS = [788269981, 1559343818, 850678639, 36298517, 2042083684,
               2106254254, 1321925152, 1556075696, 66259485, 1168686712]

const DESIGNS = Dict(
    "gap"   => (beta1 = [1.0], beta2 = [0.55, 0.5, 0.45, 0.4, 0.3], beta3 = [1.0]),
    "beta3" => (beta1 = [1.0], beta2 = [0.5], beta3 = [0.5, 2.0]),
)
const SETTINGS = [0, 1, 2, 3, 4]
const LAMBDAS = [0.25, 0.5, 0.75]
const DELTAS = [0.0]

# ─── One run

@everywhere function run_one(p, maps, gev_list, par, tot_step, warmup)
    model = create_model(;
        rng_seed = p.seed, run_id = p.run_id,
        kappa = par.kappa, acpt_score = par.acpt_score, setting = p.setting,
        add_agents = par.add_agents, rho = par.rho, save_path = "", tot_step = tot_step,
        beta1 = p.beta1, beta2 = p.beta2, beta3 = p.beta3,
        exp_lb = par.exp_lb, eta = par.eta, vartheta = par.vartheta, alpha = par.alpha,
        m = par.m, varepsilon = par.varepsilon, s_ub = par.s_ub, N_nj = par.N_nj,
        save_data_enabled = false,
        density_map = maps.nvm, map_minor = maps.map_minor, map_moderate = maps.map_moderate,
        map_severe = maps.map_severe, map_catastrophic = maps.map_catastrophic, gev_list = gev_list,
        lambda = p.lambda, delta = p.delta,
    )

    pop_sum = 0.0; pop_n = 0
    for step in 1:tot_step
        model_step!(model)
        if step > warmup
            pop_sum += abmproperties(model).tot_num_agents_active
            pop_n += 1
        end
    end

    props = abmproperties(model)
    dc, fc = props.distress_data, props.failures_data

    # Post-warm-up trigger and foreclosure totals by channel. Every trigger is resolved in its
    # step, so triggers minus foreclosures are the forbearances.
    n_trig_fin = n_trig_strat = n_fcl_fin = n_fcl_strat = 0
    for k in eachindex(dc.step)
        (warmup <= dc.step[k] < tot_step) || continue
        dc.default_reason[k] == FINANCIAL ? (n_trig_fin += 1) : (n_trig_strat += 1)
    end
    for k in eachindex(fc.step)
        (warmup <= fc.step[k] < tot_step) || continue
        fc.default_reason[k] == FINANCIAL ? (n_fcl_fin += 1) : (n_fcl_strat += 1)
    end

    # Post-warm-up foreclosures; fc.t is the age of the mortgage in months.
    keep = [k for k in eachindex(fc.step) if fc.step[k] >= warmup]
    fcl_id = fc.id[keep]

    # Distress months from the warm-up cutoff on, per foreclosed household. Histories of
    # households present at the cutoff are left-truncated.
    nk = length(keep)
    n_dis = zeros(Int, nk); n_dis_fin = zeros(Int, nk); n_dis_strat = zeros(Int, nk)
    first_t = fill(typemax(Int), nk)
    slot = Dict(fcl_id[j] => j for j in 1:nk)
    for k in eachindex(dc.id)
        dc.step[k] < warmup && continue
        j = get(slot, dc.id[k], 0)
        j == 0 && continue
        n_dis[j] += 1
        dc.default_reason[k] == FINANCIAL ? (n_dis_fin[j] += 1) : (n_dis_strat[j] += 1)
        first_t[j] = min(first_t[j], dc.t[k])
    end
    t_dis_first = [ft == typemax(Int) ? -1 : ft for ft in first_t]

    fcl = DataFrame(
        run_id = fill(p.run_id, nk), setting = fill(p.setting, nk), seed = fill(p.seed, nk),
        beta1 = fill(p.beta1, nk), beta2 = fill(p.beta2, nk), beta3 = fill(p.beta3, nk),
        lambda = fill(p.lambda, nk), delta = fill(p.delta, nk),
        step = fc.step[keep], t = fc.t[keep], id = fcl_id,
        default_reason = fc.default_reason[keep],
        ltv = fc.ltv[keep], pti = fc.pti[keep], p_wtp = fc.p_wtp[keep],
        n_dis = n_dis, n_dis_fin = n_dis_fin, n_dis_strat = n_dis_strat,
        t_dis_first = t_dis_first,
    )
    fcl.dis_span = [tf < 0 ? -1 : t - tf + 1 for (t, tf) in zip(fcl.t, fcl.t_dis_first)]

    run = (
        success = true, run_id = p.run_id, setting = p.setting, seed = p.seed,
        beta1 = p.beta1, beta2 = p.beta2, beta3 = p.beta3,
        lambda = p.lambda, delta = p.delta,
        eta = par.eta, m = par.m, varepsilon = par.varepsilon,
        n_trig_fin = n_trig_fin, n_trig_strat = n_trig_strat,
        n_fcl_fin = n_fcl_fin, n_fcl_strat = n_fcl_strat,
        n_fcl_tot = n_fcl_fin + n_fcl_strat,
        mean_active = pop_n > 0 ? pop_sum / pop_n : NaN,
        agents_end = props.tot_num_agents_active,
    )
    return run, fcl
end

# ─── Main

function design_jobs(g)
    jobs = NamedTuple[]
    rid = 0
    for s in SETTINGS, b1 in g.beta1, b2 in g.beta2, b3 in g.beta3, lam in LAMBDAS, dl in DELTAS, sd in SEEDS
        push!(jobs, (run_id = rid, setting = s, seed = sd, beta1 = b1, beta2 = b2, beta3 = b3,
                     lambda = lam, delta = dl))
        rid += 1
    end
    return jobs
end

function main()
    mkpath(A["out_dir"])
    prep = joinpath(ROOT, "preparation")
    maps = (; map_minor  = read_map(joinpath(prep, "flood_maps"), "grote.png"),
              map_moderate = read_map(joinpath(prep, "flood_maps"), "middel.png"),
              map_severe  = read_map(joinpath(prep, "flood_maps"), "kleine.png"),
              map_catastrophic = read_map(joinpath(prep, "flood_maps"), "extreme_kleine.png"),
              nvm       = read_map(joinpath(prep, "flood_maps"), "nvm_cleaned.png"))
    gev_df = CSV.read(joinpath(prep, "gev", "gev_config_static_2500.csv"), DataFrame)
    gev = Dict(s => Float64.(gev_df[!, Symbol("set$s")]) for s in SETTINGS)

    names = A["design"] == "all" ? ["gap", "beta3"] : [A["design"]]
    for name in names
        jobs = design_jobs(DESIGNS[name])
        if !isempty(strip(A["run_ids"]))
            ids = Set(parse.(Int, split(A["run_ids"], ",")))
            jobs = [j for j in jobs if j.run_id in ids]
        end
        t0 = time()
        println("Design $name: $(length(jobs)) runs")
        results = pmap(j -> run_one(j, maps, gev[j.setting], PARAMS, TOT_STEP, WARMUP), jobs)
        runs = DataFrame([r[1] for r in results])
        fcl = reduce(vcat, (r[2] for r in results))
        fcl.warmup = fill(WARMUP, nrow(fcl))
        CSV.write(joinpath(A["out_dir"], "$(name)_runs.csv"), runs)
        CSV.write(joinpath(A["out_dir"], "$(name)_foreclosures.csv"), fcl)
        @printf("Design %s done in %.1fs: %d foreclosures\n", name, time() - t0, nrow(fcl))
    end
end

main()
