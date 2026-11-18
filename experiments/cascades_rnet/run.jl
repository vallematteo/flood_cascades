"""
    Foreclosure cascades: flood scenario x spillover intensity, at the calibrated parametrisation.

    Every run with delta > 0 is paired with the delta = 0 run of the same cell and seed,
    which gives the net reproduction number of the foreclosure feedback,
        R_net = 1 - rate(delta = 0) / rate(delta),
    and the foreclosure records needed to reconstruct the cascades.

    Design (65,520 runs; run_id is the position in this order):
      cells       (S0, lambda = 0.5) and S1..S4 x lambda in {0.25, 0.5, 0.75}: 13 cells
      main block  beta2 = 0.5 (beta1 - beta2 = 0.5), kappa in {3, 7, 10, 15, 18}
      gap block   beta2 in {0.4, 0.3} (beta1 - beta2 = 0.6, 0.7), kappa = 10
      both        vartheta in {0.03, 0.05, 0.13},
                  delta in {0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.1125, 0.125, 0.15, 0.175, 0.2, 0.25},
                  20 seeds
      13 cells x (5 + 2) x 3 x 12 x 20 = 65,520 runs.
    Floods do not occur at S0, so lambda and vartheta have no effect there.

    Calibrated parametrisation (model/parameters.csv): eta, m, varepsilon from the calibration
    (the parameter combination inside both empirical bands), the other parameters at their
    baseline values.

    Seeds: np.random.default_rng(56).integers(0, 2**31, size=20).

    Output, in --out-dir, one chunk per cell (chunk c = cell c):
      runs_chunk{c}.csv   one row per run: parameters, triggers and foreclosures by channel,
                          foreclosures attributed to the spillover, mean active population,
                          annual rates per 100 active mortgages, median ages, mean disk size
      fcl_chunk{c}.csv    one row per foreclosure from step warmup - 1 on: run_id, step, x, y,
                          channel (1 = financial), attr (1 = attributed to the spillover),
                          flooded (D > 0)

    Attribution: a foreclosure is attributed to the spillover when, with the same psi, it would
    not have happened at the wtp without the foreclosure discount (wtp0):
      financial: psi > 1 - wtp0          (forborne without the discount)
      strategic: not (wtp0 < psi <= 1 - wtp0)

    Usage (from the repository root):
      full design on a cluster node:  experiments/cascades_rnet/slurm_run.sh
      a few runs, for checks:
        julia --project=. experiments/cascades_rnet/run.jl --run-ids 0,1,2 --out-dir output/cascades_check
"""

using Distributed

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))

function parse_args(args)
    d = Dict{String,Any}(
        "num_workers" => min(10, Sys.CPU_THREADS),
        "chunks"      => join(0:12, ","),   # cells to run, one after the other
        "run_ids"     => "",                # only these runs (for checks); empty = full chunks
        "out_dir"     => joinpath(ROOT, "output", "cascades_rnet", "raw"),
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
    using Statistics: mean, median
end
using Printf, Dates

# ─── Parameters

const P = model_parameters("fixed", "baseline", "calibrated")
const TOT_STEP = Int(P[:tot_step])
const WARMUP = Int(P[:warmup])

const PARAMS = (
    beta1 = P[:beta1], beta3 = P[:beta3], acpt_score = P[:acpt_score], add_agents = P[:add_agents] == 1,
    rho = P[:rho], alpha = P[:alpha], s_ub = P[:s_ub], N_nj = Int(P[:N_nj]), exp_lb = P[:exp_lb],
    eta = P[:eta], m = P[:m], varepsilon = P[:varepsilon],
)

const SEEDS = [788269981, 1559343818, 850678639, 36298517, 2042083684,
               2106254254, 1321925152, 1556075696, 66259485, 1168686712,
               663926602, 505990832, 968270353, 1302730519, 346894959,
               771529683, 2126550487, 1264868378, 1650418743, 560548168]

const SETTINGS = [0, 1, 2, 3, 4]
const LAMBDAS = [0.25, 0.5, 0.75]
const S0_LAMBDA = 0.5
const DELTAS = [0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.1125, 0.125, 0.15, 0.175, 0.2, 0.25]
const KAPPAS = [3.0, 7.0, 10.0, 15.0, 18.0]      # main block
const BETA2_MAIN = 0.5
const GAP_BETA2 = [0.4, 0.3]                     # gap block, at kappa = 10
const GAP_KAPPA = 10.0
const VARTHETAS = [0.03, 0.05, 0.13]

# ─── Design

"""Full design in a fixed order; run_id is the position. Returns (jobs, cells)."""
function build_design()
    cells = Tuple{Int,Float64}[]
    for s in SETTINGS
        s == 0 ? push!(cells, (0, S0_LAMBDA)) : append!(cells, [(s, l) for l in LAMBDAS])
    end
    jobs = NamedTuple[]
    rid = 0
    for (ci, (s, lam)) in enumerate(cells)
        blocks = vcat([("main", k, BETA2_MAIN) for k in KAPPAS], [("gap", GAP_KAPPA, b2) for b2 in GAP_BETA2])
        for (blk, k, b2) in blocks, vt in VARTHETAS, dl in DELTAS, (si, sd) in enumerate(SEEDS)
            push!(jobs, (run_id = rid, cell = ci - 1, block = blk, setting = s, lambda = lam,
                         kappa = k, beta2 = b2, vartheta = vt, delta = Float64(dl),
                         seed = sd, seed_index = si - 1))
            rid += 1
        end
    end
    return jobs, cells
end

# ─── One run

@everywhere function run_one(p, maps, gev_list, par, tot_step, warmup)
    t0 = time()
    model = create_model(;
        rng_seed = p.seed, run_id = p.run_id,
        kappa = Int(round(p.kappa)), acpt_score = par.acpt_score, settings = p.setting,
        add_agents = par.add_agents, rho = par.rho, save_path = "", tot_step = tot_step,
        beta1 = par.beta1, beta2 = p.beta2, beta3 = par.beta3,
        exp_lb = par.exp_lb, eta = par.eta, vartheta = p.vartheta, alpha = par.alpha,
        m = par.m, varepsilon = par.varepsilon, s_ub = par.s_ub, N_nj = par.N_nj,
        save_data_enabled = false,
        density_map = maps.nvm, m_fgrote = maps.m_fgrote, m_fmiddel = maps.m_fmiddel,
        m_fklein = maps.m_fklein, m_feklein = maps.m_feklein, gev_list = gev_list,
        lambda = p.lambda, delta = p.delta,
    )

    nT = tot_step - warmup
    active = zeros(Float64, nT)
    for step in 1:tot_step
        model_step!(model)
        step > warmup && (active[step - warmup] = abmproperties(model).tot_num_agents_active)
    end

    props = abmproperties(model)
    dc, fc = props.distress_data, props.failures_data
    inwin(s) = warmup <= s < tot_step     # post-warm-up window

    n_trig_fin = n_trig_strat = 0
    for k in eachindex(dc.step)
        inwin(dc.step[k]) || continue
        dc.default_reason[k] == 1 ? (n_trig_fin += 1) : (n_trig_strat += 1)
    end

    n_fcl_fin = n_fcl_strat = n_attr_fin = n_attr_strat = 0
    pi_sum = 0.0; nbox_sum = 0.0
    age_fin = Int[]; age_strat = Int[]
    for k in eachindex(fc.step)
        inwin(fc.step[k]) || continue
        fin = fc.default_reason[k] == 1
        psi, w, w0 = fc.psi[k], fc.p_wtp[k], fc.p_wtp0[k]
        attr = fin ? (psi > 1 - w0) : !((w0 < psi) && (psi <= 1 - w0))
        # expected share of the foreclosure probability due to the discount
        p1 = fin ? 1 - w  : max(0.0, 1 - 2w)
        p0 = fin ? 1 - w0 : max(0.0, 1 - 2w0)
        pi_sum += p1 > 0 ? max(p1 - p0, 0.0) / p1 : 0.0
        nbox_sum += fc.nbox[k]
        if fin
            n_fcl_fin += 1; n_attr_fin += attr; push!(age_fin, fc.t[k])
        else
            n_fcl_strat += 1; n_attr_strat += attr; push!(age_strat, fc.t[k])
        end
    end

    n_tot = n_fcl_fin + n_fcl_strat
    mean_active = mean(active)
    per100(n) = 100 * n / mean_active / (nT / 12)

    run = (
        success = true, run_id = p.run_id, block = p.block, setting = p.setting,
        lambda = p.lambda, kappa = p.kappa, beta2 = p.beta2, vartheta = p.vartheta,
        delta = p.delta, seed = p.seed, seed_index = p.seed_index,
        eta = par.eta, m = par.m, varepsilon = par.varepsilon,
        n_trig_fin = n_trig_fin, n_trig_strat = n_trig_strat,
        n_fcl_fin = n_fcl_fin, n_fcl_strat = n_fcl_strat, n_fcl_tot = n_tot,
        n_attr_fin = n_attr_fin, n_attr_strat = n_attr_strat,
        n_attr_tot = n_attr_fin + n_attr_strat, pi_sum = pi_sum,
        mean_active = mean_active, agents_end = props.tot_num_agents_active,
        rate_fin = per100(n_fcl_fin), rate_strat = per100(n_fcl_strat), rate_tot = per100(n_tot),
        median_age_fin = isempty(age_fin) ? NaN : median(age_fin),
        median_age_strat = isempty(age_strat) ? NaN : median(age_strat),
        mean_nbox_fcl = n_tot > 0 ? nbox_sum / n_tot : NaN,
        elapsed = time() - t0,
    )

    # Foreclosures from step warmup - 1 on: every member of a cascade rooted in the window is
    # in the window, and children at step warmup need the step warmup - 1 candidates.
    keep = findall(s -> s >= warmup - 1, fc.step)
    fin_all = fc.default_reason[keep] .== 1
    attr_all = [fin_all[n] ? (fc.psi[k] > 1 - fc.p_wtp0[k]) :
                !((fc.p_wtp0[k] < fc.psi[k]) && (fc.psi[k] <= 1 - fc.p_wtp0[k]))
                for (n, k) in enumerate(keep)]
    fcl = DataFrame(run_id = fill(p.run_id, length(keep)), step = fc.step[keep],
                    x = fc.x[keep], y = fc.y[keep], channel = Int.(fin_all),
                    attr = Int.(attr_all), flooded = Int.(fc.D[keep] .> 0))
    return run, fcl
end

# ─── Main

function run_chunk(c, jobs, maps, gev)
    t0 = time()
    println("Chunk $c: $(length(jobs)) runs")
    results = pmap(j -> run_one(j, maps, gev[j.setting], PARAMS, TOT_STEP, WARMUP), jobs)
    runs = DataFrame([r[1] for r in results])
    runs.warmup .= WARMUP; runs.tot_step .= TOT_STEP
    runs.run_date .= string(Dates.today())
    CSV.write(joinpath(A["out_dir"], "fcl_chunk$(c).csv"), reduce(vcat, (r[2] for r in results)))
    CSV.write(joinpath(A["out_dir"], "runs_chunk$(c).csv"), runs)
    @printf("Chunk %s done in %.1fs\n", c, time() - t0)
end

function main()
    all_jobs, cells = build_design()
    mkpath(A["out_dir"])
    prep = joinpath(ROOT, "preparation")
    maps = (; m_fgrote  = read_map(joinpath(prep, "flood_maps"), "grote.png"),
              m_fmiddel = read_map(joinpath(prep, "flood_maps"), "middel.png"),
              m_fklein  = read_map(joinpath(prep, "flood_maps"), "kleine.png"),
              m_feklein = read_map(joinpath(prep, "flood_maps"), "extreme_kleine.png"),
              nvm       = read_map(joinpath(prep, "flood_maps"), "nvm_cleaned.png"))
    gev_df = CSV.read(joinpath(prep, "gev", "gev_config_static_2500.csv"), DataFrame)
    gev = Dict(s => Float64.(gev_df[!, Symbol("set$s")]) for s in SETTINGS)

    if !isempty(strip(A["run_ids"]))
        ids = Set(parse.(Int, split(A["run_ids"], ",")))
        jobs = [j for j in all_jobs if j.run_id in ids]
        run_chunk("_check", jobs, maps, gev)
        return
    end
    println("Design: $(length(all_jobs)) runs in $(length(cells)) cells; output $(A["out_dir"])")
    for c in parse.(Int, split(A["chunks"], ","))
        run_chunk(c, [j for j in all_jobs if j.cell == c], maps, gev)
    end
end

main()
