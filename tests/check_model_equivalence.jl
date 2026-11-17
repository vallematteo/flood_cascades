"""
Check that the model in model/ produces exactly the same simulations as the model that
produced the paper (commit b10cf66, before the code clean-up).

Both models run the same short simulations (same seeds, scenarios and parameters, in
separate Julia processes, since the two define the same types). The script then compares
every foreclosure record and every distress record, value by value.

Run from the repository root:
    julia --project=. tests/check_model_equivalence.jl

Runtime: a few minutes (2 models x 4 runs x 150 steps).
"""

const ROOT = abspath(joinpath(@__DIR__, ".."))
const PAPER_COMMIT = "b10cf66"

# One short simulation per (setting, seed), at the calibrated parameters, with the
# model-variant switches used for every simulation in the paper.
const RUNS = [(setting = s, seed = sd) for s in (0, 4) for sd in (788269981, 1559343818)]
const TOT_STEP = 150

function worker_script(model_dir, out_dir, paper_model::Bool)
    # the paper model still takes the keyword arguments of the removed policy options
    extra = paper_model ? """
        level_low_ir = 1.0, gu_max = 1.0, gu_min = 0.9,
        use_ir_scaling = false, use_score_scaling = false,""" : ""
    return """
    include(joinpath("$model_dir", "model.jl"))
    using CSV, DataFrames
    prep = joinpath("$ROOT", "preparation")
    maps = (; nvm = read_map(joinpath(prep, "flood_maps"), "nvm_cleaned.png"),
              grote = read_map(joinpath(prep, "flood_maps"), "grote.png"),
              middel = read_map(joinpath(prep, "flood_maps"), "middel.png"),
              klein = read_map(joinpath(prep, "flood_maps"), "kleine.png"),
              eklein = read_map(joinpath(prep, "flood_maps"), "extreme_kleine.png"))
    gev = CSV.read(joinpath(prep, "gev", "gev_config_static_2500.csv"), DataFrame)
    for (i, r) in enumerate($RUNS)
        model = create_model(; rng_seed = r.seed, run_id = i, kappa = 10, acpt_score = 200.0,
            settings = r.setting, add_agents = true, rho = 4.0, save_path = "$out_dir",
            tot_step = $TOT_STEP, beta1 = 1.0, beta2 = 0.5, beta3 = 1.0, lambda_sen = 1.0,
            exp_lb = 1087.0, eta = 0.0382979672588408, vartheta = 0.05, g = 0.0, alpha = 150.0,
            m = 5.246293367259204, varepsilon = 0.0024974777754507, s_ub = 1.0, N_nj = 35,
            $extra
            collect_agent_data = false, save_data_enabled = false,
            density_map = maps.nvm, m_fgrote = maps.grote, m_fmiddel = maps.middel,
            m_fklein = maps.klein, m_feklein = maps.eklein,
            gev_list = Float64.(gev[!, Symbol("set\$(r.setting)")]),
            up_psi = 1.0, psi1_time_dependent = true, zeta_scale = 0.015, revival_all = true,
            lambda = 0.5, delta = 0.1, rho_mem = 0.0)
        step!(model, $TOT_STEP)
        props = abmproperties(model)
        fc = collector_to_dataframe(props.failures_data)
        dc = collector_to_dataframe(props.distress_data)
        CSV.write(joinpath("$out_dir", "failures_\$i.csv"), fc)
        CSV.write(joinpath("$out_dir", "distress_\$i.csv"), dc)
        CSV.write(joinpath("$out_dir", "population_\$i.csv"),
                  DataFrame(active = [props.tot_num_agents_active], ever = [props.tot_num_agents_his]))
    end
    """
end

function run_model(model_dir, out_dir, paper_model)
    mkpath(out_dir)
    script = joinpath(out_dir, "worker.jl")
    write(script, worker_script(model_dir, out_dir, paper_model))
    run(`$(Base.julia_cmd()) --project=$ROOT $script`)
end

function main()
    work = mktempdir()
    paper_dir = joinpath(work, "paper_model")
    mkpath(paper_dir)
    for f in ("model.jl", "agents.jl", "utils.jl")
        write(joinpath(paper_dir, f), read(`git -C $ROOT show $PAPER_COMMIT:model/$f`))
    end
    run_model(paper_dir, joinpath(work, "out_paper"), true)
    run_model(joinpath(ROOT, "model"), joinpath(work, "out_clean"), false)

    n_different = 0
    for i in eachindex(RUNS), name in ("failures", "distress", "population")
        a = read(joinpath(work, "out_paper", "$(name)_$i.csv"), String)
        b = read(joinpath(work, "out_clean", "$(name)_$i.csv"), String)
        same = a == b
        n_different += !same
        println(rpad("run $i $(RUNS[i]) $name", 60), same ? "identical" : "DIFFERENT")
    end
    println(n_different == 0 ? "\nThe two models produce identical simulations." :
                               "\n$n_different files differ.")
    exit(n_different == 0 ? 0 : 1)
end

main()
