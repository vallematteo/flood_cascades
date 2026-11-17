"""
Check that the model in model/ produces exactly the same simulations as the model that
produced the paper (commit b10cf66).

Both models run the same simulations (same seeds, scenarios and parameters), each in its
own Julia process because the two define the same types. Each run writes its full output:

    model_data      macro time series (one row per step)
    agent_data      wealth distribution of the households alive at the last step
    default_data    one row per foreclosure
    distress_data   one row per default trigger
    agents_final    every state variable of every household alive at the last step

All files are written as CSV and compared with `diff`. The paper model also writes the
column `install_scaling` of a removed policy option (equal to 1 in every step); it is
dropped before the comparison.

Run from the repository root:
    julia --project=. tests/check_model_equivalence.jl
"""

const ROOT = abspath(joinpath(@__DIR__, ".."))
const PAPER_COMMIT = "b10cf66"

# One simulation per (setting, seed) at the calibrated parameters, with spillovers on, plus
# one at a different scale of initial funds (zeta_scale, default 0.015). The clean model
# receives zeta_scale only when it differs from the default, so the default is tested too.
const RUNS = [[(setting = s, seed = sd, zeta = 0.015) for s in (0, 4) for sd in (788269981, 1559343818)];
              [(setting = 0, seed = 788269981, zeta = 0.01)]]
const TOT_STEP = 200

# Household state variables present in both models.
const STATE = [:id, :x, :y, :fgrote, :fmiddel, :fklein, :feklein, :D, :u, :t, :s_e, :s_d,
               :l, :income, :sen, :tot_expenditures, :living_expenses, :w, :tau, :ltv, :p, :n,
               :zeta, :pti, :v, :c, :sp, :delta_p, :psi_1, :psi_2, :default_reason, :p_wtp,
               :s_i, :fin_threshold, :n_kappa, :nbox, :c0, :u0, :p_wtp0]

function worker_script(model_dir, out_dir, paper_model::Bool)
    # the paper model takes the options and variants that the clean-up removed,
    # set here to the values used for every simulation in the paper
    extra = paper_model ? """
        level_low_ir = 1.0, gu_max = 1.0, gu_min = 0.9, lambda_sen = 1.0, g = 0.0,
        use_ir_scaling = false, use_score_scaling = false, collect_agent_data = false,
        up_psi = 1.0, psi1_time_dependent = true, revival_all = true,
        rho_mem = 0.0,""" : ""
    return """
    include(joinpath("$model_dir", "model.jl"))
    using CSV, DataFrames, Parquet2
    prep = joinpath("$ROOT", "preparation")
    maps = (; nvm = read_map(joinpath(prep, "flood_maps"), "nvm_cleaned.png"),
              grote = read_map(joinpath(prep, "flood_maps"), "grote.png"),
              middel = read_map(joinpath(prep, "flood_maps"), "middel.png"),
              klein = read_map(joinpath(prep, "flood_maps"), "kleine.png"),
              eklein = read_map(joinpath(prep, "flood_maps"), "extreme_kleine.png"))
    gev = CSV.read(joinpath(prep, "gev", "gev_config_static_2500.csv"), DataFrame)
    for (i, r) in enumerate($RUNS)
        run_dir = joinpath("$out_dir", "run\$i")
        model = create_model(; rng_seed = r.seed, run_id = i, kappa = 10, acpt_score = 200.0,
            settings = r.setting, add_agents = true, rho = 4.0, save_path = run_dir,
            tot_step = $TOT_STEP, beta1 = 1.0, beta2 = 0.5, beta3 = 1.0,
            exp_lb = 1087.0, eta = 0.0382979672588408, vartheta = 0.05, alpha = 150.0,
            m = 5.246293367259204, varepsilon = 0.0024974777754507, s_ub = 1.0, N_nj = 35,
            $extra
            save_data_enabled = true,
            density_map = maps.nvm, m_fgrote = maps.grote, m_fmiddel = maps.middel,
            m_fklein = maps.klein, m_feklein = maps.eklein,
            gev_list = Float64.(gev[!, Symbol("set\$(r.setting)")]),
            lambda = 0.5, delta = 0.1,
            ($paper_model || r.zeta != 0.015 ? (; zeta_scale = r.zeta) : (;))...)
        step!(model, $TOT_STEP)
        for name in ("model_data", "agent_data", "default_data", "distress_data")
            df = DataFrame(Parquet2.Dataset(joinpath(run_dir, name * ".parquet")); copycols = true)
            "install_scaling" in names(df) && select!(df, Not(:install_scaling))
            CSV.write(joinpath(run_dir, name * ".csv"), df)
        end
        ids = sort!(collect(allids(model)))
        state = DataFrame([f => [getfield(model[id], f) for id in ids] for f in $STATE])
        CSV.write(joinpath(run_dir, "agents_final.csv"), state)
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
    for i in eachindex(RUNS), name in ("model_data", "agent_data", "default_data",
                                       "distress_data", "agents_final")
        a = joinpath(work, "out_paper", "run$i", "$name.csv")
        b = joinpath(work, "out_clean", "run$i", "$name.csv")
        same = success(`diff -q $a $b`)
        n_different += !same
        rows = countlines(a) - 1
        println(rpad("run $i (S$(RUNS[i].setting), seed $(RUNS[i].seed), zeta $(RUNS[i].zeta)) $name", 70),
                lpad("$rows rows  ", 14), same ? "identical" : "DIFFERENT")
    end
    println(n_different == 0 ? "\nThe two models produce identical output." :
                               "\n$n_different files differ. Output kept in $work")
    exit(n_different == 0 ? 0 : 1)
end

main()
