"""
    Simulation runner of the global sensitivity analysis and of the calibration.

    Reads a run list (one row per run: run_id, setting, seed and the sampled parameters) and
    runs each simulation with the model in model/, writing its full output as Parquet files
    to --raw-dir:
      model_data_run<id>.parquet     macro series (active mortgages per step)
      agent_data_run<id>.parquet     wealth distribution at the last step
      distress_data_run<id>.parquet  one row per default trigger
      default_data_run<id>.parquet   one row per foreclosure
    plus run_log_<chunk>.csv with the run time and peak memory of every run.

    Run lists:
      sensitivity analysis  output/global_sa/run_list.csv       (experiments/global_sa/design.py, 166,400 runs)
      calibration           output/abc_calibration/run_list.csv (experiments/abc_calibration/design.py, 10,010 runs)
      zeta_scale test       output/zeta_scale/run_list.csv      (experiments/zeta_scale/design.py, 505 runs)

    Parameters not in the run list take their values in model/parameters.csv (set "fixed");
    so does zeta_scale, unless the run list has a zeta_scale column.

    Usage (from the repository root):
      one chunk of a run list on a cluster node:  experiments/global_sa/slurm_run.sh
      the runs of a few parameter vectors, for checks:
        julia --project=. experiments/global_sa/run.jl --sample-ids 0,1 --raw-dir output/global_sa_check
"""

using Distributed

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))

function parse_args(args)
    d = Dict{String,Any}(
        "run_list"    => joinpath(ROOT, "output", "global_sa", "run_list.csv"),
        "raw_dir"     => joinpath(ROOT, "output", "global_sa", "raw"),
        "chunk"       => 0,
        "num_chunks"  => 1,
        "sample_ids"  => "",      # only the runs of these parameter vectors (for checks)
        "num_workers" => min(10, Sys.CPU_THREADS),
    )
    i = 1
    while i <= length(args)
        key = replace(args[i], "--" => "", "-" => "_")
        haskey(d, key) || error("Unknown argument: $(args[i])")
        d[key] = key in ("chunk", "num_chunks", "num_workers") ? parse(Int, args[i+1]) : args[i+1]
        i += 2
    end
    for k in ("run_list", "raw_dir")
        isabspath(d[k]) || (d[k] = joinpath(ROOT, d[k]))
    end
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

# Parameters not in the run list, from model/parameters.csv.
const P = model_parameters("fixed")
const TOT_STEP = Int(P[:tot_step])
const FIXED = (acpt_score = P[:acpt_score], add_agents = P[:add_agents] == 1, rho = P[:rho], alpha = P[:alpha],
               s_ub = P[:s_ub], N_nj = Int(P[:N_nj]), zeta_scale = P[:zeta_scale])

# ─── One run

@everywhere function run_one(p, maps, gev_list, fixed, raw_dir, tot_step)
    t0 = time()
    model = create_model(;
        rng_seed = Int(p.seed), run_id = Int(p.run_id),
        kappa = Int(round(p.kappa)), acpt_score = fixed.acpt_score, settings = Int(p.setting),
        add_agents = fixed.add_agents, rho = fixed.rho, save_path = raw_dir, tot_step = tot_step,
        beta1 = Float64(p.beta1), beta2 = Float64(p.beta2), beta3 = Float64(p.beta3),
        exp_lb = Float64(p.exp_lb), eta = Float64(p.eta), vartheta = Float64(p.vartheta),
        alpha = fixed.alpha, m = Float64(p.m), varepsilon = Float64(p.varepsilon),
        s_ub = fixed.s_ub, N_nj = fixed.N_nj,
        file_suffix = "_run" * lpad(Int(p.run_id), 7, '0'),
        density_map = maps.nvm, m_fgrote = maps.m_fgrote, m_fmiddel = maps.m_fmiddel,
        m_fklein = maps.m_fklein, m_feklein = maps.m_feklein, gev_list = gev_list,
        lambda = Float64(p.lambda), delta = Float64(p.delta),
        zeta_scale = hasproperty(p, :zeta_scale) ? Float64(p.zeta_scale) : fixed.zeta_scale,
    )
    for _ in 1:tot_step
        model_step!(model)        # writes the Parquet output at the last step
    end
    return (run_id = Int(p.run_id), elapsed = time() - t0, maxrss_gb = Sys.maxrss() / 1024^3)
end

# ─── Main

function main()
    runs = CSV.read(A["run_list"], DataFrame)
    if !isempty(strip(A["sample_ids"]))
        id_col = "saltelli_sample_id" in names(runs) ? :saltelli_sample_id : :sample_id
        ids = Set(parse.(Int, split(A["sample_ids"], ",")))
        runs = runs[in.(runs[!, id_col], Ref(ids)), :]
        label = "check"
    else
        n, c, nc = nrow(runs), A["chunk"], A["num_chunks"]
        size_, extra = divrem(n, nc)
        first_ = c * size_ + min(c, extra) + 1
        runs = runs[first_:first_ + size_ - 1 + (c < extra ? 1 : 0), :]
        label = "chunk$(c)"
    end
    mkpath(A["raw_dir"])

    prep = joinpath(ROOT, "preparation")
    maps = (; m_fgrote  = read_map(joinpath(prep, "flood_maps"), "grote.png"),
              m_fmiddel = read_map(joinpath(prep, "flood_maps"), "middel.png"),
              m_fklein  = read_map(joinpath(prep, "flood_maps"), "kleine.png"),
              m_feklein = read_map(joinpath(prep, "flood_maps"), "extreme_kleine.png"),
              nvm       = read_map(joinpath(prep, "flood_maps"), "nvm_cleaned.png"))
    gev_df = CSV.read(joinpath(prep, "gev", "gev_config_static_2500.csv"), DataFrame)
    gev = Dict(s => Float64.(gev_df[!, Symbol("set$s")]) for s in 0:4)

    t0 = time()
    println("$(nrow(runs)) runs from $(basename(A["run_list"])) ($label) on $(nworkers()) workers")
    log = pmap(p -> run_one(p, maps, gev[Int(p.setting)], FIXED, A["raw_dir"], TOT_STEP),
               [NamedTuple(r) for r in eachrow(runs)])
    CSV.write(joinpath(A["raw_dir"], "run_log_$(label).csv"), DataFrame(log))
    @printf("Done in %.1fs\n", time() - t0)
end

main()
