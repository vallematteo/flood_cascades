"""
    model_parameters(sets...) -> Dict{Symbol,Float64}

Parameter values of model/parameters.csv in the given sets, later sets overriding earlier
ones. The calibrated parametrisation is model_parameters("fixed", "baseline", "calibrated").
"""
function model_parameters(sets::AbstractString...)
    table = CSV.read(joinpath(@__DIR__, "parameters.csv"), DataFrame; types = Dict(:value => Float64))
    values = Dict{Symbol,Float64}()
    for set in sets, row in eachrow(table)
        row.set == set && (values[Symbol(row.parameter)] = row.value)
    end
    return values
end
