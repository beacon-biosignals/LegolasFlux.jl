module LegolasFlux

export ModelRow, write_model_row, read_model_row

using Legolas
using Arrow
using Arrow.ArrowTypes
using Tables

const LEGOLAS_SCHEMA = Legolas.Schema("legolas-flux@1")

struct FlatArray
    vec::Vector{Float32}
    size::Vector{Int}
end

FlatArray(x::FlatArray) = x
FlatArray(x::AbstractArray) = FlatArray(vec(x), collect(size(x)))
Base.Array(x::FlatArray) = reshape(x.vec, x.size...)

const FLATARRAY_ARROW_NAME = Symbol("JuliaLang.LegolasFlux.FlatArray")
ArrowTypes.arrowname(::Type{<:FlatArray}) = FLATARRAY_ARROW_NAME
ArrowTypes.JuliaType(::Val{FLATARRAY_ARROW_NAME}) = FlatArray

const ModelRow = Legolas.@row("legolas-flux@1",
     weights::Vector{FlatArray} = FlatArray.(weights),
     architecture_version::Union{Missing, Int})


"""
    write_model_row(io_or_path; kwargs...)

A light wrapper around `Legolas.write` to write a table
with a single row.
"""
function write_model_row(io_or_path, row; kwargs...)    
     return Legolas.write(io_or_path, [row], LEGOLAS_SCHEMA; validate=true, kwargs...)
end

"""
    read_model_row(io_or_path; kwargs...) -> ModelRow

A light wrapper around `Legolas.read` to load a table
with a single row.
"""
function read_model_row(io_or_path; kwargs...)    
    table = Legolas.read(io_or_path; validate=true, kwargs...)
    Legolas.validate(table, LEGOLAS_SCHEMA)
    rows = ModelRow.(Tables.rows(table))
    return only(rows)
end


end # module
