module LegolasFlux

export write_model_row, read_model_row
export fetch_weights, load_weights!

using Legolas
using Legolas: @schema, @version
using Arrow
using Arrow.ArrowTypes
using Tables
using Functors
using Base: IdSet

#####
##### `FlatArray`
#####

struct FlatArray{T <: Number}
    vec::Vector{T}
    size::Vector{Int}
end

FlatArray{T}(x::AbstractArray{T}) where {T} = FlatArray{T}(vec(x), collect(size(x)))
FlatArray{T}(x::AbstractArray{S}) where {T, S} = FlatArray{T}(convert(Vector{T}, vec(x)), collect(size(x)))
FlatArray(x::AbstractArray{T}) where {T} = FlatArray{T}(vec(x), collect(size(x)))
Base.Array(x::FlatArray) = reshape(x.vec, x.size...)

for op in (:(==), :isequal)
    @eval Base.$(op)(f1::FlatArray, f2::FlatArray) = $op(f1.size, f2.size) && $op(f1.vec, f2.vec)
end
Base.hash(f::FlatArray, h::UInt) = hash(:FlatArray, hash(f.vec, hash(f.size, h)))

const FLATARRAY_ARROW_NAME = Symbol("JuliaLang.LegolasFlux.FlatArray")
ArrowTypes.arrowname(::Type{<:FlatArray}) = FLATARRAY_ARROW_NAME
ArrowTypes.JuliaType(::Val{FLATARRAY_ARROW_NAME}) = FlatArray
ArrowTypes.fromarrow(::Type{<:FlatArray}, vec, size) = FlatArray{eltype(vec)}(vec, size)

#####
##### `Weights`
#####

struct Weights{T}
    weights::Vector{FlatArray{T}}
end
Base.hash(w::Weights, h::UInt) = hash(:Weights, hash(w.weights, h))

for op in (:(==), :isequal)
    @eval Base.$(op)(w1::Weights, w2::Weights) = $op(w1.weights, w2.weights)
end

Weights(w::Weights) = w
Base.length(w::Weights) = length(w.weights)
Base.eltype(w::Weights{T}) where T = Array{T}

function Base.iterate(w::Weights, state...)
    result = iterate(w.weights, state...)
    result === nothing && return nothing
    arr, state = result
    return Array(arr), state
end

# for saving weights from the user
function Weights(v::AbstractVector)
    T = foldl(promote_type, (eltype(x) for x in v))
    return Weights(FlatArray{T}.(v))
end

# for deserializing from Arrow
# here we assume the elements are all of the same type
# but the `eltype`s may not be correct (Arrow seems to
# deserialize them as `FlatArray{Any}`).
function Weights(v::AbstractVector{<:FlatArray})
    T = typeof(first(first(v).vec))
    return Weights(convert.(FlatArray{T}, v))
end

const WEIGHTS_ARROW_NAME = Symbol("JuliaLang.LegolasFlux.Weights")
ArrowTypes.arrowname(::Type{<:Weights}) = WEIGHTS_ARROW_NAME
ArrowTypes.JuliaType(::Val{WEIGHTS_ARROW_NAME}) = Weights

#####
##### `ModelV1`
#####

@schema "legolas-flux.model" Model

@version ModelV1 begin
    # expose the concrete type of `weights` as a type parameter of `ModelV1`
    # see https://github.com/beacon-biosignals/Legolas.jl/blob/main/examples/tour.jl#L291
    weights::(<:Union{Missing,Weights}) = ismissing(weights) ? missing : Weights(weights)
    architecture_version::Union{Missing,Int}
end

#####
##### Utilities
#####

"""
    write_model_row(io_or_path, row[, schema=ModelV1SchemaVersion()]; kwargs...)

A light wrapper around `Legolas.write` to write a table
with a single row. `kwargs` are forwarded to an internal
invocation of `Arrow.write`.
"""
function write_model_row(io_or_path, row,
                         schema::Legolas.SchemaVersion=ModelV1SchemaVersion();
                         kwargs...)
    return Legolas.write(io_or_path, [row], schema; validate=true, kwargs...)
end

"""
    read_model_row(io_or_path) -> Legolas.AbstractRecord

A light wrapper around `Legolas.read` to retrieve
an `AbstractRecord` from a table with a single row, such
as the output of [`write_model_row`](@ref)`.  The schema version
is inferred from the `Arrow.Table` (as with `Legolas.read`).
"""
function read_model_row(io_or_path)
    table = Legolas.read(io_or_path; validate=true)
    # because we used validate=true above, we know that we can extract a usable
    # SchemaVersion from the table.
    sv = Legolas.extract_schema_version(table)
    RecordType = Legolas.record_type(sv)
    row = only(Tables.rows(table))
    return RecordType(row)
end

include("functors.jl")

end # module
