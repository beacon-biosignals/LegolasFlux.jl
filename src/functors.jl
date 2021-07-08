# Modified version of `fcollect` to use an `IdSet` cache so that
# distinct arrays whose values happen to be duplicates are each kept.
# <https://github.com/FluxML/Functors.jl/issues/16>
function fcollect2(x; output=[], cache=IdSet(), exclude=_ -> false)
    x in cache && return output
    if !exclude(x)
        push!(cache, x)
        push!(output, x)
        foreach(y -> fcollect2(y; cache=cache, output=output, exclude=exclude), Functors.children(x))
    end
    return output
end

"""
    fetch_weights(m) -> Vector{Array}

Returns the weights of a model by using `Functors.children` to recurse
through the model, keeping any arrays found. The `@functor` macro defines
`Functors.children` automatically so that should be sufficient to support
custom types.

Note that this function does not copy the results, so that e.g. mutating `fetch_weights(m)[1]` modifies the model.
"""
fetch_weights(m) = filter(x -> x isa Array, fcollect2(m))

"""
    load_weights!(m, xs)

Load weights `xs` into the model `m`, using [`fetch_weights`](@ref).
"""
function load_weights!(m, xs)
    model_weights = fetch_weights(m)
    if length(model_weights) != length(xs)
        throw(ArgumentError("Number of weights given ($(length(xs))) does not match number of weights model expects ($(length(model_weights)))"))
    end
    for (i, (p, x)) in enumerate(zip(model_weights, xs))
        if size(p) != size(x)
            throw(ArgumentError("For the $(i)th weight expected param size $(size(p)), got $(size(x))"))
        end
        copyto!(p, x)
    end
    return nothing
end

load_weights!(m, xs::Weights) = load_weights!(m, collect(xs))
