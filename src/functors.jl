"""
    fetch_weights(m) -> Vector{Array}

Returns the weights of a model by using `Functors.children` to recurse
through the model, keeping any numeric arrays found. The `@functor` macro defines
`Functors.children` automatically so that should be sufficient to support
custom types.

Note that this function does not copy the results, so that e.g. mutating
`fetch_weights(m)[1]` modifies the model.
"""
fetch_weights(m) = filter(is_numeric_array, fcollect(m))
is_numeric_array(x) = false
is_numeric_array(x::Array{<:Number}) = true
is_numeric_array(x::Array) = all(x -> x isa Number || is_numeric_array(x), x)

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
