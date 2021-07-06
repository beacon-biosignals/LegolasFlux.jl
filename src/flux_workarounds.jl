using Flux: BatchNorm, InstanceNorm, GroupNorm, Params, trainable
using Base: IdSet
export weights, loadweights!

"""
    LegolasFlux.other_weights(layer) -> Vararg{Array}

Given a layer with params that are not captured by `Flux.trainable`, produce
a tuple of arrays corresponding to these parameters (analogous to `Flux.trainable`).
"""
function other_weights end

other_weights(layer) = ()
other_weights(layer::BatchNorm) = (layer.μ, layer.σ²)
other_weights(layer::InstanceNorm) = (layer.μ, layer.σ²)
other_weights(layer::GroupNorm) = (layer.μ, layer.σ²)

#####
##### `weights`
#####

# The following is a copy of <https://github.com/FluxML/Flux.jl/blob/335286adf118b61ad6fffa5937bd9358477a00c9/src/functor.jl#L41-L63>
# with `params` changed to `weights` and the addition of the lines
# ```julia
# for child in other_weights(x)
#     weights!(p, child, seen)
# end
# ```
# to `weights!(p::Params, x, seen = IdSet())`.

weights!(p::Params, x::AbstractArray{<:Number}, seen = IdSet()) = push!(p, x)

function weights!(p::Params, x, seen = IdSet())
  x in seen && return
  push!(seen, x)
  for child in trainable(x)
    weights!(p, child, seen)
  end

  for child in other_weights(x)
    weights!(p, child, seen)
  end
end

function weights(m...)
  ps = Params()
  weights!(ps, m)
  return ps
end

function loadweights!(m, xs)
  for (p, x) in zip(weights(m), xs)
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x))")
    copyto!(p, x)
  end
end
