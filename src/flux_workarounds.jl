# Modified version of `fcollect` to use an `IdSet` cache so that
# distinct arrays whose values happen to be duplicates are each kept.
function fcollect2(x; output = [], cache = Base.IdSet(), exclude = v -> false)
    x in cache && return output
    if !exclude(x)
      push!(cache, x)
      push!(output, x)
      foreach(y -> fcollect2(y; cache = cache, output=output, exclude = exclude), Functors.children(x))
    end
    return output
end

weights(m) = filter(x -> x isa Array, fcollect2(m))

function loadweights!(m, xs)
  for (i, (p, x)) in enumerate(zip(weights(m), xs))
    size(p) == size(x) ||
      error("Expected param size $(size(p)), got $(size(x)) for the $(i)th weight")
    copyto!(p, x)
  end
end
