# LegolasFlux

LegolasFlux provides some simple functionality to use [Legolas.jl](https://github.com/beacon-biosignals/Legolas.jl/)'s
extensible Arrow schemas as means to serialize Flux models using Flux's `params` and `loadparams!`.

The aim is to serialize only the numeric weights, *not* the code defining the model. This is a very different approach
from e.g. BSON.jl, and hopefully much more robust.

With this approach, however, if you change the code such that the weights are no longer valid (e.g. add a layer),
you will not be able to load back the same model.

The code currently only supports `Float32` parameters.

## Usage

```julia
using Flux

function make_my_model()
    return Chain(Dense(1,10), Dense(10, 10), Dense(10, 1))
end

my_model = make_my_model()
# train it? that part is optional ;)

# Now, let's save it!
using LegolasFlux

model_row = ModelRow(; weights = collect(params(cpu(my_model))))
write_model_row("my_model.arrow", model_row)

# Great! Later on, we want to re-load our model weights.
fresh_model = make_my_model()

model_row = read_model_row("my_model.arrow")
Flux.loadparams!(fresh_model, Array.(model_row.weights))
# Now our params have been loaded back into `fresh_model`.
```

We can also make use of the `architecture_version` column to specify a version number for the architectures, in order
to keep track of for which architectures the weights are valid for.
