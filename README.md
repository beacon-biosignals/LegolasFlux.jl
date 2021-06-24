# LegolasFlux

LegolasFlux provides some simple functionality to use [Legolas.jl](https://github.com/beacon-biosignals/Legolas.jl/)'s
extensible Arrow schemas as means to serialize Flux models using Flux's `params` and `loadparams!`.

The aim is to serialize only the numeric weights, *not* the code defining the model. This is a very different approach
from e.g. BSON.jl, and hopefully much more robust.

With this approach, however, if you change the code such that the weights are no longer valid (e.g. add a layer),
you will not be able to load back the same model.

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

# We can save whatever other columns we'd like to as well as the `weights`.
model_row = ModelRow(; weights = collect(params(cpu(my_model))), architecture_version = 1, loss = 0.5)
write_model_row("my_model.arrow", model_row)

# Great! Later on, we want to re-load our model weights.
fresh_model = make_my_model()

model_row = read_model_row("my_model.arrow")
Flux.loadparams!(fresh_model, collect(model_row.weights))
# Now our params have been loaded back into `fresh_model`.
# Note we needed to `collect` the weights before we use them.

# We can also check out our other columns:
model_row.loss # 0.5

```

We can make use of the `architecture_version` column to specify a version number for the architectures, in order
to keep track of for which architectures the weights are valid for.

## `ModelRow`

A `ModelRow` is the central export of LegolasFlux. It acts as a Tables.jl-compatible row that can store the weights
of a Flux model in the `weights` column, optionally an `architecture_version` (defaults to `missing`), and any
other columns the user desires.

## Extensibility

As a Legolas.jl schema, it is meant to be extended. For example, let's say I had an MNIST classification model
that I call `Digits`. I am very committed to reproducibility, so I store the `commit_sha` of my model's repo
with every training run, and I also wish to save the accuracy and epoch. I might create a `DigitsRow` which is
a schema extension of the `legolas-flux` schema:

```julia
using Legolas, LegolasFlux

const DigitsRow = Legolas.@row("digits@1" > "legolas-flux@1",
                         epoch::Union{Missing, Int},
                         accuracy::Union{Missing, Float32},
                         commit_sha::Union{Missing, String})
```

Now I can use a `DigitsRow` much like LegolasFlux's `ModelRow`. It has the same required `weights` column and optional `architecture_version` column, as well as the additional `epoch`, `accuracy`, and `commit_sha` columns.

Check out the [Legolas.jl](https://github.com/beacon-biosignals/Legolas.jl/) repo to see more about how its extensible schema system works.
