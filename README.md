# LegolasFlux

[![CI](https://github.com/beacon-biosignals/LegolasFlux.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/beacon-biosignals/LegolasFlux.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/beacon-biosignals/LegolasFlux.jl/branch/main/graph/badge.svg?token=NHYUL22HCC)](https://codecov.io/gh/beacon-biosignals/LegolasFlux.jl)

LegolasFlux provides some simple functionality to use [Legolas.jl](https://github.com/beacon-biosignals/Legolas.jl/)'s
extensible Arrow schemas as means to serialize Flux models similarly to using Flux's `params` and `loadparams!`
(instead, we export similar functions `fetch_weights` and `load_weights!` which handle layers like `BatchNorm` correctly for this purpose).

The aim is to serialize only the numeric weights, *not* the code defining the model. This is a very different approach
from e.g. BSON.jl, and hopefully much more robust. Note that in this package, we use `weights` to refer to the numeric arrays that are modified over the course of training a model; that includes biases as well as means and variances in e.g. BatchNorms (but not e.g. configuration settings).

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

model_row = LegolasFlux.ModelV1(; weights = fetch_weights(cpu(my_model)),
                                architecture_version=1)
write_model_row("my_model.model.arrow", model_row)

# Great! Later on, we want to re-load our model weights.
fresh_model = make_my_model()

model_row = read_model_row("my_model.model.arrow")
load_weights!(fresh_model, model_row.weights)
# Now our weights have been loaded back into `fresh_model`.

```

We can make use of the `architecture_version` column to specify a version number for the architectures, in order
to keep track of for which architectures the weights are valid for.

See [examples/digits.jl](examples/digits.jl) for a larger example, which also saves out extra metadata with the model,
by using a Legolas schema extension.

## `LegolasFlux.ModelV1`

A `LegolasFlux.ModelV1` is a central object of LegolasFlux. It acts as a Tables.jl-compatible row that can store the weights
of a Flux model in the `weights` column, optionally an `architecture_version` (defaults to `missing`).

`ModelV1` is not exported because downstream models likely want to define their own rows which extend the schema provided by LegolasFlux
that might end up being called something similar. See the next section for more on extensibility.

## Extensibility

As a Legolas.jl schema, it is meant to be extended. For example, let's say I had an MNIST classification model
that I call `Digits`. I am very committed to reproducibility, so I store the `commit_sha` of my model's repo
with every training run, and I also wish to save the accuracy and epoch. I might create a `DigitsRow` which is
a schema extension of the `legolas-flux.model` schema:

```julia
using Legolas, LegolasFlux
using Legolas: @schema, @version
@schema "digits-model" DigitsRow
@version DigitsRowV1 begin
    epoch::Union{Missing, Int}
    accuracy::Union{Missing, Float32}
    commit_sha::Union{Missing, String}
end
```

Now I can use a `DigitsRowV1` much like LegolasFlux's `ModelV1`. It has the same required `weights` column and optional `architecture_version` column, as well as the additional `epoch`, `accuracy`, and `commit_sha` columns. As a naming convention,
one might name files produced by this row as e.g. `training_run.digits.model.arrow`.

Note in this example the schema is called `digits.model` instead of just say `digits`, since the package Digits might want to
create other Legolas schemas as well at some point.

Check out the [Legolas.jl](https://github.com/beacon-biosignals/Legolas.jl/) repo to see more about how its extensible schema system works,
and the example at [examples/digits.jl](examples/digits.jl).
