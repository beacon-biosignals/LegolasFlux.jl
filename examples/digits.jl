# Model modified from
# https://discourse.julialang.org/t/how-to-drop-the-dropout-layers-in-flux-jl-when-assessing-model-performance/19924

using Flux, Statistics, Random, Test
# Uncomment to use MNIST data
# using MLDatasets: MNIST
using StableRNGs
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Legolas, LegolasFlux

# This should store all the information needed
# to construct the model.
Base.@kwdef struct DigitsConfig
    seed::Int = 5
    dropout_rate::Float32 = 0f1
end

# Here's our model object itself, just a `DigitsConfig` and
# a `chain`. We keep the config around so it's easy to save out
# later.
struct DigitsModel
    chain::Chain
    config::DigitsConfig
end

# Ensure Flux can recurse into our model to find params etc
Flux.@functor DigitsModel (chain,)

# Construct the actual model from a config object. This is the only
# constructor that should be used, to ensure the model is created just
# from the config object alone.
function DigitsModel(config::DigitsConfig=DigitsConfig())
    dropout_rate = config.dropout_rate
    Random.seed!(config.seed)
    chain = Chain(Dropout(dropout_rate),
                  Conv((3, 3), 1 => 32, relu),
                  BatchNorm(32, relu),
                  MaxPool((2, 2)),
                  Dropout(dropout_rate),
                  Conv((3, 3), 32 => 16, relu),
                  Dropout(dropout_rate),
                  MaxPool((2, 2)),
                  Dropout(dropout_rate),
                  Conv((3, 3), 16 => 10, relu),
                  Dropout(dropout_rate),
                  x -> reshape(x, :, size(x, 4)),
                  Dropout(dropout_rate),
                  Dense(90, 10),
                  softmax)
    return DigitsModel(chain, config)
end

# Our model acts on input just by applying the chain.
(m::DigitsModel)(x) = m.chain(x)

# Here, we define a schema extension of the `legolas-flux.model` schema.
# We add our `DigitsConfig` object, as well as the epoch and accuracy.
@schema "digits.model" DigitsRow

@version DigitsRowV1 > LegolasFlux.ModelV1 begin
    config::DigitsConfig
    epoch::Union{Missing, Int}
    accuracy::Union{Missing, Float32}
end

# Construct a `DigitsRowV1` from a model by collecting the weights.
# This can then be saved with e.g. `LegolasFlux.write_model_row`.
function DigitsRowV1(model::DigitsModel; epoch=missing, accuracy=missing)
    return DigitsRowV1(; weights=fetch_weights(model), model.config, epoch, accuracy)
end

# Construct a `DigitsModel` from a row satisfying the `DigitsRowV1` schema,
# i.e. one with a `weights` and `config::DigitsConfig`.
# This could be the result of `LegolasFlux.read_model_row`.
function DigitsModel(row)
    m = DigitsModel(row.config)
    load_weights!(m, row.weights)
    return m
end


# Increase to get more training/test data
N_train = 1_000
N_test = 50

##
# to use MNIST data, uncomment these
# train_x, train_y = MNIST.traindata(Float32, 1:N_train)
# test_x,  test_y  = MNIST.testdata(Float32, 1:N_test)

# Random data:
rng = StableRNG(735)
train_x = rand(rng, Float32, 28, 28, N_train)
train_y = rand(rng, 0:9, N_train)
test_x = rand(rng, Float32, 28, 28, N_test)
test_y = rand(rng, 0:9, N_test)
##

# Partition into batches of size 32
batch_size = 32
train = [(reshape(train_x[:, :, I], 28, 28, 1, :), onehotbatch(train_y[I], 0:9))
         for I in partition(1:N_train, batch_size)]

tX = reshape(test_x, 28, 28, 1, :)
tY = onehotbatch(test_y, 0:9)

function accuracy(m, x, y)
    testmode!(m)
    val = mean(onecold(m(x)) .== onecold(y))
    trainmode!(m)
    return val
end

function train_model!(m; N = N_train)
    loss = (x, y) -> crossentropy(m(x), y)
    opt = ADAM()
    evalcb = throttle(() -> @show(accuracy(m, tX, tY)), 5)
    Flux.@epochs 1 Flux.train!(loss, Flux.params(m), Iterators.take(train, N), opt; cb=evalcb)
    return accuracy(m, tX, tY)
end

m = DigitsModel()

# increase N to actually train more than a tiny amount
acc = train_model!(m; N=10)

# Let's serialize out the weights into a `DigitsRowV1`.
# We could save this here with `write_model_row`.
row = DigitsRowV1(m; epoch=1, accuracy=acc)

testmode!(m)
input = tX[:, :, :, 1:1]
output = m(input)
label = tY[:, 1]

# Let's now reconstruct the model from the `row` and check that we get
# the same outputs.
m2 = DigitsModel(row)
testmode!(m2)
output2 = m2(input)

@test output ≈ output2
