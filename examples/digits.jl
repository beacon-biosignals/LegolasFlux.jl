# Model modified from
# https://discourse.julialang.org/t/how-to-drop-the-dropout-layers-in-flux-jl-when-assessing-model-performance/19924

using Flux, Statistics, Random, Test
# Uncomment to use MNIST data
# using MLDatasets: MNIST
using StableRNGs
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Legolas, LegolasFlux
using Legolas: @schema, @version
using LegolasFlux: Weights
using Tables

# This should store all the information needed
# to construct the model.
@schema "digits-config" DigitsConfig
@version DigitsConfigV1 begin
    seed::Int = coalesce(seed, 5)
    dropout_rate::Float32 = coalesce(dropout_rate, 0.0f1)
end

# Here's our model object itself, just a `DigitsConfig` and
# a `chain`. We keep the config around so it's easy to save out
# later.
struct DigitsModel
    chain::Chain
    config::DigitsConfigV1
end

# Ensure Flux can recurse into our model to find params etc
Flux.@functor DigitsModel (chain,)

# Construct the actual model from a config object. This is the only
# constructor that should be used, to ensure the model is created just
# from the config object alone.
function DigitsModel(config::DigitsConfigV1=DigitsConfigV1())
    dropout_rate = config.dropout_rate
    Random.seed!(config.seed)
    D = Dense(10, 10)
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
                  D,
                  D, # same layer twice, to test weight-sharing
                  softmax)
    return DigitsModel(chain, config)
end

# Our model acts on input just by applying the chain.
(m::DigitsModel)(x) = m.chain(x)

# Here, we define a schema extension of the `legolas-flux.model` schema.
# We add our `DigitsConfig` object, as well as the epoch and accuracy.
@schema "digits.model" DigitsRow

# construct a `DigitsConfigV1` with support for older Legolas serialization formats
# to support backwards compatibility
compat_config(config::DigitsConfigV1) = config
function compat_config(config::NamedTuple)
    # This is how these deserialize from Legolas v0.4
    if haskey(config, 1) && config[1] == "digits-config" && haskey(config, 2) &&
       config[2] == 1
        return DigitsConfigV1(config[3])
    else
        # We have some other NamedTuple, possibly from an earlier Legolas version. Trying constructing `DigitsConfigV1`.
        return DigitsConfigV1(config)
    end
end

@version DigitsRowV1 > LegolasFlux.ModelV1 begin
    # parametrize the row type on the weights type (copied from ModelV1)
    weights::(<:Union{Missing,Weights}) = ismissing(weights) ? missing : Weights(weights)
    config::Union{<:NamedTuple,DigitsConfigV1} = compat_config(config)
    epoch::Union{Missing,Int}
    accuracy::Union{Missing,Float32}
end

# Construct a `DigitsRowV1` from a model by collecting the weights.
# This can then be saved with e.g. `LegolasFlux.write_model_row`.
function DigitsRowV1(model::DigitsModel; epoch=missing, accuracy=missing)
    return DigitsRowV1(; weights=fetch_weights(model), model.config, epoch, accuracy)
end

# Construct a `DigitsModel` from a row satisfying the `DigitsRowV1` schema,
# i.e. one with a `weights` and `config::DigitsConfigV1`.
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

function train_model!(m; N=N_train)
    loss = (x, y) -> crossentropy(m(x), y)
    opt = ADAM()
    evalcb = throttle(() -> @show(accuracy(m, tX, tY)), 5)
    Flux.@epochs 1 Flux.train!(loss, Flux.params(m), Iterators.take(train, N), opt;
                               cb=evalcb)
    return accuracy(m, tX, tY)
end

m = DigitsModel()
@test m.chain[end-2] === m.chain[end-1] # test weight sharing

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

path = joinpath(pkgdir(LegolasFlux), "examples", "test.digits-model.arrow")
# The saved weights in this repo were generated by running the command:
# write_model_row(path, row, DigitsRowV1SchemaVersion())
# We don't run this every time, since we want to test that we can continue to deserialize previously saved out weights.
roundtripped = read_model_row(path)
@test roundtripped isa DigitsRowV1
@test roundtripped.config isa DigitsConfigV1

roundtripped_model = DigitsModel(roundtripped)
output3 = roundtripped_model(input)
@test output3 isa Matrix{Float32}

# Test we are still weight-sharing post-roundtrip
@test roundtripped_model.chain[end-2] === roundtripped_model.chain[end-1]

# Here, we've hardcoded the results at the time of serialization.
# This lets us check that the model we've saved gives the same answers now as it did then.
# It is OK to update this test w/ a new reference if the answers are *supposed* to change for some reason. Just make sure that is the case.
@test output3 ≈
      Float32[0.096030906; 0.105671346; 0.09510324; 0.117868274; 0.112540945; 0.08980863; 0.062402092; 0.09776583; 0.11317684; 0.109631866;;]
