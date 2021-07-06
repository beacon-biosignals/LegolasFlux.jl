# Model modified from
# https://discourse.julialang.org/t/how-to-drop-the-dropout-layers-in-flux-jl-when-assessing-model-performance/19924

using Flux, Statistics, Random, Test
# Uncomment to use MNIST data
# using MLDatasets: MNIST
using StableRNGs
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Legolas, LegolasFlux

Base.@kwdef struct DigitsConfig
    seed::Int = 5
    dropout_rate::Float32 = 0f1
end

struct DigitsModel
    chain::Chain
    config::DigitsConfig
end

Flux.@functor DigitsModel (chain,)

function DigitsModel(config::DigitsConfig = DigitsConfig())
    dropout_rate = config.dropout_rate
    Random.seed!(config.seed)
    chain = Chain(
        Dropout(dropout_rate),
        Conv((3, 3), 1=>32, relu),
        BatchNorm(32, relu),
        MaxPool((2,2)),
        Dropout(dropout_rate),
        Conv((3, 3), 32=>16, relu),
        Dropout(dropout_rate),
        MaxPool((2,2)),
        Dropout(dropout_rate),
        Conv((3, 3), 16=>10, relu),
        Dropout(dropout_rate),
        x -> reshape(x, :, size(x, 4)),
        Dropout(dropout_rate),
        Dense(90, 10), softmax)
    return DigitsModel(chain, config)
end

(m::DigitsModel)(x) = m.chain(x)

const DigitsRow = Legolas.@row("digits.model@1" > "legolas-flux.model@1",
                               config::DigitsConfig,
                               epoch::Union{Missing, Int},
                               accuracy::Union{Missing, Float32})

function DigitsRow(model::DigitsModel; epoch=missing, accuracy=missing)
    w = collect(weights(model))
    return DigitsRow(; weights=w, model.config, epoch, accuracy)
end

function DigitsModel(row)
    m = DigitsModel(row.config)
    loadweights!(m, collect(row.weights))
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
    Flux.@epochs 1 Flux.train!(loss, params(m), Iterators.take(train, N), opt, cb = evalcb)
    return accuracy(m, tX, tY)
end

m = DigitsModel()

# increase N to actually train more than a tiny amount
acc = train_model!(m; N = 10)

row = DigitsRow(m; epoch=1, accuracy=acc)

testmode!(m)
input = tX[:, :, :, 1:1]
output = m(input)
label = tY[:, 1]

m2 = DigitsModel(row)
testmode!(m2)
output2 = m2(input)

@test output ≈ output2
