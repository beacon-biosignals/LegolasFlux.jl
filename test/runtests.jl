using LegolasFlux
using Test
using Flux, LegolasFlux
using LegolasFlux: Weights, FlatArray, ModelRow
using Arrow

function make_my_model()
    return Chain(Dense(1, 10), Dense(10, 10), Dense(10, 1))
end

function test_weights()
    shapes = [(10, 1), (10,), (10, 10), (10,), (1, 10), (1,)]
    return [reshape(Float32.(1:prod(s)), s) for s in shapes]
end

# This simple model should work with both Flux's `params/loadparams!` and
# our `weights/loadweights!`. The only difference is in layers with `!isempty(other_weights(layer))`.
@testset "using ($get_weights, $load_weights)" for (get_weights, load_weights) in [(weights, loadweights!, params, Flux.loadparams!)]
    my_model = make_my_model()
    Flux.loadparams!(my_model, test_weights())

    model_row = ModelRow(; weights=collect(get_weights(my_model)))
    write_model_row("my_model.model.arrow", model_row)

    fresh_model = make_my_model()

    model_row = read_model_row("my_model.model.arrow")
    weights = collect(model_row.weights)
    load_weights(fresh_model, weights)

    @test collect(params(fresh_model)) == weights == test_weights()

    @test all(x -> eltype(x) == Float32, weights)

    rm("my_model.model.arrow")
end

@testset "`Weights`" begin
    v = [rand(Int8, 5), rand(Float32, 5, 5)]
    @test Weights(v) isa Weights{Float32}
    @test Weights(FlatArray{Float32}.(v)) isa Weights{Float32}
    @test Weights(FlatArray{Float64}.(v)) isa Weights{Float64}

    w = Weights(v)
    tbl = [(; weights = w)]
    @test Arrow.Table(Arrow.tobuffer(tbl)).weights[1] == w
end

@testset "Example" begin
    include("../examples/digits.jl")
end
