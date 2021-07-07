using LegolasFlux
using Test
using Flux, LegolasFlux
using LegolasFlux: Weights, FlatArray, ModelRow
using Arrow
using Random

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

@testset "`flux_workarounds`" begin
    @testset "layer $layer" for layer in [BatchNorm, InstanceNorm, (c) -> GroupNorm(c, 1), c -> identity]
        mk_model = () -> (Random.seed!(1); Chain(Dense(1, 10), Dense(10, 10), layer(1), Dense(10, 1)))
        model = mk_model()
        trainmode!(model)
        x = reshape([1f0], 1, 1, 1)
        for i in 1:10
            x = model(x)
        end
        testmode!(model)
        w = weights(model)
        p = collect(params(model))
        output = model(x)

        r1 = mk_model()
        loadweights!(r1, w)
        testmode!(r1)

        @test output ≈ r1(x)

        if layer == BatchNorm
            r2 = mk_model()
            Flux.loadparams!(r2, p)
            testmode!(r2)

            # If this test *fails*, meaning `output ≈ r2(x)`,
            # then perhaps we should revisit `loadweights!`
            # and could consider switching to `Flux.loadparams`.
            # See https://github.com/beacon-biosignals/LegolasFlux.jl/pull/4
            # for more.
            @test_broken output ≈ r2(x)
        end
    end
end

@testset "Example" begin
    include("../examples/digits.jl")
end
