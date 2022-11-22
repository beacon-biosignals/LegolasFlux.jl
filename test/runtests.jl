using LegolasFlux
using Test
using Flux, LegolasFlux
using LegolasFlux: Weights, FlatArray, ModelV1
using Flux: params
using Arrow
using Random
using StableRNGs
using Legolas: @version, @schema
function make_my_model()
    return Chain(Dense(1, 10), Dense(10, 10), Dense(10, 1))
end

function test_weights()
    shapes = [(10, 1), (10,), (10, 10), (10,), (1, 10), (1,)]
    return [reshape(Float32.(1:prod(s)), s) for s in shapes]
end

@testset "Roundtripping simple model" begin
    try
        # quick test with `missing` weights.
        model_row = ModelV1(; weights=missing)
        write_model_row("my_model.model.arrow", model_row)
        rt = read_model_row("my_model.model.arrow")
        @test isequal(model_row, rt)

        my_model = make_my_model()
        load_weights!(my_model, test_weights())

        model_row = ModelV1(; weights=collect(fetch_weights(my_model)))
        write_model_row("my_model.model.arrow", model_row)

        fresh_model = make_my_model()

        model_row = read_model_row("my_model.model.arrow")
        weights = collect(model_row.weights)
        load_weights!(fresh_model, weights)

        @test collect(params(fresh_model)) == weights == test_weights()

        @test all(x -> eltype(x) == Float32, weights)
    finally
        rm("my_model.model.arrow")
    end
end

struct MyArrayModel
    dense_array::Array
end
Flux.@functor MyArrayModel

@testset "Non-numeric arrays ignored" begin
    try
        m = MyArrayModel([Dense(1, 10), Dense(10, 10), Dense(10, 1)])
        weights = fetch_weights(m)
        @test length(weights) == 6

        model_row = ModelV1(; weights=collect(weights))
        write_model_row("my_model.model.arrow", model_row)

        new_model_row = read_model_row("my_model.model.arrow")
        new_weights = collect(new_model_row.weights)
        @test new_weights == weights
    finally
        rm("my_model.model.arrow")
    end
end

@testset "Errors" begin
    my_model = make_my_model()
    w = test_weights()
    w[end] = []
    @test_throws ArgumentError load_weights!(my_model, w)

    w = test_weights()
    push!(w, [])
    @test_throws ArgumentError load_weights!(my_model, w)
end

@testset "`Weights`" begin
    rng = StableRNG(245)
    v = [rand(rng, Int8, 5), rand(rng, Float32, 5, 5)]
    @test Weights(v) isa Weights{Float32}
    @test Weights(FlatArray{Float32}.(v)) isa Weights{Float32}
    @test Weights(FlatArray{Float64}.(v)) isa Weights{Float64}

    w = Weights(v)
    tbl = [(; weights=w)]
    @test Arrow.Table(Arrow.tobuffer(tbl)).weights[1] == w
end

@testset "`flux_workarounds`" begin
    @testset "layer $layer" for layer in [BatchNorm, InstanceNorm, (c) -> GroupNorm(c, 1), c -> identity]
        mk_model = () -> (Random.seed!(1); Chain(Dense(1, 10), Dense(10, 10), layer(1), Dense(10, 1)))
        model = mk_model()
        trainmode!(model)
        x = reshape([1.0f0], 1, 1, 1)
        for i in 1:10
            x = model(x)
        end
        testmode!(model)
        w = fetch_weights(model)

        p = collect(params(model))
        output = model(x)

        r1 = mk_model()
        load_weights!(r1, w)
        testmode!(r1)

        @test output ≈ r1(x)

        if layer == BatchNorm
            r2 = mk_model()
            Flux.loadparams!(r2, p)
            testmode!(r2)

            # If this test *fails*, meaning `output ≈ r2(x)`,
            # then perhaps we should revisit `load_weights!`
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
