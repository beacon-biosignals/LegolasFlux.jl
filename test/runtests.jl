using LegolasFlux
using Test
using Flux, LegolasFlux
using LegolasFlux: Weights, FlatArray
using Arrow

function make_my_model()
    return Chain(Dense(1, 10), Dense(10, 10), Dense(10, 1))
end

function test_weights()
    shapes = [(10, 1), (10,), (10, 10), (10,), (1, 10), (1,)]
    return [reshape(Float32.(1:prod(s)), s) for s in shapes]
end

@testset begin
    my_model = make_my_model()
    Flux.loadparams!(my_model, test_weights())

    model_row = ModelRow(; weights=collect(params(my_model)))
    write_model_row("my_model.arrow", model_row)

    fresh_model = make_my_model()

    model_row = read_model_row("my_model.arrow")
    weights = collect(model_row.weights)
    Flux.loadparams!(fresh_model, weights)

    @test collect(params(fresh_model)) == weights == test_weights()

    @test all(x -> eltype(x) == Float32, weights)

    rm("my_model.arrow")
end

@testset "`Weights`" begin
    v = [rand(Int8, 5), rand(Float32, 5, 5)]
    @test Weights(v) isa Weights{Float32}
    @test Weights(FlatArray{Float32}.(v)) isa Weights{Float32}
    @test Weights(FlatArray{Float64}.(v)) isa Weights{Float64}

    w = Weights(v)
    tbl = [(; weights = w)]
    Arrow.Table(Arrow.tobuffer(tbl)).weights[1] == w
end
