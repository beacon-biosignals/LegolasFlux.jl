using LegolasFlux
using Test
using Flux, LegolasFlux

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
    Flux.loadparams!(fresh_model, Array.(model_row.weights))

    @test collect(params(fresh_model)) == test_weights()

    rm("my_model.arrow")
end
