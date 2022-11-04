using UnzipLoops
using Test
using OffsetArrays

@testset "UnzipLoops" begin
    X, Y = rand(1:5, 1024), rand(1:5, 1024)
    f(x, y) = x + y, x - y
    function g(X, Y)
        @assert size(X) == size(Y)
        Base.require_one_based_indexing(X, Y)

        T = promote_type(eltype(X), eltype(Y))
        N = ndims(X)
        Z1 = Array{T,N}(undef, size(X))
        Z2 = Array{T,N}(undef, size(X))
        @inbounds @simd for i in eachindex(X)
            v = f(X[i], Y[i])
            Z1[i] = v[1]
            Z2[i] = v[2]
        end
        return Z1, Z2
    end
    Z1, Z2 = g(X, Y)
    Z1′, Z2′ = broadcast_unzip(f, X, Y)
    @test Z1 == Z1′
    @test Z2 == Z2′

    X, Y, Z = rand(1:5, 1024), rand(1:5, 1024), rand(1:5, 1024)
    f(x, y, z) = x ^ y ^ z, x / y / z, x * y * z, x / (y*z)
    out = broadcast_unzip(f, X, Y, Z)
    out_aos = map(f, X, Y, Z)
    @test out[1] == getindex.(out_aos, 1)
    @test out[2] == getindex.(out_aos, 2)
    @test out[3] == getindex.(out_aos, 3)
    @test out[4] == getindex.(out_aos, 4)

    f(x) = x, x^2, x^3
    A, B, C = broadcast_unzip(f, [1, 2, 3])
    @test collect(zip(A, B, C)) == broadcast(f, [1, 2, 3])

    # mixed axes are supported
    f(x, y) = x + y, x - y
    A, B = broadcast_unzip(f, [1, 2, 3], [4 5 6])
    @test collect(zip(A, B)) == broadcast(f, [1, 2, 3], [4 5 6])

    # offsetted arrays
    xo = OffsetArray([1, 2, 3], -1)
    A, B = broadcast_unzip(f, xo, xo)
    @test axes(A) == (0:2,)
    @test axes(B) == (0:2,)
    @test collect(zip(A, B)) == broadcast(f, xo, xo)
end
