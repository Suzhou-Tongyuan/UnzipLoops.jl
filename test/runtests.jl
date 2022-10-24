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

    @test_throws DimensionMismatch broadcast_unzip(+, [1, 2, 3], [4 5 6])

    # It's not supported yet, but we'd like to support generic array in the future
    x = OffsetArray([1, 2, 3], -1)
    msg = "offset arrays are not supported but got an array with index other than 1"
    @test_throws ArgumentError(msg) broadcast_unzip(+, x, x)
    @test_broken broadcast_unzip(+, x, x)
end
