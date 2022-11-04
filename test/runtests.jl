using UnzipLoops
using Test
using OffsetArrays

@testset "UnzipLoops" begin
    X, Y = rand(1:5, 1024), rand(1:5, 1024)
    f1(x, y) = x + y, x - y
    function g1(X, Y)
        @assert size(X) == size(Y)
        Base.require_one_based_indexing(X, Y)

        T = promote_type(eltype(X), eltype(Y))
        N = ndims(X)
        Z1 = Array{T,N}(undef, size(X))
        Z2 = Array{T,N}(undef, size(X))
        @inbounds @simd for i in eachindex(X)
            v = f1(X[i], Y[i])
            Z1[i] = v[1]
            Z2[i] = v[2]
        end
        return Z1, Z2
    end
    Z1, Z2 = g1(X, Y)
    Z1′, Z2′ = @inferred broadcast_unzip(f1, X, Y)
    @test Z1 == Z1′
    @test Z2 == Z2′

    X, Y, Z = rand(1:5, 1024), rand(1:5, 1024), rand(1:5, 1024)
    f2(x, y, z) = x ^ y ^ z, x / y / z, x * y * z, x / (y*z)
    out = @inferred broadcast_unzip(f2, X, Y, Z)
    out_aos = map(f2, X, Y, Z)
    @test out[1] == getindex.(out_aos, 1)
    @test out[2] == getindex.(out_aos, 2)
    @test out[3] == getindex.(out_aos, 3)
    @test out[4] == getindex.(out_aos, 4)

    f3(x) = x, x^2, x^3
    A, B, C = @inferred broadcast_unzip(f3, [1, 2, 3])
    @test collect(zip(A, B, C)) == broadcast(f3, [1, 2, 3])

    # mixed axes are supported
    f4(x, y) = x + y, x - y
    A, B = @inferred broadcast_unzip(f4, [1, 2, 3], [4 5 6])
    @test collect(zip(A, B)) == broadcast(f4, [1, 2, 3], [4 5 6])

    # offsetted arrays
    xo = OffsetArray([1, 2, 3], -1)
    A, B = @inferred broadcast_unzip(f4, xo, xo)
    @test axes(A) == (0:2,)
    @test axes(B) == (0:2,)
    @test collect(zip(A, B)) == broadcast(f4, xo, xo)

    # named tuple
    f5(x, y) = (; a=x+y, b=x-y)
    out = @inferred broadcast_unzip(f5, [1, 2, 3], [4 5 6])
    @test out isa NamedTuple
    @test out.a == map(x->x.a, broadcast(f5, [1, 2, 3], [4 5 6]))
    @test out.b == map(x->x.b, broadcast(f5, [1, 2, 3], [4 5 6]))
end
