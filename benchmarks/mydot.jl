using MappedArrays, StructArrays
using UnzipLoops
using LinearAlgebra
using BenchmarkTools

# Provided by @mcabbott and @jishnub
"""
    @unzip f.(g.(x))
Like `@unzipcast` but without `@.`, thus expects a broadcasting expression.
"""
macro unzip(ex)
  bc = esc(ex)
  :(_unz.($bc))
end

function _unz end  # this is never called
Broadcast.broadcasted(::typeof(_unz), x) = _Unz(x)
struct _Unz{T}; bc::T; end
Broadcast.materialize(x::_Unz) = StructArrays.components(StructArray(Broadcast.instantiate(x.bc)))


f(x, y) = (x^y, x/y)
# Baseline
@inline function mydot_v1(f, X, Y)
    tmp = broadcast(f, X, Y)
    return dot(map(first, tmp), map(last, tmp))
end

# UnzipLoops
@inline function mydot_v2(f, X, Y)
    A, B = broadcast_unzip(f, X, Y)
    return dot(A, B)
end

# StructArrays
@inline function mydot_v3(f, X, Y)
    A, B = @unzip f.(X, Y)
    return dot(A, B)
end

# vector case
X, Y = rand(1024), rand(1024);
mydot_v1(f, X, Y) ≈ mydot_v2(f, X, Y) ≈ mydot_v3(f, X, Y)
# essential computation time
@btime f.($X, $Y); # 32.636 μs (1 allocation: 16.12 KiB)
@btime dot($X, $Y); # 51.022 ns (0 allocations: 0 bytes)
# unzip overhead: 6μs vs 0μs vs 0μs
@btime mydot_v1(f, $X, $Y); # 38.929 μs (3 allocations: 32.38 KiB)
@btime mydot_v2(f, $X, $Y); # 32.067 μs (2 allocations: 16.25 KiB)
@btime mydot_v3(f, $X, $Y); # 32.311 μs (2 allocations: 16.25 KiB)

# matrix case
X, Y = rand(1024), collect(rand(1024)');
mydot_v1(f, X, Y) ≈ mydot_v2(f, X, Y) ≈ mydot_v3(f, X, Y)
# essential computation time
@btime f.($X, $Y); # 32.766 ms (2 allocations: 16.00 MiB)
@btime dot($X, $Y); # 51.138 ns (0 allocations: 0 bytes)
# unzip overhead: 11ms vs 7ms vs 13ms
@btime mydot_v1(f, $X, $Y); # 43.507 ms (6 allocations: 32.00 MiB)
@btime mydot_v2(f, $X, $Y); # 39.685 ms (4 allocations: 16.00 MiB)
@btime mydot_v3(f, $X, $Y); # 45.969 ms (4 allocations: 16.00 MiB)
