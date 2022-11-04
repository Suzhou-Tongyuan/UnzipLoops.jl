# UnzipLoops

![Julia version](https://img.shields.io/badge/julia-%3E%3D%201.7-blue)

For function `f` that returns a Tuple object, this package provides one optimized function
`broadcast_unzip(f, As...)` that works similar to `broadcast(f, As...)` (aka `f.(As...)`) but
outputs different data layout.

## Example

Broadcasting is very useful in Julia. When you use broadcasting, you'll definitely meet cases like
this:

```julia
f(x, y) = x^y, x/y

X, Y = [1, 2, 3, 4], [4 3 2 1]
out = f.(X, Y)
```

This `out` is of type `Matrix{Tuple{Int,Float64}}`. It's often the case that we'll need to unzip it
to `Tuple{Matrix{Int}, Matrix{Float64}}`. For most of the case, we can trivially split it apart:

```julia
function g(X, Y)
    out = f.(X, Y)
    return map(x->x[1], out), map(x->x[2], out)
end
```

In this case, since this requires two more broadcasting, it introduces some unnecessary overhead:

- `f.(X, Y)` allocates one extra memory to store the intermediate result, and
- the entire function requires two more loops to do the work and thus hurts the cache locality --
  locality matters for low-level performance optimization.

```julia
X, Y = rand(1:5, 1024), collect(rand(1:5, 1024)')
@btime f.($X, $Y) # 4.720 ms (2 allocations: 16.00 MiB)
@btime g($X, $Y) # 7.566 ms (6 allocations: 32.00 MiB)
```

We can observe 2.8ms overhead here.

A more optimized version for this specific `f` is to pre-allocate the output result, and do
everything in one single loop. For this simple case, we can create something similar to below:

```julia
function g(X, Y)
    T = promote_type(Float64, eltype(X), eltype(Y))
    bc = broadcast(f, X, Y)
    Z1 = similar(bc, T)
    Z2 = similar(bc, T)
    @inbounds @simd for i in eachindex(bc)
        v = bc[i]
        Z1[i] = v[1]
        Z2[i] = v[2]
    end
    return Z1, Z2
end
```

We usually call this `g` the SoA (struct of array) layout. Comparing to the AoS (array of struct)
layout that has only one contiguous array involved, this SoA layout introduces two contiguous
arrays. Thus you'll still notice some overhead here compared to the plain `f.(X, Y)` kernel:

```julia
@btime g($X, $Y) # 6.878 ms (6 allocations: 32.00 MiB)
```

but the overhead is relatively small here -- we eliminate the extra loops and allocations. From case
to case, Julia compiler can optimze this difference away.

## `broadcast_unzip`

Obviously, rewriting the trivial `getindex` solution into the verbose manual loop introduces many
work and hurts the readability. This also requires the users to understand the very low-level
mechanisim of how broadcasting works, which is not always an easy thing. This is why
`broadcast_unzip` is introduced -- it's a combination of `broadcast` and `unzip`. Most importantly,
this is simple to use and yet fast:

```julia
g(X, Y) == broadcast_unzip(f, X, Y) # true
@btime broadcast_unzip(f, $X, $Y) # 5.042 ms (4 allocations: 16.00 MiB)
@btime broadcast(f, $X, $Y) # 4.647 ms (2 allocations: 16.00 MiB)
```

The overhead is almost minimized here.

Additionally, `broadcast_unzip` accepts more inputs (just like `broadcast`) as long as `f` outputs a
`Tuple` of a scalar-like object.

```julia
X, Y, Z = rand(1:5, 1024), rand(1:5, 1024), rand(1:5, 1024)
f(x, y, z) = x ^ y ^ z, x / y / z, x * y * z, x / (y*z)
out = broadcast_unzip(f, X, Y, Z)
@assert out[1] == getindex.(f.(X, Y, Z), 1)

@btime broadcast(f, $X, $Y, $Z) # 15.692 μs (2 allocations: 32.05 KiB)
@btime broadcast_unzip(f, $X, $Y, $Z) # 16.072 μs (4 allocations: 32.50 KiB)
```

## Performance caveat -- type stability

Just like the function `map`/`broadcast`, the input function `f` has to be type-inferrable to work
performantly:

```julia
X, Y = rand(1:5, 1024), rand(1:5, 1024)
f_unstable(x, y) = x > 3 ? (x * y, x / y) : (x + y, x - y) # <-- many people make mistakes here
f_stable(x, y) = x > 3 ? (Float64(x * y), Float64(x / y)) : (Float64(x + y), Float64(x - y))

# unstable
@btime broadcast(f_unstable, $X, $Y); # 11.292 μs (1026 allocations: 56.25 KiB)
@btime broadcast_unzip(f_unstable, $X, $Y); # 8.812 μs (403 allocations: 22.52 KiB)

# stable
@btime broadcast(f_stable, $X, $Y); # 1.740 μs (1 allocation: 16.12 KiB)
@btime broadcast_unzip(f_stable, $X, $Y); # 3.018 μs (2 allocations: 16.25 KiB)
```

## Scalar output function not supported

Currently, function `f` that outputs a scalar is delibrately disallowed because there are two
possible result that both makes sense:

- `Array`: degenerates to `broadcast(f, args...)`
- `Tuple{Array}`: degenerated from the generic `broadcast_unzip((f, ), args...)` form (not supported, though)

Thus you'll get an error message here:

```julia
julia> broadcast_unzip(+, [1, 2], [3, 4])
ERROR: function + must return a tuple, instead it returns Int64
Stacktrace:
...
```

## Comparison with other solutions

This package might never exists if I had known the [MappedArrays] + [StructArrays] solutions
provided in the [ANN post]. But it turns out this package is (a little bit) superior to those
alternatives speaking of the performance, see benchmark results in [benchmarks].

[MappedArrays]: https://github.com/JuliaArrays/MappedArrays.jl
[StructArrays]: https://github.com/JuliaArrays/StructArrays.jl
[ANN post]: https://discourse.julialang.org/t/ann-unziploops-broadcasting-and-unzip-the-output-without-overhead/89190
