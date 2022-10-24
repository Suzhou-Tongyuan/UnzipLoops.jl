# UnzipLoops

![Julia version](https://img.shields.io/badge/julia-%3E%3D%201.7-blue)

This package provides one single function `broadcast_unzip(f, As...)` that works similar to `map(f,
As...)` but outputs different data layout.

## Example

Broadcasting is very useful in Julia. When you use broadcasting, you'll definitely meet cases like
this:

```julia
f(x, y) = x^y, x/y

X, Y = [1, 2, 3, 4], [4, 3, 2, 1]
out = f.(X, Y)
```

This `out` is of type `Vector{Tuple{Int,Int}}`. It's often the case that we'll need to unzip it to
`Tuple{Vector{Int}, Vector{Int}}`. For most of the case, we can trivially split it apart:

```julia
function g(X, Y)
    out = f.(X, Y)
    return getindex.(out, 1), getindex.(out, 2)
end
```

In this case, since this requires two more broadcasting, it introduces some unnecessary overhead:

- `f.(X, Y)` allocates one extra memory to store the intermediate result, and
- the entire function requires two more loops to do the work and thus hurts the cache locality --
  locality matters for low-level performance optimization.

```julia
X, Y = rand(1:5, 1024), rand(1:5, 1024)
@btime f.($X, $Y) # 3.834 μs (1 allocation: 16.12 KiB)
@btime g($X, $Y) # 5.388 μs (4 allocations: 32.41 KiB)
```

We can observe 1.5μs overhead here.

A more optimized version for this specific `f` is to pre-allocate the output result, and do
everything in one single loop. For instance

```julia
function g(X, Y)
    @assert size(X) == size(Y)
    Base.require_one_based_indexing(X, Y)

    T = promote_type(Float64, eltype(X), eltype(Y))
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
```

We usually call this `g` the SoA (struct of array) layout. Comparing to the AoS (array of struct)
layout that has only one contiguous array involved, this SoA layout introduces two contiguous
arrays. Thus you'll still notice some overhead here compared to the plain `f.(X, Y)` kernel:

```julia
@btime g($X, $Y) # 3.999 μs (2 allocations: 16.25 KiB)
```

but the overhead is relatively small here -- we eliminate the extra loops and allocations. From case
to case, Julia compiler can optimze this difference away.

## `broadcast_unzip`

Obviously, rewriting the trivial `getindex` solution into the verbose manual loop introduces many
work and hurts the readability. This is why `broadcast_unzip` is introduced -- it's a combination of
broadcasting and unzip. Most importantly, this is simple to use and yet fast:

```julia
g(X, Y) == broadcast_unzip(f, X, Y) # true
@btime broadcast_unzip(f, $X, $Y) # 4.009 μs (2 allocations: 16.25 KiB)
```

Additionally, `broadcast_unzip` accepts more inputs (just like `map`) as long as their sizes match
and `f` outputs a `Tuple` of a scalar-like object.

```julia
X, Y, Z = rand(1:5, 1024), rand(1:5, 1024), rand(1:5, 1024)
f(x, y, z) = x ^ y ^ z, x / y / z, x * y * z, x / (y*z)
out = broadcast_unzip(f, X, Y, Z)
@assert out[1] == getindex.(f.(X, Y, Z), 1)

@btime map(f, $X, $Y, $Z) # 13.682 μs (2 allocations: 32.05 KiB)
@btime broadcast_unzip(f, $X, $Y, $Z) # 13.418 μs (6 allocations: 32.58 KiB)
```

## Performance caveat -- type stability

Just like the function `map`, the input function `f` has to be type-inferrable to work performantly:

```julia
X, Y = rand(1:5, 1024), rand(1:5, 1024)
f_unstable(x, y) = x > 3 ? (x * y, x / y) : (x + y, x - y) # <-- many people make mistakes here
f_stable(x, y) = x > 3 ? (Float64(x * y), Float64(x / y)) : (Float64(x + y), Float64(x - y))

# unstable
@btime map(f_unstable, $X, $Y); # 10.619 μs (1026 allocations: 56.25 KiB)
@btime broadcast_unzip(f_unstable, $X, $Y); # 8.614 μs (428 allocations: 22.91 KiB)

# stable
@btime map(f_stable, $X, $Y); # 1.687 μs (1 allocation: 16.12 KiB)
@btime broadcast_unzip(f_stable, $X, $Y); # 3.086 μs (2 allocations: 16.25 KiB)
```
