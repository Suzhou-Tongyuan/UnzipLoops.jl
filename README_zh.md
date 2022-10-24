# UnzipLoops

![Julia version](https://img.shields.io/badge/julia-%3E%3D%201.7-blue)

这个包提供一个函数: `broadcast_unzip(f, As...)`.

## 一个简单的例子

下面是大量使用广播时的一个典型场景

```julia
f(x, y) = x^y, x/y

X, Y = [1, 2, 3, 4], [4, 3, 2, 1]
out = f.(X, Y)
```

这是 `Vector{Tuple{Int, Int}}`， 但经常会遇到希望将它转换为 `Tuple{Vector{Int}, Vector{Int}}` 的需求
(unzip)。 朴素的做法是手动对其进行拆分：

```julia
function g(X, Y)
    out = f.(X, Y)
    return getindex.(out, 1), getindex.(out, 2)
end
```

但由于它引入了两次额外的广播（for循环）， 这会带来不必要的性能损失：

```julia
X, Y = rand(1:5, 1024), rand(1:5, 1024)
@btime f.($X, $Y) # 3.834 μs (1 allocation: 16.12 KiB)
@btime g($X, $Y) # 5.388 μs (4 allocations: 32.41 KiB)
```

对于 `f` 这个特定函数而言， 更高效的方案则是预先分配内存， 然后在一次循环中处理全部事情：

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

虽然相比于 `f` 来说这并不是一个零开销的策略 （这背后涉及到内存布局的差异， 很难达到零开销）， 但它可以显著改善性能：

```julia
@btime g($X, $Y) # 3.999 μs (2 allocations: 16.25 KiB)
```

## `broadcast_unzip` 做了什么

很显然， 上面的改写方案虽然有效， 但相比于朴素的方案来说还是显得有些太长了以至于代码的实际细节被掩盖， 影响可读性。
`broadcast_unzip` 的目的就是为了让这件事变得更简单和高效：

```julia
g(X, Y) == broadcast_unzip(f, X, Y) # true
@btime broadcast_unzip(f, $X, $Y) # 4.009 μs (2 allocations: 16.25 KiB)
```

`broadcast_unzip` 的名字来源于它是两个功能的组合:

- broadcast: `Z = f.(X, Y)`
- unzip: `getindex.(Z, 1), getindex.(Z, 2)`

`broadcast_unzip` 支持多个输入参数， 但要求每个输入参数的尺寸保持一致， 同时函数 `f` 的输出应该是一
个 `Tuple` 且每一项是一个标量值。

```julia
X, Y, Z = rand(1:5, 1024), rand(1:5, 1024), rand(1:5, 1024)
f(x, y, z) = x ^ y ^ z, x / y / z, x * y * z, x / (y*z)
out = broadcast_unzip(f, X, Y, Z)
@assert out[1] == getindex.(f.(X, Y, Z), 1)

@btime map(f, $X, $Y, $Z) # 13.682 μs (2 allocations: 32.05 KiB)
@btime broadcast_unzip(f, $X, $Y, $Z) # 13.418 μs (6 allocations: 32.58 KiB)
```

## 性能要点 -- 类型稳定

与 `map` 一样， 输入函数 `f` 必须要类型稳定才能获得稳定的性能：

```julia
X, Y = rand(1:5, 1024), rand(1:5, 1024)
f_unstable(x, y) = x > 3 ? (x * y, x / y) : (x + y, x - y)
f_stable(x, y) = x > 3 ? (Float64(x * y), Float64(x / y)) : (Float64(x + y), Float64(x - y))

# 类型不稳定
@btime map(f_unstable, $X, $Y); # 10.619 μs (1026 allocations: 56.25 KiB)
@btime broadcast_unzip(f_unstable, $X, $Y); # 8.614 μs (428 allocations: 22.91 KiB)

# 类型稳定
@btime map(f_stable, $X, $Y); # 1.687 μs (1 allocation: 16.12 KiB)
@btime broadcast_unzip(f_stable, $X, $Y); # 3.086 μs (2 allocations: 16.25 KiB)
