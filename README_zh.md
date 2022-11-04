# UnzipLoops

![Julia version](https://img.shields.io/badge/julia-%3E%3D%201.7-blue)

这个包提供一个函数: `broadcast_unzip(f, As...)`.

## 一个简单的例子

下面是大量使用广播时的一个典型场景

```julia
f(x, y) = x^y, x/y

X, Y = [1, 2, 3, 4], [4 3 2 1]
out = f.(X, Y)
```

这是 `Matrix{Tuple{Int, Float64}}`， 但经常会遇到希望将它转换为 `Tuple{Matrix{Int}, Matrix{Float64}}` 的需求
(unzip)。 朴素的做法是手动对其进行拆分：

```julia
function g(X, Y)
    out = f.(X, Y)
    return map(x->x[1], out), map(x->x[2], out)
end
```

但由于它引入了两次额外的广播（for循环）， 这会带来不必要的性能损失：

```julia
X, Y = rand(1:5, 1024), collect(rand(1:5, 1024)')
@btime f.($X, $Y) # 4.720 ms (2 allocations: 16.00 MiB)
@btime g($X, $Y) # 7.566 ms (6 allocations: 32.00 MiB)
```

对于 `f` 这个特定函数而言， 更高效的方案则是预先分配内存， 然后在一次循环中处理全部事情：

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

虽然相比于 `f` 来说这并不是一个零开销的策略 （这背后涉及到内存布局的差异， 很难达到零开销）， 但它可以显著改善性能：

```julia
@btime g($X, $Y) # 6.878 ms (6 allocations: 32.00 MiB)
```

## `broadcast_unzip` 做了什么

很显然， 上面的改写方案虽然有效， 但相比于朴素的方案来说还是显得有些太长了以至于代码的实际细节被掩盖， 影响可读性。
`broadcast_unzip` 的目的就是为了让这件事变得更简单和高效：

```julia
g(X, Y) == broadcast_unzip(f, X, Y) # true
@btime broadcast_unzip(f, $X, $Y) # 5.042 ms (4 allocations: 16.00 MiB)
@btime broadcast(f, $X, $Y) # 4.647 ms (2 allocations: 16.00 MiB)
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

@btime broadcast(f, $X, $Y, $Z) # 15.692 μs (2 allocations: 32.05 KiB)
@btime broadcast_unzip(f, $X, $Y, $Z) # 16.072 μs (4 allocations: 32.50 KiB)
```

## 性能要点 -- 类型稳定

与 `map`/`broadcast` 一样， 输入函数 `f` 必须要类型稳定才能获得稳定的性能：

```julia
X, Y = rand(1:5, 1024), rand(1:5, 1024)
f_unstable(x, y) = x > 3 ? (x * y, x / y) : (x + y, x - y) # <-- 很多人在这里犯错
f_stable(x, y) = x > 3 ? (Float64(x * y), Float64(x / y)) : (Float64(x + y), Float64(x - y))

# 类型不稳定
@btime broadcast(f_unstable, $X, $Y); # 11.292 μs (1026 allocations: 56.25 KiB)
@btime broadcast_unzip(f_unstable, $X, $Y); # 8.812 μs (403 allocations: 22.52 KiB)

# 类型稳定
@btime broadcast(f_stable, $X, $Y); # 1.740 μs (1 allocation: 16.12 KiB)
@btime broadcast_unzip(f_stable, $X, $Y); # 3.018 μs (2 allocations: 16.25 KiB)
```

## 标量输出函数不支持

输出为标量的函数 `f` 的用法特意被禁止了， 因为存在两种可能的语义：

- `Array`: 退化到 `broadcast(f, args...)`
- `Tuple{Array}`: 从更一般性的 `broadcast_unzip((f, ), args...)` 形式退化过来

因此当你试图输入一个标量函数 `f` 的时候， 你会得到类似于下面的报错:

```julia
julia> broadcast_unzip(+, [1, 2], [3, 4])
ERROR: function + must return a tuple, instead it returns Int64
Stacktrace:
...
```
