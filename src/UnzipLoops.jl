module UnzipLoops

export broadcast_unzip

struct Unroll{N} end
@generated function Unroll{N}(f, args...) where {N}
    code = Expr(:block)
    for i in 1:N
        push!(code.args, :(f($i, args...)))
    end
    push!(code.args, :nothing)
    return code
end

@generated function _teltypes(::Type{Ts}) where {Ts<:Tuple}
    if Ts == Tuple
        error("Cannot infer element types of empty tuple")
    else
        Tuple{(eltype(t) for t in Ts.parameters)...}
    end
end

@generated function _create_vecs_from_record_type(::Type{Ts}, bc) where {Ts}
    if Ts != Tuple && Ts <: Tuple && Ts isa DataType
        Expr(:tuple, [:(similar(bc, $et)) for et in Ts.parameters]...)
    else
        Expr(:tuple, [:(similar(bc, Any)) for _ in 1:length(Ts.parameters)]...)
    end
end

@inline function _infer_record_type(f, xs)
    @static if isdefined(Core, :Compiler)
        T = Core.Compiler.return_type(f, _teltypes(Core.Typeof(xs)))
        TP = Base.promote_typejoin_union(T)
        TP <: Tuple || error("function $f must return a tuple, instead it returns $TP")
        return TP
    else
        return Any
    end
end

struct GetUnzipVectors{F} end
@inline function GetUnzipVectors{F}(bc) where {F}
    xs = bc.args
    length(xs) === 0 && error("get_unzip_vectors() expects more than one array input.")
    t = _infer_record_type(bc.f, xs)
    return _create_vecs_from_record_type(t, bc)
end

struct _Kernel!{M,N} end

@inline @generated function _Kernel!{M,N}(out, bc) where {M,N}
    rhs_out = [:(out[$i]) for i in 1:N]
    lhs_out = [Symbol(:out, i) for i in 1:N]
    lhs_tmp = [Symbol(:tmp, i) for i in 1:N]
    code = Expr(:block)

    for i in 1:N
        push!(code.args, Expr(:(=), lhs_out[i], rhs_out[i]))
    end
    loopblock = Expr(:block)

    push!(
        loopblock.args,
        Expr(
            :(=), Expr(:tuple, lhs_tmp...), Expr(:call, :getindex, :bc, :I)
        )
    )

    for i in 1:N
        push!(loopblock.args, Expr(:(=), :($(lhs_out[i])[I]), lhs_tmp[i]))
    end

    push!(
        code.args,
        quote
            @inbounds @simd for I in eachindex(bc)
                $loopblock
            end
        end,
    )
    return code
end

"""
    broadcast_unzip(f, As...)

For function `f` that outputs a `Tuple`, this function works similar to `map(f, As...)` but
outputs a `Tuple` of arrays instead of an array of `Tuple`s.

# Examples

```julia
julia> using UnzipLoops

julia> f(x, y) = x + y, x - y
f (generic function with 1 method)

julia> X, Y = [1, 2, 3, 4], [4, 3, 2, 1]
([1, 2, 3, 4], [4, 3, 2, 1])

julia> map(f, X, Y) # array of tuple
4-element Vector{Tuple{Int64, Int64}}:
 (5, -3)
 (5, -1)
 (5, 1)
 (5, 3)

julia> broadcast_unzip(f, X, Y) # tuple of array
([5, 5, 5, 5], [-3, -1, 1, 3])
```
"""
function broadcast_unzip(f::F, As...) where {F}
    bc = Broadcast.broadcasted(f, As...)
    out = GetUnzipVectors{F}(bc)
    M, N = length(As), length(out)
    _Kernel!{M,N}(out, bc)
    return out
end

end # module UnzipLoops
