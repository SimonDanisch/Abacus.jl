# AbacusArray — GPU-like array following GPUArrays interface (modeled after JLArrays)
# get_backend returns AbacusBackend, triggering GPU code path in KA

import Base.Broadcast: BroadcastStyle, Broadcasted

# type validation — same as Metal.jl
function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

function contains_eltype(T, X)
    if T === X
        return true
    elseif T isa Union
        for U in Base.uniontypes(T)
            contains_eltype(U, X) && return true
        end
    elseif hasfieldcount(T)
        for U in fieldtypes(T)
            contains_eltype(U, X) && return true
        end
    end
    return false
end

function check_eltype(T)
    Base.allocatedinline(T) || error("AbacusArray only supports element types that are stored inline")
    contains_eltype(T, Float64) && error("Abacus does not support Float64 values, try using Float32 instead")
    contains_eltype(T, Int128) && error("Abacus does not support Int128 values, try using Int64 instead")
    contains_eltype(T, UInt128) && error("Abacus does not support UInt128 values, try using UInt64 instead")
end

mutable struct AbacusArray{T, N} <: GPUArraysCore.AbstractGPUArray{T, N}
    data::GPUArrays.DataRef{Vector{UInt8}}
    offset::Int   # in number of elements
    dims::Dims{N}

    # allocating constructor
    function AbacusArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
        check_eltype(T)
        maxsize = prod(dims) * sizeof(T)
        ref = GPUArrays.cached_alloc((AbacusArray, maxsize)) do
            data = Vector{UInt8}(undef, maxsize)
            GPUArrays.DataRef(data) do data
                resize!(data, 0)
            end
        end
        obj = new{T,N}(ref, 0, dims)
        finalizer(GPUArrays.unsafe_free!, obj)
        return obj
    end

    # low-level constructor for wrapping existing data
    function AbacusArray{T,N}(ref::GPUArrays.DataRef{Vector{UInt8}}, dims::Dims{N};
                              offset::Int=0) where {T,N}
        obj = new{T,N}(ref, offset, dims)
        finalizer(GPUArrays.unsafe_free!, obj)
        return obj
    end
end

## type aliases
const AbacusVector{T} = AbacusArray{T,1}
const AbacusMatrix{T} = AbacusArray{T,2}
const DenseAbacusArray{T,N} = AbacusArray{T,N}
const DenseAbacusVector{T} = DenseAbacusArray{T,1}

# anything secretly backed by an AbacusArray
const AnyAbacusArray{T,N} = Union{AbacusArray{T,N}, WrappedArray{T,N,AbacusArray,AbacusArray{T,N}}}
const AnyAbacusVector{T} = AnyAbacusArray{T,1}
const AnyAbacusMatrix{T} = AnyAbacusArray{T,2}

## convenience constructors

# type and dimensionality specified
AbacusArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
    AbacusArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
AbacusArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
    AbacusArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
AbacusArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
    AbacusArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
AbacusArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
    AbacusArray{T,N}(undef, convert(Dims{N}, dims))

# empty vector
AbacusArray{T,1}() where {T} = AbacusArray{T,1}(undef, 0)


## GPUArrays storage interface

GPUArrays.storage(a::AbacusArray) = a.data

function GPUArrays.derive(::Type{T}, a::AbacusArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.data)
    offset = (a.offset * Base.elsize(a)) ÷ sizeof(T) + offset
    AbacusArray{T,N}(ref, dims; offset)
end


## array interface

Base.elsize(::Type{<:AbacusArray{T}}) where {T} = sizeof(T)

Base.size(x::AbacusArray) = x.dims
Base.sizeof(x::AbacusArray) = Base.elsize(x) * length(x)

Base.unsafe_convert(::Type{Ptr{T}}, x::AbacusArray{T}) where {T} =
    convert(Ptr{T}, pointer(x.data[])) + x.offset * Base.elsize(x)

Base.pointer(x::AbacusArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, x)
@inline function Base.pointer(x::AbacusArray{T}, i::Integer) where T
    Base.unsafe_convert(Ptr{T}, x) + Base._memory_offset(x, i)
end

Base.similar(a::AbacusArray{T,N}) where {T,N} = AbacusArray{T,N}(undef, size(a))
Base.similar(a::AbacusArray{T}, dims::Base.Dims{N}) where {T,N} = AbacusArray{T,N}(undef, dims)
Base.similar(a::AbacusArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = AbacusArray{T,N}(undef, dims)

function Base.copy(a::AbacusArray{T,N}) where {T,N}
    b = similar(a)
    @inbounds copyto!(b, a)
end

# conversion of untyped data to a typed Array (for CPU-side operations)
function typed_data(x::AbacusArray{T}) where {T}
    unsafe_wrap(Array, pointer(x), x.dims)
end


## interop with Julia arrays

function AbacusArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N}
    A = AbacusArray{T,N}(undef, size(xs))
    copyto!(A, convert(Array{T}, xs))
    return A
end

# underspecified constructors
AbacusArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = AbacusArray{T,N}(xs)
(::Type{AbacusArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = AbacusArray{S,N}(x)
AbacusArray(A::AbstractArray{T,N}) where {T,N} = AbacusArray{T,N}(A)

# idempotency
AbacusArray{T,N}(xs::AbacusArray{T,N}) where {T,N} = xs


## conversions

Base.convert(::Type{T}, x::T) where T <: AbacusArray = x

Base.collect(a::AbacusArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(a)), a)
Base.Array(a::AbacusArray{T,N}) where {T,N} = collect(a)
Base.convert(::Type{Array{T,N}}, a::AbacusArray{T,N}) where {T,N} = collect(a)

# adapt for the GPU
Adapt.adapt_storage(::Type{AbacusArray}, xs::AbstractArray) =
    isbits(xs) ? xs : convert(AbacusArray, xs)
Adapt.adapt_storage(::Type{<:AbacusArray{T}}, xs::AbstractArray) where {T} =
    isbits(xs) ? xs : convert(AbacusArray{T}, xs)

# adapt back to the CPU
Adapt.adapt_storage(::Type{Array}, xs::AbacusArray) = convert(Array, xs)


## broadcast

struct AbacusArrayStyle{N} <: GPUArrays.AbstractGPUArrayStyle{N} end
AbacusArrayStyle{M}(::Val{N}) where {N,M} = AbacusArrayStyle{N}()

BroadcastStyle(::Type{<:AbacusArray{T,N}}) where {T,N} = AbacusArrayStyle{N}()
BroadcastStyle(::Type{<:AnyAbacusArray{T,N}}) where {T,N} = AbacusArrayStyle{N}()

Base.similar(bc::Broadcasted{AbacusArrayStyle{N}}, ::Type{T}, dims) where {T,N} =
    similar(AbacusArray{T}, dims)


## memory operations

function Base.copyto!(dest::Array{T}, d_offset::Integer,
                      source::DenseAbacusArray{T}, s_offset::Integer,
                      amount::Integer) where T
    amount == 0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset + amount - 1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset + amount - 1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                  pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::Array{T}, source::DenseAbacusArray{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))

function Base.copyto!(dest::DenseAbacusArray{T}, d_offset::Integer,
                      source::Array{T}, s_offset::Integer,
                      amount::Integer) where T
    amount == 0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset + amount - 1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset + amount - 1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                  pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::DenseAbacusArray{T}, source::Array{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))

function Base.copyto!(dest::DenseAbacusArray{T}, d_offset::Integer,
                      source::DenseAbacusArray{T}, s_offset::Integer,
                      amount::Integer) where T
    amount == 0 && return dest
    @boundscheck checkbounds(dest, d_offset)
    @boundscheck checkbounds(dest, d_offset + amount - 1)
    @boundscheck checkbounds(source, s_offset)
    @boundscheck checkbounds(source, s_offset + amount - 1)
    GC.@preserve dest source Base.unsafe_copyto!(pointer(dest, d_offset),
                                                  pointer(source, s_offset), amount)
    return dest
end

Base.copyto!(dest::DenseAbacusArray{T}, source::DenseAbacusArray{T}) where {T} =
    copyto!(dest, 1, source, 1, length(source))


## resize!

function Base.resize!(a::DenseAbacusVector{T}, nl::Integer) where {T}
    b = AbacusVector{T}(undef, nl)
    copyto!(b, 1, a, 1, min(length(a), nl))
    GPUArrays.unsafe_free!(a)
    a.data = copy(b.data)
    a.offset = b.offset
    a.dims = b.dims
    return a
end


## mapreducedim!

function GPUArrays.mapreducedim!(f, op, R::AnyAbacusArray, A::Union{AbstractArray,Broadcast.Broadcasted};
                                  init=nothing)
    if init !== nothing
        fill!(R, init)
    end
    GPUArraysCore.@allowscalar Base.reducedim!(op, typed_data(R), map(f, A))
    R
end


## AbacusDeviceArray — isbits wrapper for GPU validation
# Wraps a pointer for isbits compatibility while supporting array operations during native execution.

struct AbacusDeviceArray{T, N} <: AbstractArray{T, N}
    dims::Dims{N}
    ptr::Ptr{T}
end

Base.size(a::AbacusDeviceArray) = a.dims
Base.IndexStyle(::Type{<:AbacusDeviceArray}) = IndexLinear()

@inline function Base.getindex(a::AbacusDeviceArray{T}, i::Int) where T
    unsafe_load(a.ptr, i)
end
@inline function Base.setindex!(a::AbacusDeviceArray{T}, v, i::Int) where T
    unsafe_store!(a.ptr, convert(T, v)::T, i)
    return v
end

Base.pointer(a::AbacusDeviceArray{T}) where {T} = a.ptr
@inline function Base.pointer(a::AbacusDeviceArray{T}, i::Integer) where T
    a.ptr + (i - 1) * sizeof(T)
end


## Adapt.jl: convert AbacusArray → AbacusDeviceArray for kernel args

struct Adaptor end

function Adapt.adapt_storage(::Adaptor, a::AbacusArray{T,N}) where {T,N}
    AbacusDeviceArray{T,N}(a.dims, pointer(a))
end
Adapt.adapt_storage(::Adaptor, a::Array) = a


## random number generation

using Random

const ABACUS_GLOBAL_RNG = Ref{Union{Nothing,GPUArrays.RNG}}(nothing)
function GPUArrays.default_rng(::Type{<:AbacusArray})
    if ABACUS_GLOBAL_RNG[] === nothing
        N = 256
        state = AbacusArray{NTuple{4, UInt32}}(undef, N)
        rng = GPUArrays.RNG(state)
        Random.seed!(rng)
        ABACUS_GLOBAL_RNG[] = rng
    end
    ABACUS_GLOBAL_RNG[]
end
