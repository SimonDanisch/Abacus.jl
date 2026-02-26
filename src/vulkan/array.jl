# VkArray — GPUArrays-compatible array backed by Vulkan buffer with BDA
#
# Buffers live in device-local VRAM. Host↔device transfers use staging.
# Mirrors AMDGPU's ROCArray pattern.

export VkArray, VkVector, VkMatrix

const VkArrayStorageType = VkManagedBuffer

mutable struct VkArray{T, N} <: GPUArrays.AbstractGPUArray{T, N}
    data::GPUArrays.DataRef{VkArrayStorageType}
    offset::Int   # offset in number of elements
    dims::Dims{N}

    function VkArray{T, N}(data::GPUArrays.DataRef{VkArrayStorageType},
                           dims::Dims{N}; offset::Int = 0) where {T, N}
        return new{T, N}(data, offset, dims)
    end
end

const VkVector{T} = VkArray{T, 1}
const VkMatrix{T} = VkArray{T, 2}

# --- constructors ---

function VkArray{T, N}(::UndefInitializer, dims::Dims{N}) where {T, N}
    buf = vk_alloc(prod(dims) * sizeof(T))
    ref = GPUArrays.DataRef(_vk_free!, buf)
    return VkArray{T, N}(ref, dims)
end
VkArray{T}(::UndefInitializer, dims::Dims{N}) where {T, N} = VkArray{T, N}(undef, dims)
VkArray{T}(::UndefInitializer, dims::Integer...) where T = VkArray{T}(undef, Dims(dims))
VkArray{T, N}(::UndefInitializer, dims::Integer...) where {T, N} = VkArray{T, N}(undef, Dims(dims))
VkArray{T}(::UndefInitializer, dims::Tuple{Vararg{Integer}}) where T = VkArray{T}(undef, Dims(Int.(dims)))
VkArray{T, N}(::UndefInitializer, dims::Tuple{Vararg{Integer, N}}) where {T, N} = VkArray{T, N}(undef, Dims(Int.(dims)))

VkArray{T, 1}() where {T} = VkArray{T, 1}(undef, (0,))

function VkArray(A::AbstractArray{T, N}) where {T, N}
    arr = VkArray{T, N}(undef, size(A))
    copyto!(arr, collect(A))
    return arr
end

function VkArray{T}(A::AbstractArray{S, N}) where {T, S, N}
    arr = VkArray{T, N}(undef, size(A))
    copyto!(arr, collect(T, A))
    return arr
end

function VkArray{T, N}(A::AbstractArray{S, N}) where {T, S, N}
    arr = VkArray{T, N}(undef, size(A))
    copyto!(arr, collect(T, A))
    return arr
end

# --- GPUArrays interface ---

GPUArrays.storage(a::VkArray) = a.data

function GPUArrays.derive(::Type{T}, a::VkArray{S}, dims::Dims{N}, offset::Int) where {T, S, N}
    ref = copy(a.data)
    offset += (a.offset * sizeof(S)) ÷ sizeof(T)
    return VkArray{T, N}(ref, dims; offset)
end

Base.size(a::VkArray) = a.dims

function Base.sizeof(a::VkArray{T}) where T
    return prod(a.dims) * sizeof(T)
end

# --- device address ---

function device_address(a::VkArray{T}) where T
    buf = a.data[]
    return buf.address + UInt64(a.offset * sizeof(T))
end

# --- scalar indexing (host-side, via staging) ---

function Base.getindex(a::VkArray{T}, i::Int) where T
    @boundscheck checkbounds(a, i)
    GPUArrays.assertscalar("getindex")
    result = Ref{T}()
    GC.@preserve result begin
        byte_offset = (a.offset + i - 1) * sizeof(T)
        download!(Ptr{Nothing}(pointer_from_objref(result)),
                  a.data[], byte_offset, sizeof(T))
    end
    return result[]
end

function Base.setindex!(a::VkArray{T}, v, i::Int) where T
    @boundscheck checkbounds(a, i)
    GPUArrays.assertscalar("setindex!")
    val = Ref(convert(T, v))
    GC.@preserve val begin
        byte_offset = (a.offset + i - 1) * sizeof(T)
        upload!(a.data[], byte_offset,
                Ptr{Nothing}(pointer_from_objref(val)), sizeof(T))
    end
    return a
end

# --- copyto! (mirrors AMDGPU's pattern: offset-based with upload!/download!/transfer!) ---

function Base.copyto!(dest::VkArray{T}, d_offset::Integer,
                      src::Array{T}, s_offset::Integer, n::Integer) where T
    n == 0 && return dest
    @boundscheck checkbounds(dest, d_offset + n - 1)
    @boundscheck checkbounds(src, s_offset + n - 1)
    nbytes = n * sizeof(T)
    byte_offset = (dest.offset + d_offset - 1) * sizeof(T)
    upload!(dest.data[], byte_offset, Ptr{Nothing}(pointer(src, s_offset)), nbytes)
    return dest
end

function Base.copyto!(dest::Array{T}, d_offset::Integer,
                      src::VkArray{T}, s_offset::Integer, n::Integer) where T
    n == 0 && return dest
    @boundscheck checkbounds(dest, d_offset + n - 1)
    @boundscheck checkbounds(src, s_offset + n - 1)
    nbytes = n * sizeof(T)
    byte_offset = (src.offset + s_offset - 1) * sizeof(T)
    download!(Ptr{Nothing}(pointer(dest, d_offset)), src.data[], byte_offset, nbytes)
    return dest
end

function Base.copyto!(dest::VkArray{T}, d_offset::Integer,
                      src::VkArray{T}, s_offset::Integer, n::Integer) where T
    n == 0 && return dest
    @boundscheck checkbounds(dest, d_offset + n - 1)
    @boundscheck checkbounds(src, s_offset + n - 1)
    nbytes = n * sizeof(T)
    dst_byte = (dest.offset + d_offset - 1) * sizeof(T)
    src_byte = (src.offset + s_offset - 1) * sizeof(T)
    transfer!(dest.data[], dst_byte, src.data[], src_byte, nbytes)
    return dest
end

# Convenience: whole-array copyto! delegating to offset-based versions
function Base.copyto!(dest::VkArray{T}, src::Array{T}) where T
    @assert length(dest) == length(src)
    copyto!(dest, 1, src, 1, length(src))
end

function Base.copyto!(dest::Array{T}, src::VkArray{T}) where T
    @assert length(dest) == length(src)
    copyto!(dest, 1, src, 1, length(src))
end

function Base.copyto!(dest::VkArray{T}, src::VkArray{T}) where T
    @assert length(dest) == length(src)
    copyto!(dest, 1, src, 1, length(src))
end

# --- copy ---

function Base.copy(a::VkArray{T, N}) where {T, N}
    b = similar(a)
    copyto!(b, 1, a, 1, length(a))
    return b
end

# --- Array conversion ---

function Base.convert(::Type{Array{T}}, a::VkArray{T}) where T
    result = Vector{T}(undef, length(a))
    copyto!(result, a)
    return reshape(result, size(a))
end

Base.collect(a::VkArray{T}) where T = convert(Array{T}, a)
Base.Array(a::VkArray{T}) where T = convert(Array{T}, a)

# --- similar ---

Base.similar(a::VkArray{T, N}) where {T, N} = VkArray{T, N}(undef, size(a))
Base.similar(a::VkArray{T}, dims::Base.Dims{N}) where {T, N} = VkArray{T, N}(undef, dims)
Base.similar(a::VkArray, ::Type{T}, dims::Base.Dims{N}) where {T, N} = VkArray{T, N}(undef, dims)

# --- eltype conversion ---

function Base.convert(::Type{VkArray{T, N}}, a::AbstractArray{S, N}) where {T, S, N}
    arr = VkArray{T, N}(undef, size(a))
    copyto!(arr, collect(T, a))
    return arr
end

# --- BroadcastStyle ---

import Base.Broadcast: BroadcastStyle, Broadcasted

struct VkArrayStyle{N} <: GPUArraysCore.AbstractGPUArrayStyle{N} end
VkArrayStyle{M}(::Val{N}) where {N, M} = VkArrayStyle{N}()

BroadcastStyle(::Type{<:VkArray{T, N}}) where {T, N} = VkArrayStyle{N}()

Base.similar(bc::Broadcasted{VkArrayStyle{N}}, ::Type{T}, dims) where {T, N} =
    similar(VkArray{T}, dims)

# --- show ---

function Base.show(io::IO, ::MIME"text/plain", a::VkArray{T, N}) where {T, N}
    Base.summary(io, a)
    println(io, ":")
    Base.print_array(io, Array(a))
end

# --- Adapt.jl ---

"""
    VkDeviceArray{T, N}

isbits device-side representation of VkArray. Stores a BDA (UInt64) for GPU access.
"""
struct VkDeviceArray{T, N} <: AbstractArray{T, N}
    dims::Dims{N}
    ptr::UInt64   # buffer device address
end

Base.size(a::VkDeviceArray) = a.dims

@inline function Base.getindex(a::VkDeviceArray{T}, i::Integer) where T
    # No bounds checking on device — GPU can't throw exceptions, and the
    # error paths from @boundscheck create IR patterns (shared error blocks
    # inside loops) that StructurizeCFG cannot handle correctly.
    addr = a.ptr + ((i % UInt64) - UInt64(1)) * UInt64(sizeof(T))
    return unsafe_load(Core.LLVMPtr{T, 1}(addr))
end

@inline function Base.setindex!(a::VkDeviceArray{T}, v, i::Integer) where T
    addr = a.ptr + ((i % UInt64) - UInt64(1)) * UInt64(sizeof(T))
    unsafe_store!(Core.LLVMPtr{T, 1}(addr), convert(T, v))
    return a
end

# Override multi-arg indexing to skip bounds checking.
# Base.getindex(A::AbstractArray, I...) calls @boundscheck checkbounds(A, I...)
# which generates error paths that break StructurizeCFG on GPU.
@inline function Base.getindex(a::VkDeviceArray, I::Integer...)
    @inbounds a[Base._to_linear_index(a, I...)]
end

@inline function Base.setindex!(a::VkDeviceArray, v, I::Integer...)
    @inbounds a[Base._to_linear_index(a, I...)] = v
    return a
end

Base.IndexStyle(::Type{<:VkDeviceArray}) = IndexLinear()

struct VkAdaptor end

function Adapt.adapt_storage(::VkAdaptor, a::VkArray{T, N}) where {T, N}
    return VkDeviceArray{T, N}(a.dims, device_address(a))
end

struct VkRefValue{T} <: Ref{T}
    x::T
end
Base.getindex(r::VkRefValue) = r.x
Adapt.adapt_structure(to::VkAdaptor, ref::Base.RefValue) = VkRefValue(Adapt.adapt(to, ref[]))

# --- VkPtr: primitive BDA pointer type for Vulkan kernels ---

"""
    VkPtr{T}

Primitive 64-bit BDA pointer type for Vulkan compute kernels.
"""
primitive type VkPtr{T} 64 end

@inline VkPtr{T}(addr::UInt64) where T = reinterpret(VkPtr{T}, addr)
@inline Base.UInt64(p::VkPtr) = reinterpret(UInt64, p)

@inline function Base.getindex(p::VkPtr{T}, i::Integer) where T
    addr = reinterpret(UInt64, p) + (UInt64(i) - UInt64(1)) * UInt64(sizeof(T))
    return unsafe_load(Core.LLVMPtr{T, 1}(addr))
end

@inline function Base.setindex!(p::VkPtr{T}, v, i::Integer) where T
    addr = reinterpret(UInt64, p) + (UInt64(i) - UInt64(1)) * UInt64(sizeof(T))
    unsafe_store!(Core.LLVMPtr{T, 1}(addr), convert(T, v))
    return p
end

# --- Adapt.adapt_storage for Type{VkArray} (needed by Adapt.adapt(VkArray, x)) ---

Adapt.adapt_storage(::Type{VkArray}, xs::AbstractArray) = convert(VkArray, xs)
Adapt.adapt_storage(::Type{<:VkArray{T}}, xs::AbstractArray) where {T} = convert(VkArray{T}, xs)

# --- resize! ---

function Base.resize!(a::VkVector{T}, nl::Integer) where {T}
    b = VkVector{T}(undef, (Int(nl),))
    copyto!(b, 1, a, 1, min(length(a), nl))
    a.data = copy(b.data)
    a.offset = b.offset
    a.dims = b.dims
    return a
end

# --- kernel_convert ---
kernel_convert(a::VkArray) = device_address(a)
kernel_convert(x) = x

kernel_convert_ka(a::VkArray{T,N}) where {T,N} = VkDeviceArray{T,N}(size(a), device_address(a))
kernel_convert_ka(x) = Adapt.adapt(VkAdaptor(), x)
