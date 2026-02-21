module AbacusKernels

using ..Abacus
using ..Abacus: @device_override, Adaptor, AbacusArray, AbacusDeviceArray, @abacus, abacusconvert

import KernelAbstractions as KA

using StaticArrays: MArray

import Adapt


## back-end

export AbacusBackend

struct AbacusBackend <: KA.GPU
end

@inline KA.allocate(::AbacusBackend, ::Type{T}, dims::Tuple) where T =
    AbacusArray{T, length(dims)}(undef, dims)

KA.get_backend(::AbacusArray) = AbacusBackend()
KA.synchronize(::AbacusBackend) = nothing

KA.supports_float64(::AbacusBackend) = false
KA.supports_atomics(::AbacusBackend) = true

Adapt.adapt_storage(::AbacusBackend, a::Array) = Adapt.adapt(AbacusArray, a)
Adapt.adapt_storage(::AbacusBackend, a::AbacusArray) = a
Adapt.adapt_storage(::KA.CPU, a::AbacusArray) = convert(Array, a)


## memory operations

function KA.copyto!(::AbacusBackend, dest::AbacusArray{T}, src::AbacusArray{T}) where T
    GC.@preserve dest src copyto!(dest, src)
    return dest
end

function KA.copyto!(::AbacusBackend, dest::Array{T}, src::AbacusArray{T}) where T
    GC.@preserve dest src copyto!(dest, src)
    return dest
end

function KA.copyto!(::AbacusBackend, dest::AbacusArray{T}, src::Array{T}) where T
    GC.@preserve dest src copyto!(dest, src)
    return dest
end


## AbacusContext — wraps CompilerMetadata for dispatch isolation
# This prevents our regular Julia methods from conflicting with KA.CPU's methods,
# which also dispatch on CompilerMetadata.

struct AbacusContext{CM}
    cm::CM
end

# Forward all CompilerMetadata accessors
@inline KA.__iterspace(ctx::AbacusContext) = KA.__iterspace(ctx.cm)
@inline KA.__groupindex(ctx::AbacusContext) = KA.__groupindex(ctx.cm)
@inline KA.__groupsize(ctx::AbacusContext) = KA.__groupsize(ctx.cm)
@inline KA.__dynamic_checkbounds(ctx::AbacusContext) = KA.__dynamic_checkbounds(ctx.cm)
@inline KA.__ndrange(ctx::AbacusContext) = KA.__ndrange(ctx.cm)
@inline KA.__workitems_iterspace(ctx::AbacusContext) = KA.__workitems_iterspace(ctx.cm)
@inline KA.groupsize(ctx::AbacusContext) = KA.groupsize(ctx.cm)
@inline KA.ndrange(ctx::AbacusContext) = KA.ndrange(ctx.cm)
@inline Base.ndims(ctx::AbacusContext) = ndims(ctx.cm)


## kernel launch

function KA.mkcontext(kernel::KA.Kernel{AbacusBackend}, _ndrange, iterspace)
    cm = KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
    AbacusContext(cm)
end
function KA.mkcontext(kernel::KA.Kernel{AbacusBackend}, I, _ndrange, iterspace,
                      ::Dynamic) where Dynamic
    cm = KA.CompilerMetadata{KA.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
    AbacusContext(cm)
end

function KA.launch_config(kernel::KA.Kernel{AbacusBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KA.workgroupsize(kernel) <: KA.DynamicSize &&
                            workgroupsize === nothing
        KA.partition(kernel, ndrange, ndrange)
    else
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

KA.argconvert(::KA.Kernel{AbacusBackend}, arg) = Abacus.abacusconvert(arg)

function (obj::KA.Kernel{AbacusBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    # Validate with GPUCompiler (unwrap AbacusContext for validation since GPUCompiler
    # sees CompilerMetadata via @device_override)
    kernel = @abacus launch=false obj.f(ctx.cm, args...)

    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        groupsize = 256
        new_workgroupsize = threads_to_workgroupsize(groupsize, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, new_workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    groups = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))

    if groups == 0
        return nothing
    end

    # Launch kernel — pass CompilerMetadata (matches what was compiled)
    kernel(ctx.cm, args...; threads, groups)
    return nothing
end


## indexing — @device_override versions (for GPUCompiler validation)
# These use CompilerMetadata (unwrapped) and read from kernel_state()

## COV_EXCL_START
@device_override @inline function KA.__index_Local_Linear(ctx)
    return Abacus.thread_position_in_threadgroup_x()
end

@device_override @inline function KA.__index_Group_Linear(ctx)
    return Abacus.threadgroup_position_in_grid_x()
end

@device_override @inline function KA.__index_Global_Linear(ctx)
    I = @inbounds KA.expand(KA.__iterspace(ctx), Abacus.threadgroup_position_in_grid_x(),
                            Abacus.thread_position_in_threadgroup_x())
    @inbounds LinearIndices(KA.__ndrange(ctx))[I]
end

@device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[Abacus.thread_position_in_threadgroup_x()]
end

@device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[Abacus.threadgroup_position_in_grid_x()]
end

@device_override @inline function KA.__index_Global_Cartesian(ctx)
    return @inbounds KA.expand(KA.__iterspace(ctx), Abacus.threadgroup_position_in_grid_x(),
                               Abacus.thread_position_in_threadgroup_x())
end

@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), Abacus.threadgroup_position_in_grid_x(),
                                Abacus.thread_position_in_threadgroup_x())
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end

@device_override @inline function KA.SharedMemory(::Type{T}, ::Val{Dims},
                                                  ::Val{Id}) where {T, Dims, Id}
    N = prod(Dims)
    ptr = Abacus.malloc(N * sizeof(T))
    AbacusDeviceArray{T, length(Dims)}(Dims, reinterpret(Ptr{T}, ptr))
end

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    N = prod(Dims)
    ptr = Abacus.malloc(N * sizeof(T))
    AbacusDeviceArray{T, length(Dims)}(Dims, reinterpret(Ptr{T}, ptr))
end

@device_override @inline function KA.__synchronize()
    nothing
end

@device_override @inline function KA.__print(args...)
    nothing
end
## COV_EXCL_STOP


## indexing — regular Julia methods (for native execution via AbacusContext)

@inline function KA.__index_Local_Linear(ctx::AbacusContext)
    return Abacus.thread_position_in_threadgroup_x()
end

@inline function KA.__index_Group_Linear(ctx::AbacusContext)
    return Abacus.threadgroup_position_in_grid_x()
end

@inline function KA.__index_Global_Linear(ctx::AbacusContext)
    I = @inbounds KA.expand(KA.__iterspace(ctx), Abacus.threadgroup_position_in_grid_x(),
                            Abacus.thread_position_in_threadgroup_x())
    @inbounds LinearIndices(KA.__ndrange(ctx))[I]
end

@inline function KA.__index_Local_Cartesian(ctx::AbacusContext)
    @inbounds KA.workitems(KA.__iterspace(ctx))[Abacus.thread_position_in_threadgroup_x()]
end

@inline function KA.__index_Group_Cartesian(ctx::AbacusContext)
    @inbounds KA.blocks(KA.__iterspace(ctx))[Abacus.threadgroup_position_in_grid_x()]
end

@inline function KA.__index_Global_Cartesian(ctx::AbacusContext)
    return @inbounds KA.expand(KA.__iterspace(ctx), Abacus.threadgroup_position_in_grid_x(),
                               Abacus.thread_position_in_threadgroup_x())
end

@inline function KA.__validindex(ctx::AbacusContext)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), Abacus.threadgroup_position_in_grid_x(),
                                Abacus.thread_position_in_threadgroup_x())
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end


## AcceleratedKernels sort overrides
# GPU merge sort kernels require true shared memory / parallel threads within a workgroup.
# On our sequential CPU backend this doesn't work. Delegate to CPU sort instead.

import AcceleratedKernels as AK

function AK.sort!(v::AbacusArray; kwargs...)
    cpu = Array(v)
    sort!(cpu; kwargs...)
    copyto!(v, cpu)
    return v
end

function AK.sortperm!(ix::AbacusArray, v::AbacusArray; kwargs...)
    cpu_v = Array(v)
    cpu_ix = Array(ix)
    sortperm!(cpu_ix, cpu_v; kwargs...)
    copyto!(ix, cpu_ix)
    return ix
end

function AK.sortperm(v::AbacusArray, ::KA.Backend=AbacusBackend(); kwargs...)
    cpu_v = Array(v)
    cpu_ix = sortperm(cpu_v; kwargs...)
    return AbacusArray(cpu_ix)
end

function AK.merge_sort_by_key!(
    keys::AbacusArray, values::AbacusArray, ::KA.Backend=AbacusBackend(); kwargs...
)
    cpu_keys = Array(keys)
    cpu_values = Array(values)
    perm = sortperm(cpu_keys; kwargs...)
    cpu_keys .= cpu_keys[perm]
    cpu_values .= cpu_values[perm]
    copyto!(keys, cpu_keys)
    copyto!(values, cpu_values)
    return nothing
end

end
