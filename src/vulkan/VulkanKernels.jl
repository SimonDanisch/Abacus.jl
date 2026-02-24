module VulkanKernels

using ..Abacus
using ..Abacus: @vk_device_override, VkArray, VkDeviceArray, VkPtr,
    kernel_convert_ka, vkfunction, _pack_push_constants, vk_dispatch!

import KernelAbstractions as KA
import Adapt

export VulkanBackend


## back-end

struct VulkanBackend <: KA.GPU end

@inline KA.allocate(::VulkanBackend, ::Type{T}, dims::Tuple) where T =
    VkArray{T, length(dims)}(undef, dims)

KA.get_backend(::VkArray) = VulkanBackend()
KA.synchronize(::VulkanBackend) = nothing   # dispatch is blocking

KA.supports_float64(::VulkanBackend) = true
KA.supports_atomics(::VulkanBackend) = false

Adapt.adapt_storage(::VulkanBackend, a::Array) = Adapt.adapt(VkArray, a)
Adapt.adapt_storage(::VulkanBackend, a::VkArray) = a
Adapt.adapt_storage(::KA.CPU, a::VkArray) = convert(Array, a)


## memory operations

function KA.copyto!(::VulkanBackend, dest::VkArray{T}, src::VkArray{T}) where T
    GC.@preserve dest src copyto!(dest, src)
    return dest
end

function KA.copyto!(::VulkanBackend, dest::Array{T}, src::VkArray{T}) where T
    GC.@preserve dest src copyto!(dest, src)
    return dest
end

function KA.copyto!(::VulkanBackend, dest::VkArray{T}, src::Array{T}) where T
    GC.@preserve dest src copyto!(dest, src)
    return dest
end


## arg conversion — VkArray → VkDeviceArray for device-side indexing

KA.argconvert(::KA.Kernel{VulkanBackend}, arg) = kernel_convert_ka(arg)


## context creation — uses KA.CompilerMetadata (mirrors AMDGPU's ROCKernels)

function KA.mkcontext(kernel::KA.Kernel{VulkanBackend}, _ndrange, iterspace)
    KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
end

function KA.mkcontext(kernel::KA.Kernel{VulkanBackend}, I, _ndrange, iterspace,
                      ::Dynamic) where Dynamic
    KA.CompilerMetadata{KA.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end


## indexing — @vk_device_override (mirrors AMDGPU's @device_override)
#
# These live on vk_method_table (separate from the CPU backend's method_table),
# so they don't conflict with AbacusKernels' @device_override versions.
#
# Like AMDGPU, we linearize to 1D dispatch and use KA.expand to map back to
# multi-dimensional CartesianIndices via the iterspace stored in CompilerMetadata.

## COV_EXCL_START
@vk_device_override @inline function KA.__index_Local_Linear(ctx)
    return Int(Abacus.vk_local_invocation_id_x()) + 1
end

@vk_device_override @inline function KA.__index_Group_Linear(ctx)
    return Int(Abacus.vk_workgroup_id_x()) + 1
end

@vk_device_override @inline function KA.__index_Global_Linear(ctx)
    I = @inbounds KA.expand(KA.__iterspace(ctx),
                            Int(Abacus.vk_workgroup_id_x()) + 1,
                            Int(Abacus.vk_local_invocation_id_x()) + 1)
    @inbounds LinearIndices(KA.__ndrange(ctx))[I]
end

@vk_device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[Int(Abacus.vk_local_invocation_id_x()) + 1]
end

@vk_device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[Int(Abacus.vk_workgroup_id_x()) + 1]
end

@vk_device_override @inline function KA.__index_Global_Cartesian(ctx)
    return @inbounds KA.expand(KA.__iterspace(ctx),
                               Int(Abacus.vk_workgroup_id_x()) + 1,
                               Int(Abacus.vk_local_invocation_id_x()) + 1)
end

@vk_device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx),
                                Int(Abacus.vk_workgroup_id_x()) + 1,
                                Int(Abacus.vk_local_invocation_id_x()) + 1)
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end

@vk_device_override @inline function KA.__synchronize()
    Abacus.vk_workgroup_barrier()
end

# Vulkan SPIR-V (Shader capability) does not support OpOrdered/OpUnordered,
# which Julia's isless generates via `fcmp ord` for NaN checks.
# Override with simple < comparison (NaN handling dropped, matching GPU convention).
@vk_device_override @inline function Base.isless(x::Float32, y::Float32)
    x < y
end

@vk_device_override @inline function KA.SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    len = prod(Dims)
    Abacus.vk_localmemory(Val(Id), T, Val(len))
end
## COV_EXCL_STOP


## kernel launch (mirrors AMDGPU's ROCKernels pattern)

function KA.launch_config(kernel::KA.Kernel{VulkanBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize,)
    end

    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    # Default workgroup size: match ndrange dimensionality (like AMDGPU)
    # First dimension gets 64 threads, rest get 1.
    if KA.workgroupsize(kernel) <: KA.DynamicSize && workgroupsize === nothing
        if ndrange !== nothing
            workgroupsize = ntuple(
                i -> i == 1 ? min(64, prod(ndrange)) : 1,
                length(ndrange))
        else
            workgroupsize = (64,)
        end
    end

    iterspace, dynamic = KA.partition(kernel, ndrange, workgroupsize)
    return ndrange, workgroupsize, iterspace, dynamic
end

function (obj::KA.Kernel{VulkanBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)

    # Create context with full iterspace (enables multi-D indexing via KA.expand)
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # Convert args: VkArray → VkDeviceArray (supports AbstractArray interface on device)
    converted = map(kernel_convert_ka, args)

    # Build full argument tuple: (ctx, converted_args...)
    all_args = (ctx, converted...)
    tt = Tuple{map(Core.Typeof, all_args)...}

    # 1D linearized dispatch (like AMDGPU)
    nblocks = length(KA.blocks(iterspace))
    nthreads = length(KA.workitems(iterspace))
    nblocks == 0 && return nothing

    # Compile (cached via vkfunction)
    kernel = vkfunction(obj.f, tt; workgroup_size=(nthreads, 1, 1))

    # Pack push constants (recursively flattens structs to match LLVM layout)
    push_data = _pack_push_constants(all_args, kernel.push_size)

    # Dispatch
    groups = (nblocks, 1, 1)
    GC.@preserve args begin
        vk_dispatch!(kernel.pipeline, push_data, groups)
    end

    return nothing
end

end # module VulkanKernels
