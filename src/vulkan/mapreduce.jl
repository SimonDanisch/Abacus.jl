# GPU mapreduce for VkArray — workgroup-based parallel reduction
#
# Uses shared memory and tree reduction within workgroups, with a two-pass
# strategy for large inputs: first pass reduces to per-workgroup partials,
# second pass reduces partials to the final result.

## COV_EXCL_START

# Default workgroup size for reduction kernels
const REDUCE_WGS = 256

"""
Reduce a value across a workgroup using shared memory tree reduction.

Uses compile-time unrolled barriers (no while loop) to avoid structured
control flow issues in Vulkan SPIR-V. Mirrors AcceleratedKernels' reduce_group!.
"""
@inline function _reduce_workgroup(op::OP, val::T, ::T,
                                   shared::VkLocalArray{T},
                                   lid::UInt32, ::Val{WGS}) where {OP, T, WGS}
    @inbounds shared[lid + UInt32(1)] = val
    vk_workgroup_barrier()

    # Compile-time unrolled tree reduction — each `if` is statically eliminated
    # when WGS is smaller than the threshold, so no runtime loop needed.
    if WGS >= 1024
        if lid < UInt32(512)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(512) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 512
        if lid < UInt32(256)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(256) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 256
        if lid < UInt32(128)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(128) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 128
        if lid < UInt32(64)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(64) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 64
        if lid < UInt32(32)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(32) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 32
        if lid < UInt32(16)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(16) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 16
        if lid < UInt32(8)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(8) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 8
        if lid < UInt32(4)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(4) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 4
        if lid < UInt32(2)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(2) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end
    if WGS >= 2
        if lid < UInt32(1)
            @inbounds shared[lid + UInt32(1)] = op(shared[lid + UInt32(1)], shared[lid + UInt32(1) + UInt32(1)])
        end
        vk_workgroup_barrier()
    end

    return @inbounds shared[UInt32(1)]
end

"""
GPU reduction kernel: map + reduce input array to per-workgroup partial results.

Each workgroup processes a strided chunk of the input, reduces within the
workgroup using shared memory, and writes one result per workgroup to `R`.
"""
function _vk_reduce_kernel(f::F, op::OP,
                           A_addr::UInt64, R_addr::UInt64,
                           n::UInt32, neutral::T,
                           ::Val{WGS}, ::Val{MAX_ITERS}) where {F, OP, T, WGS, MAX_ITERS}
    A = VkPtr{T}(A_addr)
    R = VkPtr{T}(R_addr)

    lid = vk_local_invocation_id_x()
    gid = vk_workgroup_id_x()
    ngroups = vk_num_workgroups_x()
    wgs = UInt32(WGS)

    # Allocate workgroup shared memory
    shared = vk_localmemory(T, Val(WGS))

    # Grid-stride accumulation with bounded iteration count.
    # MAX_ITERS is computed host-side as ceil(n / (wgs * ngroups)).
    # Using a bounded for loop avoids data-dependent while loops that
    # cause structured control flow issues in Vulkan SPIR-V.
    val = neutral
    idx = lid + gid * wgs
    stride = wgs * ngroups
    for _ in UInt32(1):UInt32(MAX_ITERS)
        if idx < n
            val = op(val, f(A[(idx % UInt64) + UInt64(1)]))
        end
        idx += stride
    end

    # Workgroup tree reduction (compile-time unrolled, no while loop)
    result = _reduce_workgroup(op, val, neutral, shared, lid, Val(WGS))

    # First thread writes workgroup result
    if lid == UInt32(0)
        R[(gid % UInt64) + UInt64(1)] = result
    end

    return nothing
end

## COV_EXCL_STOP

"""
    _gpu_full_reduce!(f, op, R::VkArray{T}, A::VkArray, init) -> R

Full reduction of `A` (all dimensions) into scalar `R` using GPU compute.
Two-pass strategy: first pass produces per-workgroup partials, second pass
reduces those to a single value.
"""
function _gpu_full_reduce!(f, op, R::VkArray{T}, A::VkArray, init) where T
    n = length(A)
    wgs = REDUCE_WGS

    if init === nothing
        init = GPUArrays.neutral_element(op, T)
    end
    neutral = convert(T, init)

    ngroups = min(cld(n, wgs), 1024)

    if ngroups <= 1
        ngroups = 1
        _dispatch_reduce!(f, op, A, R, UInt32(n), neutral, wgs, ngroups)
    else
        # Two-pass reduction
        partial = VkArray{T, 1}(undef, (ngroups,))
        GC.@preserve partial begin
            _dispatch_reduce!(f, op, A, partial, UInt32(n), neutral, wgs, ngroups)
            _dispatch_reduce!(identity, op, partial, R, UInt32(ngroups), neutral, wgs, 1)
        end
    end

    return R
end

"""Compile (cached) and dispatch the reduction kernel."""
function _dispatch_reduce!(f, op, A::VkArray{T}, R::VkArray{T},
                           n::UInt32, neutral::T,
                           wgs::Int, ngroups::Int) where T
    # Compute max iterations for grid-stride loop (compile-time constant)
    total_threads = wgs * ngroups
    max_iters = cld(Int(n), total_threads)

    # Build argument tuple matching _vk_reduce_kernel signature
    kernel_args = (f, op,
                   device_address(A), device_address(R),
                   n, neutral, Val(wgs), Val(max_iters))
    adapted_args = map(kernel_convert, kernel_args)
    adapted_tt = Tuple{map(Core.Typeof, adapted_args)...}

    kernel = vkfunction(_vk_reduce_kernel, adapted_tt;
                        workgroup_size=(wgs, 1, 1))
    push_data = _pack_push_constants(adapted_args, kernel.push_size)

    groups = (ngroups, 1, 1)
    GC.@preserve A R begin
        vk_dispatch!(kernel.pipeline, push_data, groups)
    end
end

# --- GPUArrays.mapreducedim! implementation ---

function GPUArrays.mapreducedim!(f, op, R::VkArray{T},
                                 A::Union{VkArray, Base.Broadcast.Broadcasted};
                                 init=nothing) where T
    Base.check_reducedims(R, A)
    length(A) == 0 && return R

    # GPU fast path: full reduction of VkArray input (R is scalar-sized)
    if A isa VkArray && prod(size(R)) == 1
        return _gpu_full_reduce!(f, op, R, A, init)
    end

    # CPU fallback for partial reductions and Broadcasted inputs.
    # Uses bulk memcpy (not scalar indexing) — acceptable until we implement
    # a general GPU partial-reduction kernel with CartesianIndex support.
    hostA = A isa VkArray ? Array(A) : GPUArrays.@allowscalar map(identity, A)
    hostR = Array(R)
    if init !== nothing
        fill!(hostR, init)
    end
    Base.mapreducedim!(f, op, hostR, hostA)
    copyto!(R, hostR)
    return R
end
