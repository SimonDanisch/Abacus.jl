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

Arguments:
- `op`:  reduction operator
- `val`: this invocation's accumulated value
- `neutral`: neutral element for `op`
- `shared`: VkLocalArray with at least `wgs` elements (1-based indexing)
- `lid`:  local invocation ID (0-based UInt32)
- `wgs`:  workgroup size
"""
@inline function _reduce_workgroup(op::OP, val::T, neutral::T,
                                   shared::VkLocalArray{T},
                                   lid::UInt32, wgs::UInt32) where {OP, T}
    @inbounds shared[lid + UInt32(1)] = val
    vk_workgroup_barrier()

    stride = wgs >> UInt32(1)
    while stride > UInt32(0)
        if lid < stride
            @inbounds left = shared[lid + UInt32(1)]
            other_idx = lid + stride
            @inbounds right = other_idx < wgs ? shared[other_idx + UInt32(1)] : neutral
            @inbounds shared[lid + UInt32(1)] = op(left, right)
        end
        vk_workgroup_barrier()
        stride >>= UInt32(1)
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
                           ::Val{WGS}) where {F, OP, T, WGS}
    A = VkPtr{T}(A_addr)
    R = VkPtr{T}(R_addr)

    lid = vk_local_invocation_id_x()
    gid = vk_workgroup_id_x()
    ngroups = vk_num_workgroups_x()
    wgs = UInt32(WGS)

    # Allocate workgroup shared memory
    shared = vk_localmemory(T, Val(WGS))

    # Grid-stride accumulation: each thread reduces multiple input elements
    val = neutral
    idx = lid + gid * wgs
    while idx < n
        val = op(val, f(A[(idx % UInt64) + UInt64(1)]))
        idx += wgs * ngroups
    end

    # Workgroup tree reduction
    result = _reduce_workgroup(op, val, neutral, shared, lid, wgs)

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
    # Build argument tuple matching _vk_reduce_kernel signature
    kernel_args = (f, op,
                   device_address(A), device_address(R),
                   n, neutral, Val(wgs))
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
