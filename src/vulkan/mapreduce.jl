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
                           ::Val{WGS}, ::Val{MAX_ITERS},
                           ::Val{S}) where {F, OP, T, WGS, MAX_ITERS, S}
    A = VkPtr{S}(A_addr)
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

"""
Partial reduction kernel: reduce along selected dimensions using CartesianIndices.

Mirrors AMDGPU's `partial_mapreduce_device`. Each workgroup handles one element
from the "other" (non-reduced) dimensions and collaboratively reduces across
the "reduce" dimensions using shared memory tree reduction.

`Rreduce` and `Rother` are CartesianIndices that decompose the input domain:
- `Rother` covers the output dimensions (each element = one output slot)
- `Rreduce` covers the dimensions being reduced
- `max(Iother, Ireduce)` recombines them into a full input index

`R` has an extra trailing dimension for multi-pass support (size 1 for single-pass,
`reduce_groups` for multi-pass).
"""
function _vk_partial_reduce_kernel(f::F, op::OP, neutral::T,
                                    Rreduce::RR, Rother::RO,
                                    R::RA, A::AA,
                                    ::Val{WGS}, ::Val{MAX_ITERS}) where {F, OP, T, RR, RO, RA, AA, WGS, MAX_ITERS}
    shared = vk_localmemory(T, Val(WGS))

    lid = vk_local_invocation_id_x()       # 0-indexed
    gid = vk_workgroup_id_x() + UInt32(1)  # 1-indexed
    ngroups = vk_num_workgroups_x()
    wgs = UInt32(WGS)

    other_len = UInt32(length(Rother))

    # Decompose workgroup index: inner = other, outer = reduce
    # Matches AMDGPU's fldmod1(workgroupIdx().x, length(Rother))
    groupIdx_other  = ((gid - UInt32(1)) % other_len) + UInt32(1)
    groupIdx_reduce = ((gid - UInt32(1)) ÷ other_len) + UInt32(1)
    groupDim_reduce = ngroups ÷ other_len

    @inbounds if groupIdx_other <= other_len
        Iother = Rother[groupIdx_other]

        # Output index includes trailing reduce-group dimension
        Iout = CartesianIndex(Tuple(Iother)..., groupIdx_reduce)

        val = op(neutral, neutral)

        # Grid-stride reduction with bounded iteration count
        ireduce = lid + UInt32(1) + (groupIdx_reduce - UInt32(1)) * wgs
        reduce_len = UInt32(length(Rreduce))
        stride = wgs * groupDim_reduce

        for _ in UInt32(1):UInt32(MAX_ITERS)
            if ireduce <= reduce_len
                Ireduce = Rreduce[ireduce]
                J = max(Iother, Ireduce)
                val = op(val, f(A[J]))
            end
            ireduce += stride
        end

        # Workgroup tree reduction
        result = _reduce_workgroup(op, val, neutral, shared, lid, Val(WGS))

        # First thread writes workgroup result
        if lid == UInt32(0)
            R[Iout] = result
        end
    end

    return nothing
end

## COV_EXCL_STOP

# ── Full reduction (existing optimized 1D kernel) ──

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

"""Compile (cached) and dispatch the full reduction kernel."""
function _dispatch_reduce!(f, op, A::VkArray{S}, R::VkArray{T},
                           n::UInt32, neutral::T,
                           wgs::Int, ngroups::Int) where {S, T}
    total_threads = wgs * ngroups
    max_iters = cld(Int(n), total_threads)

    kernel_args = (f, op,
                   device_address(A), device_address(R),
                   n, neutral, Val(wgs), Val(max_iters), Val(S))
    adapted_args = map(kernel_convert, kernel_args)
    adapted_tt = Tuple{map(Core.Typeof, adapted_args)...}

    kernel = vkfunction(_vk_reduce_kernel, adapted_tt;
                        workgroup_size=(wgs, 1, 1))
    bda_data, arg_buf = _pack_and_upload_args(adapted_args, kernel.arg_buffer_size)

    groups = (ngroups, 1, 1)
    GC.@preserve A R begin
        vk_dispatch!(kernel.pipeline, bda_data, groups)
    end
end

# ── Partial reduction (general CartesianIndex-based kernel) ──

"""
    _gpu_partial_reduce!(f, op, R::VkArray{T}, A, init) -> R

General partial reduction using CartesianIndices decomposition.
Handles partial reductions along arbitrary dimensions and Broadcasted inputs.
Mirrors AMDGPU's mapreducedim! strategy.
"""
function _gpu_partial_reduce!(f, op, R::VkArray{T}, A, init) where T
    if init === nothing
        neutral = GPUArrays.neutral_element(op, T)
    else
        neutral = convert(T, init)
    end

    # Add singleton dimensions if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    # Split iteration domain: reduce × other = all
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    @assert length(Rall) == length(Rother) * length(Rreduce)

    # Add trailing dimension for multi-pass support
    R′ = reshape(R, (size(R)..., 1))

    # Workgroup size: power of 2, capped at REDUCE_WGS
    wgs = min(nextpow(2, length(Rreduce)), REDUCE_WGS)
    wgs = max(wgs, 1)

    # Cap reduction groups to keep second pass small
    reduce_groups = min(cld(length(Rreduce), wgs), 64)

    if reduce_groups == 1
        _dispatch_partial_reduce!(f, op, neutral, Rreduce, Rother, R′, A, wgs, 1)
    else
        partial = similar(R, (size(R)..., reduce_groups))
        GC.@preserve partial begin
            _dispatch_partial_reduce!(f, op, neutral, Rreduce, Rother, partial, A, wgs, reduce_groups)
            # Second pass: reduce partial results along the trailing dimension
            GPUArrays.mapreducedim!(identity, op, R′, partial; init=neutral)
        end
    end

    return R
end

"""Compile (cached) and dispatch the partial reduction kernel."""
function _dispatch_partial_reduce!(f, op, neutral::T, Rreduce, Rother,
                                    R::VkArray, A,
                                    wgs::Int, reduce_groups::Int) where T
    ngroups = reduce_groups * length(Rother)
    max_iters = max(1, cld(length(Rreduce), wgs * reduce_groups))

    # Convert arrays for device-side indexing (VkArray → VkDeviceArray)
    R_dev = kernel_convert_ka(R)
    A_dev = kernel_convert_ka(A)

    args = (f, op, neutral, Rreduce, Rother, R_dev, A_dev,
            Val(wgs), Val(max_iters))
    tt = Tuple{map(Core.Typeof, args)...}

    kernel = vkfunction(_vk_partial_reduce_kernel, tt;
                        workgroup_size=(wgs, 1, 1))
    bda_data, arg_buf = _pack_and_upload_args(args, kernel.arg_buffer_size)

    groups = (ngroups, 1, 1)
    GC.@preserve R A begin
        vk_dispatch!(kernel.pipeline, bda_data, groups)
    end
end

# ── GPUArrays.mapreducedim! implementation ──

function GPUArrays.mapreducedim!(f, op, R::VkArray{T},
                                 A::Union{VkArray, Base.Broadcast.Broadcasted};
                                 init=nothing) where T
    Base.check_reducedims(R, A)
    length(A) == 0 && return R

    # Fast path: full reduction of VkArray input (optimized 1D kernel)
    if A isa VkArray && prod(size(R)) == 1
        return _gpu_full_reduce!(f, op, R, A, init)
    end

    # General path: partial reductions and/or Broadcasted inputs
    return _gpu_partial_reduce!(f, op, R, A, init)
end

# ── CPU fallback for findfirst/findlast ──
# GPU findfirst/findlast uses partial reduce with Tuple{Bool, Int64} which
# generates OpConvertPtrToU on workgroup pointers (illegal in Vulkan SPIR-V).
# The full reduce path works (via _gpu_full_reduce!), but partial reduce
# doesn't because struct-field GEPs aren't fully merged in that kernel.
# Fall back to CPU for now; fix requires deeper LLVM IR fixup for partial reduce.

function Base.findfirst(f::Function, A::VkArray)
    cpu_arr = Array(A)
    return findfirst(f, cpu_arr)
end

function Base.findlast(f::Function, A::VkArray)
    cpu_arr = Array(A)
    return findlast(f, cpu_arr)
end

Base.findfirst(A::VkArray{Bool}) = findfirst(identity, A)
Base.findlast(A::VkArray{Bool}) = findlast(identity, A)
