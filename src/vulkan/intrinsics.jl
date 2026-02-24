# SPIR-V built-in variable access for Vulkan compute shaders
#
# Uses LLVM addrspace(7) globals which map to SPIR-V Input storage class.
# Globals use the __spirv_BuiltIn* naming convention so that the LLVM SPIR-V
# backend automatically generates the correct BuiltIn decorations.

export vk_global_invocation_id_x, vk_local_invocation_id_x,
       vk_workgroup_id_x, vk_num_workgroups_x,
       vk_workgroup_barrier, VkLocalArray

# --- Thread indexing built-ins ---

"""
    vk_global_invocation_id_x() -> UInt32

Return the x-component of gl_GlobalInvocationID (0-based).
Only valid in Vulkan compute shader code compiled via `vk_compile_spirv`.
"""
@inline function vk_global_invocation_id_x()
    Base.llvmcall(("""
        @__spirv_BuiltInGlobalInvocationId = external addrspace(7) global <3 x i32>
        define i32 @entry() #0 {
            %vec = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInGlobalInvocationId, align 16
            %x = extractelement <3 x i32> %vec, i32 0
            ret i32 %x
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

"""
    vk_local_invocation_id_x() -> UInt32

Return the x-component of gl_LocalInvocationID (0-based local thread index).
"""
@inline function vk_local_invocation_id_x()
    Base.llvmcall(("""
        @__spirv_BuiltInLocalInvocationId = external addrspace(7) global <3 x i32>
        define i32 @entry() #0 {
            %vec = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInLocalInvocationId, align 16
            %x = extractelement <3 x i32> %vec, i32 0
            ret i32 %x
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

"""
    vk_workgroup_id_x() -> UInt32

Return the x-component of gl_WorkGroupID (0-based workgroup index).
"""
@inline function vk_workgroup_id_x()
    Base.llvmcall(("""
        @__spirv_BuiltInWorkgroupId = external addrspace(7) global <3 x i32>
        define i32 @entry() #0 {
            %vec = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInWorkgroupId, align 16
            %x = extractelement <3 x i32> %vec, i32 0
            ret i32 %x
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

"""
    vk_num_workgroups_x() -> UInt32

Return the x-component of gl_NumWorkGroups (total number of workgroups).
"""
@inline function vk_num_workgroups_x()
    Base.llvmcall(("""
        @__spirv_BuiltInNumWorkgroups = external addrspace(7) global <3 x i32>
        define i32 @entry() #0 {
            %vec = load <3 x i32>, ptr addrspace(7) @__spirv_BuiltInNumWorkgroups, align 16
            %x = extractelement <3 x i32> %vec, i32 0
            ret i32 %x
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end

# --- Workgroup barrier ---

"""
    vk_workgroup_barrier()

Execute an `OpControlBarrier` with Workgroup execution scope and
AcquireRelease semantics. All invocations in the workgroup must reach
this barrier before any can proceed.

Uses the LLVM SPIR-V backend intrinsic `@llvm.spv.group.memory.barrier.with.group.sync`
which llc directly lowers to `OpControlBarrier` — no SPIR-V text fixup needed.
"""
@inline function vk_workgroup_barrier()
    Base.llvmcall(("""
        declare void @llvm.spv.group.memory.barrier.with.group.sync() #0
        define void @entry() #1 {
            call void @llvm.spv.group.memory.barrier.with.group.sync()
            ret void
        }
        attributes #0 = { convergent nounwind }
        attributes #1 = { alwaysinline convergent }
    """, "entry"), Cvoid, Tuple{})
end

# --- Workgroup shared memory ---
#
# Follows the AMDGPU/OpenCL pattern:
#   1. vk_localmemory() allocates an addrspace(3) global via llvmcall, returns pointer
#   2. VkLocalArray wraps the pointer with standard getindex/setindex! (1-based)
#   3. Access goes through unsafe_load/unsafe_store! on Core.LLVMPtr{T,3}
#
# After inlining, LLVM's InstCombine may flatten the two-level GEP into a
# single-index form that the SPIR-V backend silently drops.  The
# `_fix_shared_geps!` pass in `_prepare_module_for_vulkan!` rewrites these
# back to the structured form before calling llc.

"""Map Julia type to LLVM IR type string."""
function _llvm_type_str(::Type{T}) where T
    T === Float32 && return "float"
    T === Float64 && return "double"
    (T === Int32 || T === UInt32) && return "i32"
    (T === Int64 || T === UInt64) && return "i64"
    (T === Int16 || T === UInt16) && return "i16"
    (T === Int8 || T === UInt8) && return "i8"
    error("Unsupported type $T for Vulkan shared memory")
end

"""
    VkLocalArray{T}

Device-side wrapper around an addrspace(3) shared-memory pointer.
Supports 1-based `getindex`/`setindex!` via `unsafe_load`/`unsafe_store!`
on `Core.LLVMPtr{T, 3}`, matching the AMDGPU `ROCDeviceArray` and
OpenCL `CLDeviceArray` patterns.
"""
struct VkLocalArray{T}
    ptr::Core.LLVMPtr{T, 3}
end

Base.@propagate_inbounds function Base.getindex(a::VkLocalArray{T}, i::Integer) where T
    # 1-based indexing; unsafe_load on LLVMPtr is also 1-based
    unsafe_load(a.ptr, i)
end

Base.@propagate_inbounds function Base.setindex!(a::VkLocalArray{T}, v, i::Integer) where T
    unsafe_store!(a.ptr, convert(T, v), i)
    return a
end

"""
    vk_localmemory(T, Val(len)) -> VkLocalArray{T}

Allocate `len` elements of type `T` in workgroup-local (shared) memory.
Returns a `VkLocalArray` with 1-based indexing.
The length must be a compile-time constant (passed as `Val`).
"""
@generated function vk_localmemory(::Type{T}, ::Val{len}) where {T, len}
    llvm_t = _llvm_type_str(T)
    align = max(sizeof(T), 4)
    # Unique name based on element count and type size to avoid conflicts
    gv_name = "__vk_shared_$(len)x$(sizeof(T))b"

    ir = """
        @$gv_name = internal addrspace(3) global [$len x $llvm_t] zeroinitializer, align $align
        define ptr addrspace(3) @entry() #0 {
            %ptr = getelementptr [$len x $llvm_t], ptr addrspace(3) @$gv_name, i32 0, i32 0
            ret ptr addrspace(3) %ptr
        }
        attributes #0 = { alwaysinline }
    """

    return :(VkLocalArray{$T}(Base.llvmcall(($ir, "entry"), Core.LLVMPtr{$T, 3}, Tuple{})))
end
