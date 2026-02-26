# Vulkan SPIR-V atomic operations
#
# Uses LLVM `atomicrmw` instructions with `syncscope("device")` which the
# LLVM SPIR-V backend translates to OpAtomic* instructions with Device scope.
# Vulkan requires explicit scope — the default (CrossDevice) is invalid.

import Atomix

# ── Low-level atomicrmw intrinsics ───────────────────────────────────

# Generate atomicrmw intrinsics for integer types on addrspace(1) (PhysicalStorageBuffer).
# The LLVM SPIR-V backend maps these to OpAtomicIAdd, OpAtomicISub, etc.
#
# IR strings must be constructed before @eval to avoid runtime string interpolation.

const _ATOMIC_INT_TYPES = (
    (Int32,  "i32"),
    (UInt32, "i32"),   # same LLVM type, Julia dispatches on the wrapper
)

for (jltype, llvm_ty) in _ATOMIC_INT_TYPES
    for (op_name, rmw_op) in ((:_vk_atomic_add, "add"),
                               (:_vk_atomic_sub, "sub"),
                               (:_vk_atomic_and, "and"),
                               (:_vk_atomic_or,  "or"),
                               (:_vk_atomic_xor, "xor"),
                               (:_vk_atomic_xchg, "xchg"))
        ir = """
            define $llvm_ty @entry(ptr addrspace(1) %ptr, $llvm_ty %val) #0 {
                %old = atomicrmw $rmw_op ptr addrspace(1) %ptr, $llvm_ty %val syncscope("device") monotonic
                ret $llvm_ty %old
            }
            attributes #0 = { alwaysinline }
        """
        @eval @inline function $op_name(ptr::Core.LLVMPtr{$jltype, 1}, val::$jltype)
            Base.llvmcall(($ir, "entry"), $jltype, Tuple{Core.LLVMPtr{$jltype, 1}, $jltype}, ptr, val)
        end
    end

    # min/max: signed vs unsigned
    min_op = jltype <: Signed ? "min" : "umin"
    max_op = jltype <: Signed ? "max" : "umax"

    for (op_name, rmw_op) in ((:_vk_atomic_min, min_op), (:_vk_atomic_max, max_op))
        ir = """
            define $llvm_ty @entry(ptr addrspace(1) %ptr, $llvm_ty %val) #0 {
                %old = atomicrmw $rmw_op ptr addrspace(1) %ptr, $llvm_ty %val syncscope("device") monotonic
                ret $llvm_ty %old
            }
            attributes #0 = { alwaysinline }
        """
        @eval @inline function $op_name(ptr::Core.LLVMPtr{$jltype, 1}, val::$jltype)
            Base.llvmcall(($ir, "entry"), $jltype, Tuple{Core.LLVMPtr{$jltype, 1}, $jltype}, ptr, val)
        end
    end
end

# Float32 atomic add via CAS loop (atomicrmw fadd not supported by SPIR-V backend)
@inline function _vk_atomic_add(ptr::Core.LLVMPtr{Float32, 1}, val::Float32)
    # Reinterpret as UInt32, do CAS loop
    iptr = Core.LLVMPtr{UInt32, 1}(ptr)
    while true
        old_bits = Base.llvmcall(("""
            define i32 @entry(ptr addrspace(1) %ptr) #0 {
                %val = load i32, ptr addrspace(1) %ptr
                ret i32 %val
            }
            attributes #0 = { alwaysinline }
        """, "entry"), UInt32, Tuple{Core.LLVMPtr{UInt32, 1}}, iptr)
        old_f = reinterpret(Float32, old_bits)
        new_f = old_f + val
        new_bits = reinterpret(UInt32, new_f)
        # cmpxchg returns {i32, i1} — extract both via custom llvmcall
        exchanged = Base.llvmcall(("""
            define i32 @entry(ptr addrspace(1) %ptr, i32 %expected, i32 %desired) #0 {
                %result = cmpxchg ptr addrspace(1) %ptr, i32 %expected, i32 %desired syncscope("device") monotonic monotonic
                %old = extractvalue {i32, i1} %result, 0
                %success = extractvalue {i32, i1} %result, 1
                %ret = select i1 %success, i32 1, i32 0
                ret i32 %ret
            }
            attributes #0 = { alwaysinline }
        """, "entry"), UInt32, Tuple{Core.LLVMPtr{UInt32, 1}, UInt32, UInt32}, iptr, old_bits, new_bits)
        exchanged != UInt32(0) && return old_f
    end
end

# ── Dispatch table: op -> intrinsic ──────────────────────────────────

@inline _vk_atomic_op(::typeof(+), ptr, val) = _vk_atomic_add(ptr, val)
@inline _vk_atomic_op(::typeof(-), ptr, val) = _vk_atomic_sub(ptr, val)
@inline _vk_atomic_op(::typeof(&), ptr, val) = _vk_atomic_and(ptr, val)
@inline _vk_atomic_op(::typeof(|), ptr, val) = _vk_atomic_or(ptr, val)
@inline _vk_atomic_op(::typeof(xor), ptr, val) = _vk_atomic_xor(ptr, val)
@inline _vk_atomic_op(::typeof(min), ptr, val) = _vk_atomic_min(ptr, val)
@inline _vk_atomic_op(::typeof(max), ptr, val) = _vk_atomic_max(ptr, val)

# ── Atomix.jl integration ───────────────────────────────────────────
#
# Hook into Atomix.modify! for VkDeviceArray so that
#   @atomic arr[i] += val
# generates hardware atomic instructions.

const VkIndexableRef{Indexable <: VkDeviceArray} = Atomix.IndexableRef{Indexable}

@inline function Atomix.modify!(ref::VkIndexableRef, op::OP, x, ord) where OP
    x = Atomix.asstorable(ref, x)
    # Compute the LLVMPtr for the element
    arr = ref.data::VkDeviceArray
    idx = ref.indices[1]::Int
    T = eltype(arr)
    addr = arr.ptr + ((idx % UInt64) - UInt64(1)) * UInt64(sizeof(T))
    ptr = Core.LLVMPtr{T, 1}(addr)
    old = _vk_atomic_op(op, ptr, x)
    return old => op(old, x)
end
