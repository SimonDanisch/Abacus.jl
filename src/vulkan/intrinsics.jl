# SPIR-V intrinsics for Vulkan compute shaders
#
# Adapted from OpenCL.jl's SPIRVIntrinsics library. Both backends target SPIR-V
# so the intrinsic names and calling conventions are identical.
#
# Uses addrspace(7) for SPIR-V Input storage class (built-in variables).
# Uses addrspace(3) for SPIR-V Workgroup storage class (shared memory).
# Uses the __spirv_BuiltIn* naming convention for automatic BuiltIn decorations.

using LLVM.Interop: @typed_ccall
using Core: LLVMPtr

export vk_global_invocation_id_x, vk_local_invocation_id_x,
       vk_workgroup_id_x, vk_num_workgroups_x,
       vk_workgroup_barrier, VkLocalArray


# ── @builtin_ccall ─────────────────────────────────────────────────────
# C++ Itanium-mangled calls to OpenCL/SPIR-V built-in functions.
# Copied from OpenCL.jl/lib/intrinsics/src/utils.jl.

const known_intrinsics = String[]

macro builtin_ccall(name, ret, argtypes, args...)
    @assert Meta.isexpr(argtypes, :tuple)
    argtypes = argtypes.args

    function mangle(T::Type)
        T == Int8    ? "c" :
        T == UInt8   ? "h" :
        T == Int16   ? "s" :
        T == UInt16  ? "t" :
        T == Int32   ? "i" :
        T == UInt32  ? "j" :
        T == Int64   ? "l" :
        T == UInt64  ? "m" :
        T == Float16 ? "Dh" :
        T == Float32 ? "f" :
        T == Float64 ? "d" :
        T <: LLVMPtr ? begin
            elt, as = T.parameters
            (as == 10 ? "P" : "PU3AS$as") * "V" * mangle(elt)
        end :
        error("Unknown type $T for @builtin_ccall mangling")
    end
    mangle(::Type{NTuple{N, VecElement{T}}}) where {N, T} = "Dv$(N)_" * mangle(T)

    mangled = "_Z$(length(name))$name"
    for t in argtypes
        t = (isa(t, Symbol) || isa(t, Expr)) ? __module__.eval(t) : t
        mangled *= mangle(t)
    end

    push!(__module__.known_intrinsics, mangled)
    esc(quote
        @typed_ccall($mangled, llvmcall, $ret, ($(argtypes...),), $(args...))
    end)
end


# ── SPIR-V Scope constants ────────────────────────────────────────────

module Scope
    const CrossDevice = UInt32(0)
    const Device      = UInt32(1)
    const Workgroup   = UInt32(2)
    const Subgroup    = UInt32(3)
    const Invocation  = UInt32(4)
end

# ── SPIR-V MemorySemantics constants ──────────────────────────────────

module MemorySemantics
    const Relaxed                = UInt32(0x0000)
    const Acquire                = UInt32(0x0002)
    const Release                = UInt32(0x0004)
    const AcquireRelease         = UInt32(0x0008)
    const SequentiallyConsistent = UInt32(0x0010)
    const UniformMemory          = UInt32(0x0040)
    const SubgroupMemory         = UInt32(0x0080)
    const WorkgroupMemory        = UInt32(0x0100)
    const CrossWorkgroupMemory   = UInt32(0x0200)
    const ImageMemory            = UInt32(0x0800)
end


# ── Synchronization ───────────────────────────────────────────────────

# NOTE: The Itanium-mangled __spirv_ControlBarrier / __spirv_MemoryBarrier names
# do NOT work with the Vulkan SPIR-V target (llc leaves them as unresolved function
# declarations requiring Linkage capability). Only the llvm.spv.* intrinsics work.
# The only available barrier is @llvm.spv.group.memory.barrier.with.group.sync which
# hardcodes SequentiallyConsistent semantics. _fix_barrier_semantics! in spirv_fixup.jl
# patches these to AcquireRelease|WorkgroupMemory in the binary SPIR-V output.

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


# ── Thread indexing built-ins ─────────────────────────────────────────
# 3D built-in variables via addrspace(7) globals.
# dim argument is 1-based (Julia convention): 1=x, 2=y, 3=z.

const _BUILTIN_3D = Dict(
    :vk_global_invocation_id => :__spirv_BuiltInGlobalInvocationId,
    :vk_local_invocation_id  => :__spirv_BuiltInLocalInvocationId,
    :vk_workgroup_id         => :__spirv_BuiltInWorkgroupId,
    :vk_num_workgroups       => :__spirv_BuiltInNumWorkgroups,
    :vk_workgroup_size       => :__spirv_BuiltInWorkgroupSize,
)

for (jl_name, spirv_name) in _BUILTIN_3D
    gvar = "@$spirv_name"
    ir = """
        $gvar = external addrspace(7) global <3 x i32>
        define i32 @entry(i32 %dim) #0 {
            %vec = load <3 x i32>, ptr addrspace(7) $gvar, align 16
            %x = extractelement <3 x i32> %vec, i32 %dim
            ret i32 %x
        }
        attributes #0 = { alwaysinline }
    """

    # 3D version: vk_foo(dim) where dim is 1-based
    @eval @inline function $jl_name(dim::Integer=1)
        Base.llvmcall(($ir, "entry"), UInt32, Tuple{UInt32}, UInt32(dim - 1))
    end

    # Convenience: vk_foo_x(), vk_foo_y(), vk_foo_z()
    x_name = Symbol(jl_name, :_x)
    y_name = Symbol(jl_name, :_y)
    z_name = Symbol(jl_name, :_z)
    @eval @inline $x_name() = $jl_name(1)
    @eval @inline $y_name() = $jl_name(2)
    @eval @inline $z_name() = $jl_name(3)
end

# 1D scalar built-ins
@inline function vk_local_invocation_index()
    Base.llvmcall(("""
        @__spirv_BuiltInLocalInvocationIndex = external addrspace(7) global i32
        define i32 @entry() #0 {
            %val = load i32, ptr addrspace(7) @__spirv_BuiltInLocalInvocationIndex, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
end


# ── Shared memory ─────────────────────────────────────────────────────

"""Map Julia type to LLVM IR type string."""
function _llvm_type_str(::Type{T}) where T
    T === Float32 && return "float"
    T === Float64 && return "double"
    T === Float16 && return "half"
    (T === Int32 || T === UInt32) && return "i32"
    (T === Int64 || T === UInt64) && return "i64"
    (T === Int16 || T === UInt16) && return "i16"
    (T === Int8 || T === UInt8) && return "i8"
    error("Unsupported type $T for Vulkan shared memory")
end

struct VkLocalArray{T}
    ptr::Core.LLVMPtr{T, 3}
end

Base.@propagate_inbounds function Base.getindex(a::VkLocalArray{T}, i::Integer) where T
    unsafe_load(a.ptr, i)
end

Base.@propagate_inbounds function Base.setindex!(a::VkLocalArray{T}, v, i::Integer) where T
    unsafe_store!(a.ptr, convert(T, v), i)
    return a
end

@generated function vk_localmemory(::Val{Id}, ::Type{T}, ::Val{len}) where {Id, T, len}
    llvm_t = _llvm_type_str(T)
    align = max(sizeof(T), 4)
    safe_id = replace(string(Id), r"[^a-zA-Z0-9_]" => "_")
    gv_name = "__vk_shared_$(safe_id)"

    ir = """
        @$gv_name = internal addrspace(3) global [$len x $llvm_t] undef, align $align
        define ptr addrspace(3) @entry() #0 {
            %ptr = getelementptr [$len x $llvm_t], ptr addrspace(3) @$gv_name, i32 0, i32 0
            ret ptr addrspace(3) %ptr
        }
        attributes #0 = { alwaysinline }
    """
    return :(VkLocalArray{$T}(Base.llvmcall(($ir, "entry"), Core.LLVMPtr{$T, 3}, Tuple{})))
end

@generated function vk_localmemory(::Type{T}, ::Val{len}) where {T, len}
    llvm_t = _llvm_type_str(T)
    align = max(sizeof(T), 4)
    gv_name = "__vk_shared_$(len)x$(sizeof(T))b"

    ir = """
        @$gv_name = internal addrspace(3) global [$len x $llvm_t] undef, align $align
        define ptr addrspace(3) @entry() #0 {
            %ptr = getelementptr [$len x $llvm_t], ptr addrspace(3) @$gv_name, i32 0, i32 0
            ret ptr addrspace(3) %ptr
        }
        attributes #0 = { alwaysinline }
    """
    return :(VkLocalArray{$T}(Base.llvmcall(($ir, "entry"), Core.LLVMPtr{$T, 3}, Tuple{})))
end


# ── Math intrinsics ───────────────────────────────────────────────────
# Uses raw llvmcall with OpenCL C++ Itanium-mangled names (_Z3sinf etc.)
# to call GLSL.std.450 extended instructions via the LLVM SPIR-V backend.
#
# The backend translates these to OpExtInst with correct GLSL.std.450
# instruction numbers but mislabels the import as "OpenCL.std".
# spirv_fixup.jl remaps the import to "GLSL.std.450".
#
# For functions that have llvm.* equivalents (sin, cos, exp, etc.),
# the llvm.* path also works (backend maps directly to GLSL.std.450).
# device/math.jl already uses llvm.* for Float32/Float16 of those.
#
# Here we add overrides via OpenCL mangled names for:
#   - Functions NOT in device/math.jl (tan, asin, sinh, etc.) → all types
#   - ALL functions for Float64 (Vulkan shaderFloat64 extension)
#   - min/max via llvm.minnum/llvm.maxnum (Float32/Float16) + mangled (Float64)

# C++ Itanium mangling suffix for each type
_mangle_suffix(::Type{Float32}) = "f"
_mangle_suffix(::Type{Float64}) = "d"
_mangle_suffix(::Type{Float16}) = "Dh"

# LLVM IR type name for each type
_llvm_ir_type(::Type{Float32}) = "float"
_llvm_ir_type(::Type{Float64}) = "double"
_llvm_ir_type(::Type{Float16}) = "half"

# Register a mangled name as a known intrinsic (for GPUCompiler.isintrinsic)
function _register_intrinsic!(name::String)
    name in known_intrinsics || push!(known_intrinsics, name)
end

# ── Unary math via OpenCL mangled llvmcall ──
# (Julia function, OpenCL name, types to override)
const _VK_UNARY_MATH = [
    # Trig — NOT in device/math.jl → Float32, Float16, Float64
    (:(Base.tan),            "tan",     [Float32, Float16, Float64]),
    (:(Base.asin),           "asin",    [Float32, Float16, Float64]),
    (:(Base.acos),           "acos",    [Float32, Float16, Float64]),
    (:(Base.atan),           "atan",    [Float32, Float16, Float64]),
    # Hyperbolic — NOT in device/math.jl → all types
    (:(Base.sinh),           "sinh",    [Float32, Float16, Float64]),
    (:(Base.cosh),           "cosh",    [Float32, Float16, Float64]),
    (:(Base.tanh),           "tanh",    [Float32, Float16, Float64]),
    (:(Base.asinh),          "asinh",   [Float32, Float16, Float64]),
    (:(Base.acosh),          "acosh",   [Float32, Float16, Float64]),
    (:(Base.atanh),          "atanh",   [Float32, Float16, Float64]),
    # Misc — NOT in device/math.jl → all types
    (:(Base.sign),           "sign",    [Float32, Float16, Float64]),
    # Functions IN device/math.jl → Float64 only (Float32/Float16 already handled)
    (:(Base.sin),            "sin",     [Float64]),
    (:(Base.cos),            "cos",     [Float64]),
    (:(Base.exp),            "exp",     [Float64]),
    (:(Base.exp2),           "exp2",    [Float64]),
    (:(Base.log),            "log",     [Float64]),
    (:(Base.log2),           "log2",    [Float64]),
    (:(Base.log10),          "log10",   [Float64]),
    (:(Base.sqrt),           "sqrt",    [Float64]),
    (:(Base.abs),            "fabs",    [Float64]),
    (:(Base.floor),          "floor",   [Float64]),
    (:(Base.ceil),           "ceil",    [Float64]),
    (:(Base.trunc),          "trunc",   [Float64]),
    (:(Base.round),          "round",   [Float64]),
    (:(Base.Math.expm1),     "expm1",   [Float64]),
    (:(Base.Math.log1p),     "log1p",   [Float64]),
]

for (jlfun, ocl_name, types) in _VK_UNARY_MATH
    for T in types
        mangled = "_Z$(length(ocl_name))$(ocl_name)$(_mangle_suffix(T))"
        lt = _llvm_ir_type(T)
        _register_intrinsic!(mangled)
        @eval @vk_device_override @inline $jlfun(x::$T) = Base.llvmcall(
            ($("declare $lt @$mangled($lt)\ndefine $lt @entry($lt %0) {\n  %r = call $lt @$mangled($lt %0)\n  ret $lt %r\n}"), "entry"),
            $T, Tuple{$T}, x)
    end
end

# ── Binary min/max ──
# llvm.minnum/llvm.maxnum for Float32/Float16 (direct GLSL.std.450)
for (jlfun, llvmfun) in [
    (:(Base.min),  "llvm.minnum"),
    (:(Base.max),  "llvm.maxnum"),
]
    for (T, lt, suffix) in [(Float32, "float", "f32"), (Float16, "half", "f16")]
        llvm_name = "$(llvmfun).$(suffix)"
        @eval @vk_device_override @inline $jlfun(x::$T, y::$T) = Base.llvmcall(
            ($("declare $lt @$llvm_name($lt, $lt)\ndefine $lt @entry($lt %0, $lt %1) {\n  %r = call $lt @$llvm_name($lt %0, $lt %1)\n  ret $lt %r\n}"), "entry"),
            $T, Tuple{$T, $T}, x, y)
    end
end
# Float64 min/max via OpenCL mangled name
for (jlfun, ocl_name) in [(:(Base.min), "fmin"), (:(Base.max), "fmax")]
    mangled = "_Z$(length(ocl_name))$(ocl_name)dd"
    _register_intrinsic!(mangled)
    @eval @vk_device_override @inline $jlfun(x::Float64, y::Float64) = Base.llvmcall(
        ($("declare double @$mangled(double, double)\ndefine double @entry(double %0, double %1) {\n  %r = call double @$mangled(double %0, double %1)\n  ret double %r\n}"), "entry"),
        Float64, Tuple{Float64, Float64}, x, y)
end

# ── Binary math via OpenCL mangled llvmcall ──
const _VK_BINARY_MATH = [
    # (Julia function, OpenCL name, types)
    (:(Base.atan),      "atan2",    [Float32, Float16, Float64]),  # 2-arg atan
    (:(Base.copysign),  "copysign", [Float64]),                    # F32/F16 in device/math.jl
    (:(Base.hypot),     "hypot",    [Float64]),                    # F32/F16 in device/math.jl
    (:(Base.:(^)),      "pow",      [Float64]),                    # F32/F16 in device/math.jl
]

for (jlfun, ocl_name, types) in _VK_BINARY_MATH
    for T in types
        s = _mangle_suffix(T)
        mangled = "_Z$(length(ocl_name))$(ocl_name)$(s)$(s)"
        lt = _llvm_ir_type(T)
        _register_intrinsic!(mangled)
        @eval @vk_device_override @inline $jlfun(x::$T, y::$T) = Base.llvmcall(
            ($("declare $lt @$mangled($lt, $lt)\ndefine $lt @entry($lt %0, $lt %1) {\n  %r = call $lt @$mangled($lt %0, $lt %1)\n  ret $lt %r\n}"), "entry"),
            $T, Tuple{$T, $T}, x, y)
    end
end

# ── Ternary: fma for Float64 ──
# Float32/Float16 fma already in device/math.jl via llvm.fma
let mangled = "_Z3fmaddd", lt = "double"
    _register_intrinsic!(mangled)
    @eval @vk_device_override @inline Base.fma(a::Float64, b::Float64, c::Float64) = Base.llvmcall(
        ($("declare $lt @$mangled($lt, $lt, $lt)\ndefine $lt @entry($lt %0, $lt %1, $lt %2) {\n  %r = call $lt @$mangled($lt %0, $lt %1, $lt %2)\n  ret $lt %r\n}"), "entry"),
        Float64, Tuple{Float64, Float64, Float64}, a, b, c)
end
