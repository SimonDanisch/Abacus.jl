# Vulkan SPIR-V compilation pipeline via GPUCompiler
#
# Pipeline: Julia function → GPUCompiler → LLVM IR → wrap entry point
# (push constants) → LLVM SPIR-V backend → SPIR-V fixup → valid Vulkan SPIR-V

struct VkCompilerParams <: GPUCompiler.AbstractCompilerParams
    workgroup_size::NTuple{3,Int}
end
VkCompilerParams() = VkCompilerParams((64, 1, 1))
const VkCompilerConfig = GPUCompiler.CompilerConfig{GPUCompiler.SPIRVCompilerTarget, VkCompilerParams}
const VkCompilerJob = GPUCompiler.CompilerJob{GPUCompiler.SPIRVCompilerTarget, VkCompilerParams}

GPUCompiler.runtime_module(::VkCompilerJob) = Abacus

GPUCompiler.method_table(::VkCompilerJob) = vk_method_table

# Vulkan doesn't use GPUCompiler's kernel state mechanism
GPUCompiler.kernel_state_type(::VkCompilerJob) = Nothing

# Allow SPIR-V intrinsic function calls to pass IR validation.
# llvm.spv.* intrinsics are lowered directly to SPIR-V opcodes by llc.
# NOTE: __spirv_* (SPIR-V friendly IR) naming convention does NOT work with
# the Vulkan target — only with OpenCL SPIR-V (spirv64-unknown-unknown).
GPUCompiler.isintrinsic(::VkCompilerJob, fn::String) =
    startswith(fn, "llvm.spv.") || fn in known_intrinsics

# Compiler cache: keyed by (MethodInstance objectid, world, config) via GPUCompiler.cached_compilation
const _vk_compiler_cache = Dict{Any, Any}()

# Kernel instance cache: compiled pipeline + metadata, keyed by (linked_obj, f, tt)
const _vk_kernel_cache = Dict{UInt, Any}()

const _vk_function_lock = ReentrantLock()

# cache of compiler configurations
const _vk_compiler_configs = Dict{UInt, VkCompilerConfig}()

function vk_compiler_config(; workgroup_size::NTuple{3,Int} = (64, 1, 1), kwargs...)
    h = hash((workgroup_size, kwargs))
    config = get(_vk_compiler_configs, h, nothing)
    if config !== nothing
        return config
    end
    config = _vk_compiler_config(; workgroup_size, kwargs...)
    _vk_compiler_configs[h] = config
    return config
end

@noinline function _vk_compiler_config(;
        kernel = true,
        name = nothing,
        always_inline = true,
        workgroup_size::NTuple{3,Int} = (64, 1, 1),
        kwargs...)
    target = GPUCompiler.SPIRVCompilerTarget(;
        backend = :vulkan,
        validate = false,   # we validate after fixup
        supports_fp64 = true,
    )
    params = VkCompilerParams(workgroup_size)
    GPUCompiler.CompilerConfig(target, params; kernel, name, always_inline)
end

"""
    vk_compile_spirv(f, tt; workgroup_size=(64,1,1)) -> (spirv_bytes, entry_name, push_size)

Compile a Julia function to Vulkan SPIR-V binary (uncached).
Returns the SPIR-V bytes, entry point name, and push constant size in bytes.

For cached compilation, use `vkfunction(f, tt)` instead.
"""
function vk_compile_spirv(@nospecialize(f), @nospecialize(tt);
                          workgroup_size::NTuple{3,Int} = (64, 1, 1))
    config = vk_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    job = GPUCompiler.CompilerJob(source, config)
    compiled = vkcompile(job)
    linked = vklink(job, compiled)
    return (; spirv_bytes = linked.spirv_bytes,
              entry_name = linked.entry_name,
              push_size = linked.push_size,
              arg_buffer_size = linked.arg_buffer_size)
end

# --- Cached compilation pipeline (mirrors AMDGPU's hipfunction/hipcompile/hiplink) ---

"""Compiled Vulkan kernel: holds the pipeline and metadata needed for dispatch."""
struct VkKernel{F, TT}
    fun::F
    pipeline::VkComputePipeline
    entry_name::String
    push_size::Int
    arg_buffer_size::Int
end

"""
    vkfunction(f, tt; workgroup_size=(64,1,1)) -> VkKernel

Compile and cache a Julia function as a Vulkan compute kernel.
On repeated calls with the same function signature, returns the cached kernel
without recompilation. Mirrors AMDGPU's `hipfunction`.
"""
function vkfunction(@nospecialize(f::F), @nospecialize(tt::Type{TT} = Tuple{});
                    workgroup_size::NTuple{3,Int} = (64, 1, 1)) where {F <: Core.Function, TT}
    Base.@lock _vk_function_lock begin
        config = vk_compiler_config(; workgroup_size)
        source = GPUCompiler.methodinstance(F, tt)

        # Level 1: GPUCompiler compilation cache (keyed by MethodInstance + world + config)
        linked = GPUCompiler.cached_compilation(
            _vk_compiler_cache, source, config, vkcompile, vklink)

        # Level 2: kernel instance cache (keyed by linked object + function + types)
        h = hash(linked, hash(f, hash(tt)))
        kernel = get(_vk_kernel_cache, h, nothing)
        if kernel !== nothing
            return kernel::VkKernel{F, TT}
        end
        k = VkKernel{F, TT}(f, linked.pipeline, linked.entry_name, linked.push_size, linked.arg_buffer_size)
        _vk_kernel_cache[h] = k
        return k
    end
end

"""Compiler function for `GPUCompiler.cached_compilation`.

Compiles a Julia function to Vulkan SPIR-V binary via the full pipeline:
GPUCompiler → LLVM IR → push constant wrapping → optimization → SPIR-V backend → fixup.
"""
function vkcompile(@nospecialize(job::GPUCompiler.CompilerJob))
    GPUCompiler.JuliaContext() do ctx
        # Compile to LLVM IR
        mod, meta = GPUCompiler.compile(:llvm, job)
        entry = meta.entry

        # Read workgroup_size from compiler params
        wgs = job.config.params.workgroup_size

        # Wrap the entry point: create void() wrapper that loads from push constants
        push_info = _wrap_entry_for_vulkan!(mod, entry; workgroup_size=wgs)

        # Run LLVM AlwaysInliner to inline the original entry into the wrapper
        # (llc doesn't run optimization passes, so alwaysinline wouldn't take effect)
        LLVM.run!(LLVM.AlwaysInlinerPass(), mod)

        # Run SROA + InstCombine to eliminate allocas from byval struct handling.
        # Without this, the alloca+store pattern generates @store_deref/@load_deref
        # in SPIR-V NIR which RADV's ACO compiler can't handle.
        LLVM.run!(LLVM.SROAPass(), mod)
        LLVM.run!(LLVM.InstCombinePass(), mod)

        # Aggressive scalar optimization: SCCP propagates constants through
        # conditionals, GVN eliminates redundant computations, then SROA
        # can decompose allocas that had variable-index GEPs (now resolved
        # to constants). This is critical for Broadcasted types with Tuple
        # args where the Extruded keeps/defaults mechanism generates
        # variable-indexed GEPs that the SPIR-V backend can't legalize.
        LLVM.run!(LLVM.SCCPPass(), mod)
        LLVM.run!(LLVM.GVNPass(), mod)
        LLVM.run!(LLVM.SROAPass(), mod)
        LLVM.run!(LLVM.InstCombinePass(), mod)

        # Lift byte-offset GEPs on struct allocas to typed struct GEPs.
        # SROA rewrites struct accesses as `gep i8, ptr %alloca, <offset>` which
        # crashes the LLVM SPIR-V backend's "legalize bitcast" pass. Converting
        # to `gep %struct, ptr %alloca, 0, <member_path...>` gives the backend
        # the type information it needs for OpAccessChain generation.
        _lift_byte_geps_on_allocas!(mod)

        # After lifting GEPs, run SROA again — the typed GEPs may allow full
        # alloca elimination for simple cases.
        LLVM.run!(LLVM.SROAPass(), mod)
        LLVM.run!(LLVM.InstCombinePass(), mod)

        # Remove llvm.trap and replace unreachable blocks with branches to exit.
        # GPUCompiler's lower_throw! converts `throw(nothing)` into:
        #   gpu_report_exception() + gpu_signal_exception() + llvm.trap() + unreachable
        # These unreachable terminators create dead-end blocks that the LLVM SPIR-V
        # backend cannot structure into valid Vulkan SPIR-V.
        # This mirrors Metal's approach (replace_unreachable!) and uses GPUCompiler's
        # own rm_trap!. Must happen BEFORE SimplifyCFG/StructurizeCFG.
        GPUCompiler.rm_trap!(mod)
        _replace_unreachable!(mod)
        _replace_freeze!(mod)
        _strip_noreturn!(mod)
        _strip_assume!(mod)

        # SimplifyCFG merges nested conditionals that both branch to the same
        # exit block. Without this, the LLVM SPIR-V backend produces two
        # OpSelectionMerge instructions targeting the same block, which is
        # invalid in Vulkan SPIR-V ("Block is already a merge block").
        LLVM.run!(LLVM.SimplifyCFGPass(), mod)

        # Isolate shared merge targets BEFORE StructurizeCFG.
        # When a block is the target of multiple conditional branches (e.g.,
        # an error path reachable from both a loop preheader and the loop body),
        # StructurizeCFG can't structurize the loop correctly — it replaces the
        # back-edge condition with `true`, causing the loop to exit after one
        # iteration. Inserting trampolines gives each construct its own unique
        # exit target, like clspv's FixupStructuredCFGPass.
        _fixup_structured_cfg!(mod)

        # Structurize CFG for Vulkan SPIR-V compliance.
        # Vulkan requires structured control flow (no irreducible loops,
        # single-entry/single-exit regions). Complex Julia code (e.g.
        # CartesianIndices, bounds checking) generates CFGs that violate
        # this. The same pass sequence is used by AMD's GPU compiler.
        LLVM.run!(LLVM.LowerSwitchPass(), mod)
        LLVM.run!(LLVM.UnifyFunctionExitNodesPass(), mod)
        LLVM.run!(LLVM.FixIrreduciblePass(), mod)
        LLVM.run!(LLVM.StructurizeCFGPass(), mod)

        # StructurizeCFG introduces reg2mem (alloca+load+store) patterns.
        # InstCombine cleans up bitcasts that crash llc's "SPIRV legalize bitcast pass".
        # Do NOT run SimplifyCFG here — it merges blocks and destroys the
        # structured control flow, causing "Block is already a merge block" errors.
        LLVM.run!(LLVM.InstCombinePass(), mod)

        # Lift any remaining byte-offset GEPs on struct allocas (may have been
        # introduced by StructurizeCFG's reg2mem or InstCombine).
        _lift_byte_geps_on_allocas!(mod)

        # Strip debug info (SPIR-V tools don't handle Julia's debug info)
        LLVM.strip_debuginfo!(mod)

        # LLVM IR-level fixups for Vulkan SPIR-V compliance
        _prepare_module_for_vulkan!(mod, push_info.wrapper_name)

        # Generate SPIR-V via LLVM backend
        raw_spv = _emit_spirv(mod, job.config.target)

        # Apply Vulkan fixups
        fixed_spv = spirv_fixup(raw_spv, push_info.member_layout;
                               entry_name = push_info.wrapper_name)

        return (; spirv_bytes = fixed_spv,
                  entry_name = push_info.wrapper_name,
                  push_size = push_info.push_size,
                  arg_buffer_size = push_info.arg_buffer_size)
    end
end

"""Linker function for `GPUCompiler.cached_compilation`.

Takes compiled SPIR-V and creates the Vulkan compute pipeline.
"""
function vklink(@nospecialize(job::GPUCompiler.CompilerJob), compiled)
    pipeline = get_pipeline(compiled.spirv_bytes, compiled.entry_name,
                            compiled.push_size)
    return (; compiled.spirv_bytes, compiled.entry_name, compiled.push_size,
              compiled.arg_buffer_size, pipeline)
end

"""
Push constant layout info returned by `_wrap_entry_for_vulkan!`.
"""
struct PushConstantInfo
    wrapper_name::String
    push_size::Int           # always 8 (BDA i64)
    arg_buffer_size::Int     # total size of the argument data buffer
    # Vector of (offset, size) for each push constant struct member (just the BDA i64)
    member_layout::Vector{Pair{Int,Int}}
end

"""
    _wrap_entry_for_vulkan!(mod, entry) -> PushConstantInfo

Transform the LLVM module so the entry point is a void() function that loads
kernel arguments from a BDA (Buffer Device Address) argument buffer.

The push constant is a single i64 containing the BDA of the argument buffer.
The wrapper loads each kernel argument from the buffer via byte-offset GEPs
on `ptr addrspace(1)` (PhysicalStorageBuffer).

- Original entry: internal + alwaysinline (becomes a callee)
- Wrapper: void(), hlsl.shader=compute, hlsl.numthreads=64,1,1
- Push constant struct: `{i64}` — just the BDA pointer
- Argument data: loaded from PhysicalStorageBuffer via byte-offset GEPs
"""
function _wrap_entry_for_vulkan!(mod::LLVM.Module, entry::LLVM.Function;
                                 workgroup_size::NTuple{3,Int} = (64, 1, 1))
    entry_name = LLVM.name(entry)
    ft = LLVM.function_type(entry)
    param_types = LLVM.parameters(ft)

    # If no parameters, just ensure the hlsl attributes are present
    if isempty(param_types)
        return PushConstantInfo(entry_name, 0, 0, Pair{Int,Int}[])
    end

    # Remove hlsl attributes from original entry
    for attr in collect(LLVM.function_attributes(entry))
        k = LLVM.kind(attr)
        if k == LLVM.kind(LLVM.StringAttribute("hlsl.shader", "")) ||
           k == LLVM.kind(LLVM.StringAttribute("hlsl.numthreads", ""))
            delete!(LLVM.function_attributes(entry), attr)
        end
    end
    LLVM.linkage!(entry, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(entry), LLVM.EnumAttribute("alwaysinline"))

    # Detect byval attributes to find struct-by-reference parameters
    byval_kind = LLVM.kind(LLVM.TypeAttribute("byval", LLVM.Int32Type()))

    # Build the full argument struct type (for computing offsets and sizes)
    # - Pointer params without byval → i64 BDA (convert to pointer)
    # - Pointer params with byval(struct_type) → flatten struct fields into scalars
    # - Non-pointer scalar params → as-is
    T_i64 = LLVM.Int64Type()
    T_i8 = LLVM.Int8Type()
    arg_fields = LLVM.LLVMType[]
    # param_info entries:
    #   (:ptr, start_idx)                    — BDA pointer (i64 → inttoptr)
    #   (:scalar, start_idx)                 — scalar (load directly)
    #   (:byval, start_idx, byval_type)      — byval struct (alloca + store fields + pass ptr)
    param_info = []

    for (i, pt) in enumerate(param_types)
        start_idx = length(arg_fields)
        is_ptr = pt isa LLVM.PointerType || occursin("ptr", string(pt))

        if is_ptr
            # Check if this pointer has a byval attribute
            byval_type = nothing
            for attr in collect(LLVM.parameter_attributes(entry, i))
                if attr isa LLVM.TypeAttribute && LLVM.kind(attr) == byval_kind
                    byval_type = LLVM.value(attr)
                    break
                end
            end

            if byval_type !== nothing
                # Byval struct: flatten all leaf fields into arg buffer scalars
                _flatten_struct_fields!(arg_fields, byval_type)
                push!(param_info, (:byval, start_idx, byval_type))
            else
                # Regular pointer (BDA): store as i64
                push!(arg_fields, T_i64)
                push!(param_info, (:ptr, start_idx))
            end
        else
            push!(arg_fields, pt)
            push!(param_info, (:scalar, start_idx))
        end
    end

    # Compute argument buffer layout (offsets and sizes for each field)
    arg_member_layout = Pair{Int,Int}[]
    offset = 0
    for field in arg_fields
        sz = _llvm_type_size(field)
        align = sz  # natural alignment
        offset = (offset + align - 1) & ~(align - 1)
        push!(arg_member_layout, offset => sz)
        offset += sz
    end
    arg_buffer_size = offset

    # Push constant struct: just {i64} for the BDA pointer
    T_push = LLVM.StructType([T_i64])
    push_size = 8
    # Member layout for the push constant struct (single i64 at offset 0)
    push_member_layout = Pair{Int,Int}[0 => 8]

    # Create global in addrspace 2 (maps to UniformConstant; spirv_fixup converts to PushConstant)
    gv = LLVM.GlobalVariable(mod, T_push, "__push_constants", 2)
    LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)

    # Create wrapper function
    T_void = LLVM.VoidType()
    wrapper_ft = LLVM.FunctionType(T_void)
    wrapper_name = entry_name * "_wrapper"
    wrapper = LLVM.Function(mod, wrapper_name, wrapper_ft)
    wgs_str = join(workgroup_size, ",")
    push!(LLVM.function_attributes(wrapper), LLVM.StringAttribute("hlsl.shader", "compute"))
    push!(LLVM.function_attributes(wrapper), LLVM.StringAttribute("hlsl.numthreads", wgs_str))

    bb = LLVM.BasicBlock(wrapper, "entry")
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, bb)

        # Load BDA from push constants
        push_struct = LLVM.load!(builder, T_push, gv)
        bda_int = LLVM.extract_value!(builder, push_struct, 0)

        # Load each argument from the BDA buffer.
        # Use inttoptr(bda + offset) for each field — byte-offset GEPs on
        # ptr addrspace(1) produce OpAccessChain with type mismatches in SPIR-V.
        T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)
        args = LLVM.Value[]
        for (i, pt) in enumerate(param_types)
            info = param_info[i]
            if info[1] == :ptr
                field_offset = arg_member_layout[info[2] + 1].first
                addr = LLVM.add!(builder, bda_int, LLVM.ConstantInt(T_i64, field_offset), "bda_off_$(i)")
                field_ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1)
                bda_val = LLVM.load!(builder, T_i64, field_ptr)
                LLVM.alignment!(bda_val, 8)
                ptr_val = LLVM.inttoptr!(builder, bda_val, pt)
                push!(args, ptr_val)
            elseif info[1] == :scalar
                field_offset = arg_member_layout[info[2] + 1].first
                addr = LLVM.add!(builder, bda_int, LLVM.ConstantInt(T_i64, field_offset), "bda_off_$(i)")
                field_ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1)
                val = LLVM.load!(builder, pt, field_ptr)
                align = max(4, _llvm_type_size(pt))
                LLVM.alignment!(val, align)
                push!(args, val)
            elseif info[1] == :byval
                # Byval struct: alloca on stack, load flattened fields from
                # BDA buffer and store into alloca, pass pointer.
                byval_type = info[3]
                alloca = LLVM.alloca!(builder, byval_type, "byval_arg_$i")
                flat_idx = Ref(info[2])
                _store_flattened_fields_from_bda!(builder, bda_int, arg_member_layout, alloca, byval_type, flat_idx)
                push!(args, alloca)
            end
        end

        LLVM.call!(builder, ft, entry, args)
        LLVM.ret!(builder)
    end

    return PushConstantInfo(wrapper_name, push_size, arg_buffer_size, push_member_layout)
end

"""Recursively flatten a struct type's leaf fields into `push_fields`."""
function _flatten_struct_fields!(push_fields::Vector{LLVM.LLVMType}, t::LLVM.LLVMType)
    if t isa LLVM.StructType
        for m in LLVM.elements(t)
            _flatten_struct_fields!(push_fields, m)
        end
    elseif t isa LLVM.ArrayType
        et = LLVM.eltype(t)
        for _ in 1:length(t)
            _flatten_struct_fields!(push_fields, et)
        end
    else
        push!(push_fields, t)
    end
end

"""Store flattened scalar push constant values into a stack-allocated struct.

Recursively walks `struct_type`, loading leaf scalars from the push constant
global at consecutive flat indices, and storing them into the corresponding
GEP positions of `dest_ptr`."""
function _store_flattened_fields!(builder, T_push, gv, dest_ptr, struct_type, flat_idx::Ref{Int})
    for (j, member_t) in enumerate(LLVM.elements(struct_type))
        member_ptr = LLVM.struct_gep!(builder, struct_type, dest_ptr, j - 1)
        if member_t isa LLVM.StructType
            _store_flattened_fields!(builder, T_push, gv, member_ptr, member_t, flat_idx)
        elseif member_t isa LLVM.ArrayType
            et = LLVM.eltype(member_t)
            for k in 0:(length(member_t) - 1)
                elem_ptr = LLVM.struct_gep!(builder, member_t, member_ptr, k)
                if et isa LLVM.StructType
                    _store_flattened_fields!(builder, T_push, gv, elem_ptr, et, flat_idx)
                else
                    fp = LLVM.struct_gep!(builder, T_push, gv, flat_idx[])
                    sv = LLVM.load!(builder, et, fp)
                    LLVM.store!(builder, sv, elem_ptr)
                    flat_idx[] += 1
                end
            end
        else
            fp = LLVM.struct_gep!(builder, T_push, gv, flat_idx[])
            sv = LLVM.load!(builder, member_t, fp)
            LLVM.store!(builder, sv, member_ptr)
            flat_idx[] += 1
        end
    end
end

"""Store flattened scalar values from an extractvalue'd push constant struct into a stack alloca.

Like `_store_flattened_fields!` but uses `extract_value!` from a loaded struct value
instead of `struct_gep!` from a global pointer. This avoids GEP constant expressions
that the LLVM SPIR-V backend converts to raw byte arithmetic."""
function _store_flattened_fields_ev!(builder, push_struct, T_push, dest_ptr, typ, flat_idx::Ref{Int})
    if typ isa LLVM.StructType
        for (j, member_t) in enumerate(LLVM.elements(typ))
            member_ptr = LLVM.struct_gep!(builder, typ, dest_ptr, j - 1)
            _store_flattened_fields_ev!(builder, push_struct, T_push, member_ptr, member_t, flat_idx)
        end
    elseif typ isa LLVM.ArrayType
        et = LLVM.eltype(typ)
        for k in 0:(length(typ) - 1)
            elem_ptr = LLVM.struct_gep!(builder, typ, dest_ptr, k)
            _store_flattened_fields_ev!(builder, push_struct, T_push, elem_ptr, et, flat_idx)
        end
    else
        # Leaf scalar type
        sv = LLVM.extract_value!(builder, push_struct, flat_idx[])
        LLVM.store!(builder, sv, dest_ptr)
        flat_idx[] += 1
    end
end

"""Store flattened scalar values from a BDA buffer into a stack alloca.

Like `_store_flattened_fields_ev!` but loads from a BDA integer (addrspace 1)
via inttoptr(bda + offset) instead of extractvalue from a push constant struct."""
function _store_flattened_fields_from_bda!(builder, bda_int, member_layout, dest_ptr, typ, flat_idx::Ref{Int})
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)
    if typ isa LLVM.StructType
        for (j, member_t) in enumerate(LLVM.elements(typ))
            member_ptr = LLVM.struct_gep!(builder, typ, dest_ptr, j - 1)
            _store_flattened_fields_from_bda!(builder, bda_int, member_layout, member_ptr, member_t, flat_idx)
        end
    elseif typ isa LLVM.ArrayType
        et = LLVM.eltype(typ)
        for k in 0:(length(typ) - 1)
            elem_ptr = LLVM.struct_gep!(builder, typ, dest_ptr, k)
            _store_flattened_fields_from_bda!(builder, bda_int, member_layout, elem_ptr, et, flat_idx)
        end
    else
        # Leaf scalar type: load from BDA buffer at the field's byte offset
        field_offset = member_layout[flat_idx[] + 1].first
        addr = LLVM.add!(builder, bda_int, LLVM.ConstantInt(T_i64, field_offset), "bda_off_$(flat_idx[])")
        field_ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1)
        sv = LLVM.load!(builder, typ, field_ptr)
        align = max(4, _llvm_type_size(typ))
        LLVM.alignment!(sv, align)
        LLVM.store!(builder, sv, dest_ptr)
        flat_idx[] += 1
    end
end

"""Get the size in bytes of an LLVM type."""
function _llvm_type_size(t::LLVM.LLVMType)
    if t isa LLVM.IntegerType
        return div(LLVM.width(t) + 7, 8)
    elseif t == LLVM.FloatType()
        return 4
    elseif t == LLVM.DoubleType()
        return 8
    elseif t == LLVM.HalfType()
        return 2
    elseif t isa LLVM.ArrayType
        return length(t) * _llvm_type_size(LLVM.eltype(t))
    elseif t isa LLVM.StructType
        total = 0
        for m in LLVM.elements(t)
            total += _llvm_type_size(m)
        end
        return total
    else
        # Fallback: assume 8 bytes (pointer-sized)
        return 8
    end
end

# Return the size in bytes of the largest scalar component in `ty`.
function _scalar_size(t::LLVM.LLVMType)
    if t isa LLVM.IntegerType
        return div(LLVM.width(t) + 7, 8)
    elseif t == LLVM.FloatType()
        return 4
    elseif t == LLVM.DoubleType()
        return 8
    elseif t == LLVM.HalfType()
        return 2
    elseif t isa LLVM.StructType
        sz = 0
        for m in LLVM.elements(t)
            sz = max(sz, _scalar_size(m))
        end
        return sz
    elseif t isa LLVM.ArrayType || t isa LLVM.VectorType
        return _scalar_size(LLVM.eltype(t))
    elseif t isa LLVM.PointerType
        return 8
    else
        return 4
    end
end

"""
    _lift_byte_geps_on_allocas!(mod)

Convert byte-offset GEPs on struct allocas to properly typed struct GEPs.

After SROA partially decomposes struct allocas, it rewrites accesses as:
    %p = getelementptr i8, ptr %alloca, <byte_offset>
The LLVM SPIR-V backend's "legalize bitcast" pass crashes on these because
it can't map byte offsets back to struct member access chains (OpAccessChain).

This pass converts them to:
    %p = getelementptr %alloca_type, ptr %alloca, 0, <member_path...>
giving the SPIR-V backend the type information it needs.

For GEPs used as the base of another typed GEP (e.g., variable-indexed array access),
the navigation stops at the aggregate member boundary. For GEPs used by loads/stores,
it navigates to the scalar leaf.
"""
function _lift_byte_geps_on_allocas!(mod::LLVM.Module)
    dl = LLVM.datalayout(mod)

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        # Collect all allocas with aggregate types (struct or array)
        allocas = Pair{LLVM.Instruction, LLVM.LLVMType}[]
        for inst in LLVM.instructions(first(LLVM.blocks(f)))
            inst isa LLVM.AllocaInst || continue
            at = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(inst))
            (at isa LLVM.StructType || at isa LLVM.ArrayType) || continue
            push!(allocas, inst => at)
        end
        isempty(allocas) && continue

        to_erase = LLVM.Instruction[]

        for (alloca_inst, alloca_type) in allocas
            # Find byte-offset GEPs that use this alloca as base:
            #   %p = getelementptr [inbounds] i8, ptr %alloca, i64 <const>
            for use in LLVM.uses(alloca_inst)
                user = LLVM.user(use)
                user isa LLVM.GetElementPtrInst || continue

                # Check if this is a byte-offset GEP (source element type = i8)
                src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
                src_ty == LLVM.Int8Type() || continue

                # Get the byte offset (must be a constant)
                ops = LLVM.operands(user)
                length(ops) == 2 || continue  # expecting: ptr, offset
                offset_val = ops[2]
                offset_val isa LLVM.ConstantInt || continue
                byte_offset = convert(Int, offset_val)

                # Classify users: typed-GEP users need aggregate depth,
                # load/store users need leaf depth.
                has_typed_gep_users = false
                has_leaf_users = false
                for u in LLVM.uses(user)
                    usr = LLVM.user(u)
                    if usr isa LLVM.GetElementPtrInst
                        u_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(usr))
                        if u_src != LLVM.Int8Type()
                            has_typed_gep_users = true
                        else
                            has_leaf_users = true
                        end
                    else
                        has_leaf_users = true
                    end
                end


                is_inbounds = LLVM.API.LLVMIsInBounds(user)

                # Helper to build a typed GEP from index path
                function _build_typed_gep(builder, indices, name)
                    idx_vals = LLVM.Value[LLVM.ConstantInt(LLVM.Int32Type(), 0)]
                    for idx in indices
                        push!(idx_vals, LLVM.ConstantInt(LLVM.Int32Type(), idx))
                    end
                    gep = LLVM.gep!(builder, alloca_type, alloca_inst, idx_vals, name)
                    LLVM.API.LLVMSetIsInBounds(gep, is_inbounds)
                    return gep
                end

                if has_typed_gep_users && has_leaf_users
                    # DUAL-USE: the same byte-offset GEP is used by both:
                    # 1. Typed GEPs (e.g., variable-indexed array access)
                    # 2. Loads/stores (need scalar leaf pointer)
                    #
                    # For typed-GEP users: MERGE the aggregate path + user's
                    # indices into a single GEP from the alloca. This avoids
                    # an intermediate aggregate pointer that the SPIR-V backend
                    # can't type-match with the downstream GEP's source type.
                    agg_indices = _byte_offset_to_gep_indices(alloca_type, byte_offset, dl, true)
                    leaf_indices = _byte_offset_to_gep_indices(alloca_type, byte_offset, dl, false)
                    (agg_indices === nothing || leaf_indices === nothing) && continue

                    # Collect typed-GEP users FIRST — modifying operands
                    # invalidates the use iterator.
                    typed_gep_users = LLVM.Instruction[]
                    for u in LLVM.uses(user)
                        usr = LLVM.user(u)
                        if usr isa LLVM.GetElementPtrInst
                            u_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(usr))
                            if u_src != LLVM.Int8Type()
                                push!(typed_gep_users, usr)
                            end
                        end
                    end

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        # For each typed-GEP user, create a merged GEP that
                        # goes directly from the alloca through the aggregate
                        # path and then the user's own indices.
                        for usr in typed_gep_users
                            LLVM.position!(builder, usr)
                            usr_ops = LLVM.operands(usr)
                            # Build merged index path: [0, agg_path..., user_indices...]
                            # The user GEP has form: gep elem_type, ptr base, idx0, idx1, ...
                            # where idx0 is the "array index" (often variable) and
                            # idx1... are struct member indices.
                            merged_vals = LLVM.Value[LLVM.ConstantInt(LLVM.Int32Type(), 0)]
                            for idx in agg_indices
                                push!(merged_vals, LLVM.ConstantInt(LLVM.Int32Type(), idx))
                            end
                            # Append the user GEP's indices (skip operand 0 = base pointer)
                            for oi in 2:length(usr_ops)
                                push!(merged_vals, usr_ops[oi])
                            end
                            merged_gep = LLVM.gep!(builder, alloca_type, alloca_inst,
                                                    merged_vals, "typed_gep_merged")
                            LLVM.API.LLVMSetIsInBounds(merged_gep, is_inbounds)
                            LLVM.replace_uses!(usr, merged_gep)
                            push!(to_erase, usr)
                        end

                        # Redirect all remaining uses (loads/stores) to leaf GEP
                        LLVM.position!(builder, user)
                        leaf_gep = _build_typed_gep(builder, leaf_indices, "typed_gep_leaf")
                        LLVM.replace_uses!(user, leaf_gep)
                    end
                else
                    # Single-use case: pick the appropriate depth
                    stop_at_agg = has_typed_gep_users
                    indices = _byte_offset_to_gep_indices(alloca_type, byte_offset, dl,
                                                           stop_at_agg)
                    indices === nothing && continue

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, user)
                        new_gep = _build_typed_gep(builder, indices, "typed_gep")
                        LLVM.replace_uses!(user, new_gep)
                    end
                end
                push!(to_erase, user)
            end

            # Second pattern: element-typed flat GEPs on array allocas.
            # LLVM generates: gep float, ptr %[3xfloat]_alloca, i64 %idx
            # SPIR-V needs:   gep [3 x float], ptr %alloca, i64 0, %idx
            # This happens when Julia/LLVM flattens array indexing to pointer
            # arithmetic with the element type as source.
            if alloca_type isa LLVM.ArrayType
                elem_type = LLVM.eltype(alloca_type)
                for use in LLVM.uses(alloca_inst)
                    user = LLVM.user(use)
                    user isa LLVM.GetElementPtrInst || continue
                    user in to_erase && continue

                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
                    # Match: source type is the array's element type (not i8, not the array type)
                    src_ty == elem_type || continue
                    src_ty == alloca_type && continue  # already correct

                    ops = LLVM.operands(user)
                    length(ops) == 2 || continue  # gep elem, ptr, idx

                    is_inbounds = LLVM.API.LLVMIsInBounds(user)
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, user)
                        idx_vals = LLVM.Value[
                            LLVM.ConstantInt(LLVM.Int64Type(), 0),
                            ops[2]  # the original index (may be variable)
                        ]
                        new_gep = LLVM.gep!(builder, alloca_type, alloca_inst,
                                            idx_vals, "typed_arr_gep")
                        LLVM.API.LLVMSetIsInBounds(new_gep, is_inbounds)
                        LLVM.replace_uses!(user, new_gep)
                    end
                    push!(to_erase, user)
                end

                # Third pattern: direct store/load of element type to array alloca.
                # LLVM optimizes `gep i8, ptr %alloca, 0` away, leaving:
                #   store float %val, ptr %byval_arg   (where alloca is [3 x float])
                # The SPIR-V backend can't reconcile the float-typed access with
                # the [3 x float] alloca when variable-indexed GEPs also exist.
                # Fix: insert gep [3 x float], ptr %alloca, 0, 0 before each such use.
                direct_users = LLVM.Instruction[]
                for use in LLVM.uses(alloca_inst)
                    usr = LLVM.user(use)
                    usr in to_erase && continue
                    if usr isa LLVM.StoreInst || usr isa LLVM.LoadInst
                        push!(direct_users, usr)
                    end
                end
                if !isempty(direct_users)
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        # Insert GEP before first use
                        LLVM.position!(builder, first(direct_users))
                        idx_vals = LLVM.Value[
                            LLVM.ConstantInt(LLVM.Int64Type(), 0),
                            LLVM.ConstantInt(LLVM.Int64Type(), 0),
                        ]
                        elem0_gep = LLVM.gep!(builder, alloca_type, alloca_inst,
                                               idx_vals, "arr_elem0")
                        # Redirect all direct load/store uses from alloca to elem0_gep
                        for usr in direct_users
                            if usr isa LLVM.StoreInst
                                # store val, ptr %alloca → store val, ptr %elem0_gep
                                # The pointer is operand 2 (value=op1, ptr=op2)
                                LLVM.API.LLVMSetOperand(usr, 1, elem0_gep)
                            elseif usr isa LLVM.LoadInst
                                LLVM.API.LLVMSetOperand(usr, 0, elem0_gep)
                            end
                        end
                    end
                end
            end
        end

        for inst in to_erase
            LLVM.erase!(inst)
        end

        # Fourth pattern: Flat GEP chains on alloca-derived typed GEPs.
        # The SPIR-V backend can't do pointer arithmetic on struct/array pointers:
        #   %a = gep <type>, ptr %alloca, <indices...>, %array_idx
        #   %b = gep <elem_type>, ptr %a, %offset, <member_indices...>
        # Merge into: gep <type>, ptr %alloca, <indices...>, (%array_idx + %offset), <member_indices...>
        # Same as Pattern B in _fix_shared_geps! but for Private memory.
        # Handles both simple (gep [N x T], ptr, 0, %idx) and nested struct cases
        # (gep {struct}, ptr, 0, 0, 1, %idx) where the last index goes into an array.
        changed = true
        while changed
            changed = false
            alloca_set = Set{LLVM.Instruction}(first(p) for p in allocas)
            for bb in LLVM.blocks(f)
                chain_to_fix = Tuple{LLVM.GetElementPtrInst, LLVM.GetElementPtrInst}[]
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.GetElementPtrInst || continue
                    ops = LLVM.operands(inst)
                    length(ops) >= 3 || continue  # need: ptr, first_idx, member_idx...
                    base = ops[1]
                    base isa LLVM.GetElementPtrInst || continue
                    # Check that the base GEP targets one of our allocas
                    base_ops = LLVM.operands(base)
                    base_ptr = base_ops[1]
                    base_ptr in alloca_set || continue
                    # Walk base GEP indices through the type hierarchy to find
                    # what type the LAST index indexes into. Must be an ArrayType.
                    base_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(base))
                    n_base_idx = length(base_ops) - 1  # number of index operands
                    n_base_idx >= 2 || continue  # need at least ptr-level + one real index
                    # Walk indices (skip first = ptr-level index)
                    cur_type = base_src
                    last_indexed_type = nothing
                    valid = true
                    for idx_i in 2:n_base_idx
                        last_indexed_type = cur_type
                        if cur_type isa LLVM.StructType
                            idx_op = base_ops[idx_i + 1]  # +1 because ops[1] is ptr
                            idx_op isa LLVM.ConstantInt || (valid = false; break)
                            member = convert(Int, idx_op)
                            cur_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(cur_type, member))
                        elseif cur_type isa LLVM.ArrayType
                            cur_type = LLVM.eltype(cur_type)
                        else
                            valid = false; break
                        end
                    end
                    valid || continue
                    # The last level indexed must be an array
                    last_indexed_type isa LLVM.ArrayType || continue
                    # The chain GEP's source type should match the array element type
                    chain_src = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                    chain_src == cur_type || continue
                    push!(chain_to_fix, (inst, base))
                end

                if !isempty(chain_to_fix)
                    changed = true
                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        for (gep, base_gep) in chain_to_fix
                            LLVM.position!(builder, gep)
                            gep_ops = LLVM.operands(gep)
                            base_ops = LLVM.operands(base_gep)
                            # base_gep indices: [0, %idx]
                            # gep indices: [%offset, member_indices...]
                            # merged: [0, %idx + %offset, member_indices...]
                            base_last_idx = base_ops[length(base_ops)]
                            chain_first_idx = gep_ops[2]  # the pointer arithmetic offset
                            # Type-match for add
                            idx_ty = LLVM.value_type(base_last_idx)
                            off_ty = LLVM.value_type(chain_first_idx)
                            if off_ty != idx_ty
                                chain_first_idx = LLVM.sext!(builder, chain_first_idx, idx_ty, "chain_off_ext")
                            end
                            adj_idx = LLVM.add!(builder, base_last_idx, chain_first_idx, "arr_chain_adj")
                            # Build new GEP: gep array_type, ptr alloca, [base_indices_except_last..., adj_idx, chain_member_indices...]
                            src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(base_gep))
                            new_indices = LLVM.Value[]
                            for i in 2:(length(base_ops)-1)
                                push!(new_indices, base_ops[i])
                            end
                            push!(new_indices, adj_idx)
                            for i in 3:length(gep_ops)  # skip ptr and first_idx
                                push!(new_indices, gep_ops[i])
                            end
                            new_gep = LLVM.gep!(builder, src_ty, base_ops[1], new_indices,
                                               "alloca_chain_fix")
                            is_ib = LLVM.API.LLVMIsInBounds(gep) | LLVM.API.LLVMIsInBounds(base_gep)
                            LLVM.API.LLVMSetIsInBounds(new_gep, is_ib)
                            LLVM.replace_uses!(gep, new_gep)
                            LLVM.erase!(gep)
                        end
                    end
                end
            end
        end
    end
end

"""Check if a GEP instruction is used as the base pointer of another typed GEP."""
function _used_by_typed_gep(gep::LLVM.Instruction)
    for use in LLVM.uses(gep)
        user = LLVM.user(use)
        user isa LLVM.GetElementPtrInst || continue
        src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(user))
        # If the user GEP's source type is NOT i8, it's a typed GEP
        src_ty == LLVM.Int8Type() && continue
        return true
    end
    return false
end

"""
    _byte_offset_to_gep_indices(type, offset, dl, stop_at_aggregate) -> Vector{Int} or nothing

Map a byte offset within a struct type to a sequence of GEP indices.

If `stop_at_aggregate` is true, stop navigating when the remaining offset is 0
and the current type is an aggregate (struct/array). This is needed when the GEP
result is used as the base of another typed GEP (e.g., variable-indexed array access).

If `stop_at_aggregate` is false, navigate all the way to the scalar leaf field.
"""
function _byte_offset_to_gep_indices(type::LLVM.LLVMType, offset::Int,
                                      dl::LLVM.DataLayout, stop_at_aggregate::Bool)
    indices = Int[]
    _resolve_offset!(indices, type, offset, dl, stop_at_aggregate) && return indices
    return nothing
end

function _resolve_offset!(indices::Vector{Int}, type::LLVM.LLVMType, offset::Int,
                           dl::LLVM.DataLayout, stop_at_aggregate::Bool)
    # At aggregate boundary with offset=0: stop if we want aggregate-level navigation
    if offset == 0 && stop_at_aggregate
        return true
    end
    # At scalar leaf with offset=0: always stop
    if offset == 0 && !(type isa LLVM.StructType) && !(type isa LLVM.ArrayType)
        return true
    end

    if type isa LLVM.StructType
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_size = Int(LLVM.storage_size(dl, member_type))
            if offset >= member_offset && offset < member_offset + member_size
                push!(indices, i)
                return _resolve_offset!(indices, member_type,
                                        offset - member_offset, dl, stop_at_aggregate)
            end
        end
        return false  # offset in padding

    elseif type isa LLVM.ArrayType
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        elem_size == 0 && return false
        n = LLVM.length(type)
        idx = div(offset, elem_size)
        idx < n || return false
        push!(indices, idx)
        return _resolve_offset!(indices, elem_type, offset - idx * elem_size, dl,
                                stop_at_aggregate)

    else
        return offset == 0
    end
end

"""
    _load_composite_from_psb(builder, base_int, type, byte_offset, dl)

Recursively load a composite type from PhysicalStorageBuffer (addrspace 1).
For leaf scalar types: inttoptr(base + offset) → load.
For struct/array types: recursively load members and reconstruct via insertvalue.
"""
function _load_composite_from_psb(builder, base_int, type::LLVM.LLVMType,
                                   byte_offset::Int, dl::LLVM.DataLayout)
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)

    if type isa LLVM.StructType
        result = LLVM.UndefValue(type)
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = _load_composite_from_psb(builder, base_int, member_type,
                                                   byte_offset + member_offset, dl)
            result = LLVM.insert_value!(builder, result, member_val, i)
        end
        return result
    elseif type isa LLVM.ArrayType
        result = LLVM.UndefValue(type)
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = _load_composite_from_psb(builder, base_int, elem_type,
                                                 byte_offset + i * elem_size, dl)
            result = LLVM.insert_value!(builder, result, elem_val, i)
        end
        return result
    else
        # Leaf scalar: inttoptr(base + offset) → load
        if byte_offset == 0
            field_addr = base_int
        else
            field_addr = LLVM.add!(builder, base_int,
                LLVM.ConstantInt(T_i64, byte_offset))
        end
        field_ptr = LLVM.inttoptr!(builder, field_addr, T_ptr_as1)
        val = LLVM.load!(builder, type, field_ptr)
        LLVM.alignment!(val, max(4, _llvm_type_size(type)))
        return val
    end
end

"""
    _store_composite_to_psb(builder, val, base_int, type, byte_offset, dl)

Recursively store a composite type to PhysicalStorageBuffer (addrspace 1).
For leaf scalar types: extractvalue → inttoptr(base + offset) → store.
For struct/array types: recursively extract and store members.
"""
function _store_composite_to_psb(builder, val, base_int, type::LLVM.LLVMType,
                                  byte_offset::Int, dl::LLVM.DataLayout)
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(LLVM.Int8Type(), 1)

    if type isa LLVM.StructType
        n = LLVM.API.LLVMCountStructElementTypes(type)
        for i in 0:(n-1)
            member_offset = Int(LLVM.API.LLVMOffsetOfElement(dl, type, i))
            member_type = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(type, i))
            member_val = LLVM.extract_value!(builder, val, i)
            _store_composite_to_psb(builder, member_val, base_int, member_type,
                                     byte_offset + member_offset, dl)
        end
    elseif type isa LLVM.ArrayType
        elem_type = LLVM.eltype(type)
        elem_size = Int(LLVM.storage_size(dl, elem_type))
        for i in 0:(LLVM.length(type)-1)
            elem_val = LLVM.extract_value!(builder, val, i)
            _store_composite_to_psb(builder, elem_val, base_int, elem_type,
                                     byte_offset + i * elem_size, dl)
        end
    else
        # Leaf scalar: inttoptr(base + offset) → store
        if byte_offset == 0
            field_addr = base_int
        else
            field_addr = LLVM.add!(builder, base_int,
                LLVM.ConstantInt(T_i64, byte_offset))
        end
        field_ptr = LLVM.inttoptr!(builder, field_addr, T_ptr_as1)
        st = LLVM.store!(builder, val, field_ptr)
        LLVM.alignment!(st, max(4, _llvm_type_size(type)))
    end
end

"""
    _lower_psb_memops!(mod)

Lower `llvm.memset` and `llvm.memcpy` intrinsics on `ptr addrspace(1)` to
explicit store/load loops. The SPIR-V backend doesn't support these intrinsics
on PhysicalStorageBuffer pointers.

Memset is lowered to i32 stores (4-byte aligned). Memcpy is lowered to i32
load/store pairs. Both handle non-4-byte-aligned tails with i8 stores.
"""
function _lower_psb_memops!(mod::LLVM.Module)
    T_i8 = LLVM.Int8Type()
    T_i32 = LLVM.Int32Type()
    T_i64 = LLVM.Int64Type()
    T_ptr_as1 = LLVM.PointerType(T_i8, 1)

    for f in LLVM.functions(mod)
        for bb in LLVM.blocks(f)
            to_erase = LLVM.Instruction[]
            for inst in LLVM.instructions(bb)
                inst isa LLVM.CallInst || continue
                callee = LLVM.called_operand(inst)
                callee isa LLVM.Function || continue
                cname = LLVM.name(callee)

                if startswith(cname, "llvm.memset.p1")
                    # llvm.memset.p1.iN(ptr as(1) dst, i8 val, iN len, i1 volatile)
                    ops = LLVM.operands(inst)
                    dst_ptr = ops[1]
                    fill_val = ops[2]  # i8
                    len_val = ops[3]

                    # Only handle constant-length memsets (common for struct init)
                    len_val isa LLVM.ConstantInt || continue
                    nbytes = convert(Int, len_val)

                    LLVM.IRBuilder() do builder
                        LLVM.position!(builder, inst)
                        base_int = LLVM.ptrtoint!(builder, dst_ptr, T_i64, "memset.base")

                        # Build fill word: replicate i8 val to i32
                        val8 = fill_val
                        val32 = LLVM.zext!(builder, val8, T_i32)
                        v1 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 8))
                        val32 = LLVM.or!(builder, val32, v1)
                        v2 = LLVM.shl!(builder, val32, LLVM.ConstantInt(T_i32, 16))
                        val32 = LLVM.or!(builder, val32, v2)

                        # Store i32s for the bulk
                        n_words = nbytes ÷ 4
                        for i in 0:(n_words-1)
                            off = i * 4
                            addr = if off == 0
                                base_int
                            else
                                LLVM.add!(builder, base_int, LLVM.ConstantInt(T_i64, off))
                            end
                            ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1)
                            st = LLVM.store!(builder, val32, ptr)
                            LLVM.alignment!(st, 4)
                        end

                        # Handle tail bytes
                        for i in (n_words*4):(nbytes-1)
                            addr = LLVM.add!(builder, base_int, LLVM.ConstantInt(T_i64, i))
                            ptr = LLVM.inttoptr!(builder, addr, T_ptr_as1)
                            st = LLVM.store!(builder, val8, ptr)
                            LLVM.alignment!(st, 1)
                        end
                    end
                    push!(to_erase, inst)

                elseif startswith(cname, "llvm.memcpy.p1") || startswith(cname, "llvm.memcpy.p0.p1") || startswith(cname, "llvm.memcpy.p1.p0")
                    # Lower memcpy involving addrspace(1)
                    ops = LLVM.operands(inst)
                    dst_ptr = ops[1]
                    src_ptr = ops[2]
                    len_val = ops[3]

                    len_val isa LLVM.ConstantInt || continue
                    nbytes = convert(Int, len_val)

                    LLVM.IRBuilder() do builder
                        LLVM.position!(builder, inst)
                        dst_int = LLVM.ptrtoint!(builder, dst_ptr, T_i64, "memcpy.dst")
                        src_int = LLVM.ptrtoint!(builder, src_ptr, T_i64, "memcpy.src")

                        dst_as = LLVM.addrspace(LLVM.value_type(dst_ptr))
                        src_as = LLVM.addrspace(LLVM.value_type(src_ptr))
                        T_dst_ptr = LLVM.PointerType(T_i8, dst_as)
                        T_src_ptr = LLVM.PointerType(T_i8, src_as)

                        n_words = nbytes ÷ 4
                        for i in 0:(n_words-1)
                            off = i * 4
                            s_addr = off == 0 ? src_int : LLVM.add!(builder, src_int, LLVM.ConstantInt(T_i64, off))
                            s_ptr = LLVM.inttoptr!(builder, s_addr, T_src_ptr)
                            val = LLVM.load!(builder, T_i32, s_ptr)
                            LLVM.alignment!(val, 4)

                            d_addr = off == 0 ? dst_int : LLVM.add!(builder, dst_int, LLVM.ConstantInt(T_i64, off))
                            d_ptr = LLVM.inttoptr!(builder, d_addr, T_dst_ptr)
                            st = LLVM.store!(builder, val, d_ptr)
                            LLVM.alignment!(st, 4)
                        end

                        for i in (n_words*4):(nbytes-1)
                            s_addr = LLVM.add!(builder, src_int, LLVM.ConstantInt(T_i64, i))
                            s_ptr = LLVM.inttoptr!(builder, s_addr, T_src_ptr)
                            val = LLVM.load!(builder, T_i8, s_ptr)
                            LLVM.alignment!(val, 1)

                            d_addr = LLVM.add!(builder, dst_int, LLVM.ConstantInt(T_i64, i))
                            d_ptr = LLVM.inttoptr!(builder, d_addr, T_dst_ptr)
                            st = LLVM.store!(builder, val, d_ptr)
                            LLVM.alignment!(st, 1)
                        end
                    end
                    push!(to_erase, inst)
                end
            end
            for inst in to_erase
                LLVM.erase!(inst)
            end
        end
    end

    # Clean up now-unused memset/memcpy declarations
    for f in collect(LLVM.functions(mod))
        fname = LLVM.name(f)
        if (startswith(fname, "llvm.memset.p1") || startswith(fname, "llvm.memcpy.p1") ||
            startswith(fname, "llvm.memcpy.p0.p1") || startswith(fname, "llvm.memcpy.p1.p0"))
            if isempty(LLVM.blocks(f)) && isempty(LLVM.uses(f))
                LLVM.erase!(f)
            end
        end
    end
end

"""
    _decompose_composite_psb_accesses!(mod, dl)

Decompose composite (struct/array) loads/stores on `ptr addrspace(1)` into
individual scalar loads/stores with `inttoptr(base + offset)`.

The LLVM SPIR-V backend's "legalize bitcast" pass crashes when processing
`load %large_struct, ptr addrspace(1)` instructions. This occurs when device
code (e.g. `VkDeviceArray.getindex`) loads composite types from BDA pointers.
"""
function _decompose_composite_psb_accesses!(mod::LLVM.Module, dl::LLVM.DataLayout)
    T_i64 = LLVM.Int64Type()

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        to_erase = LLVM.Instruction[]

        for bb in LLVM.blocks(f)
            for inst in LLVM.instructions(bb)
                # Handle composite loads from addrspace(1)
                if inst isa LLVM.LoadInst
                    loaded_type = LLVM.value_type(inst)
                    (loaded_type isa LLVM.StructType || loaded_type isa LLVM.ArrayType) || continue

                    ptr = LLVM.operands(inst)[1]
                    ptr_ty = LLVM.value_type(ptr)
                    ptr_ty isa LLVM.PointerType || continue
                    LLVM.addrspace(ptr_ty) == 1 || continue

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        base_int = LLVM.ptrtoint!(builder, ptr, T_i64, "psb_base")
                        result = _load_composite_from_psb(builder, base_int,
                                                           loaded_type, 0, dl)
                        LLVM.replace_uses!(inst, result)
                    end
                    push!(to_erase, inst)

                # Handle composite stores to addrspace(1)
                elseif inst isa LLVM.StoreInst
                    ops = LLVM.operands(inst)
                    val = ops[1]
                    ptr = ops[2]
                    stored_type = LLVM.value_type(val)
                    (stored_type isa LLVM.StructType || stored_type isa LLVM.ArrayType) || continue

                    ptr_ty = LLVM.value_type(ptr)
                    ptr_ty isa LLVM.PointerType || continue
                    LLVM.addrspace(ptr_ty) == 1 || continue

                    LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder, inst)
                        base_int = LLVM.ptrtoint!(builder, ptr, T_i64, "psb_st_base")
                        _store_composite_to_psb(builder, val, base_int,
                                                 stored_type, 0, dl)
                    end
                    push!(to_erase, inst)
                end
            end
        end

        for inst in to_erase
            LLVM.erase!(inst)
        end
    end
end

function _prepare_module_for_vulkan!(mod::LLVM.Module, entry_name::String;
                                     dl::LLVM.DataLayout=LLVM.datalayout(mod))
    # 1. Strip llvm.lifetime.start/end intrinsics and set internal linkage
    for f in LLVM.functions(mod)
        fname = LLVM.name(f)

        # Set non-entry function definitions to internal linkage.
        # This prevents the LLVM SPIR-V backend from emitting LinkageAttributes
        # decorations and the Linkage capability (both illegal in Vulkan).
        # Skip declarations (no basic blocks) — they must keep external linkage.
        is_declaration = LLVM.API.LLVMIsDeclaration(f) != 0
        if fname != entry_name && !startswith(fname, "llvm.") && !is_declaration
            LLVM.linkage!(f, LLVM.API.LLVMInternalLinkage)
        end

        # Walk instructions and erase lifetime intrinsics + fix alignment
        for bb in LLVM.blocks(f)
            to_erase = LLVM.Instruction[]
            for inst in LLVM.instructions(bb)
                if inst isa LLVM.CallInst
                    callee = LLVM.called_operand(inst)
                    if callee isa LLVM.Function
                        cname = LLVM.name(callee)
                        if startswith(cname, "llvm.lifetime.")
                            push!(to_erase, inst)
                        end
                    end
                elseif inst isa LLVM.LoadInst || inst isa LLVM.StoreInst
                    # Fix alignment: Vulkan PhysicalStorageBuffer requires
                    # alignment >= natural size of the largest scalar in the type
                    # (VUID-StandaloneSpirv-PhysicalStorageBuffer64-06314).
                    align = LLVM.alignment(inst)
                    if align > 0
                        ty = if inst isa LLVM.LoadInst
                            LLVM.value_type(inst)
                        else
                            LLVM.value_type(LLVM.operands(inst)[1])
                        end
                        min_align = max(4, _scalar_size(ty))
                        if align < min_align
                            LLVM.alignment!(inst, min_align)
                        end
                    end
                end
            end
            for inst in to_erase
                LLVM.erase!(inst)
            end
        end
    end

    # 2. Remove declarations of lifetime intrinsics (now unused)
    for f in collect(LLVM.functions(mod))
        fname = LLVM.name(f)
        if startswith(fname, "llvm.lifetime.") && isempty(LLVM.blocks(f))
            LLVM.erase!(f)
        end
    end

    # 3. Fix flat GEPs on addrspace(3) array globals.
    # After inlining + InstCombine, two-level GEPs like:
    #   getelementptr [64 x float], ptr addrspace(3) @arr, 0, %idx
    # get flattened to:
    #   getelementptr float, ptr addrspace(3) @arr, %idx
    # The LLVM SPIR-V backend silently drops the index in this flat form,
    # producing OpAccessChain without indices.  Rewrite back to structured form.
    _fix_shared_geps!(mod)

    # 4. Flatten array-typed GEPs in addrspace(1) (BDA / PhysicalStorageBuffer).
    #    Julia compiles tuple stores as `getelementptr [3 x float], ptr addrspace(1), 0, idx`.
    #    The SPIR-V backend emits OpTypeArray without ArrayStride, which Vulkan requires
    #    for PhysicalStorageBuffer. Flatten to `getelementptr float, ptr, idx` so the
    #    array type never appears in SPIR-V.
    _flatten_bda_array_geps!(mod)

    # 5. Lower llvm.memset/memcpy on addrspace(1) to explicit stores/loads.
    # Julia emits llvm.memset.p1 when zeroing struct fields in BDA memory (e.g.
    # BVHNode2 initialization). The SPIR-V backend can't handle these intrinsics
    # on PhysicalStorageBuffer pointers.
    _lower_psb_memops!(mod)

    # 6. Decompose composite loads/stores on addrspace(1) (PhysicalStorageBuffer).
    # Julia device code (e.g. VkDeviceArray.getindex for struct element types) may
    # generate `load %large_struct, ptr addrspace(1)`. The SPIR-V backend's bitcast
    # legalization pass crashes on these. Decompose into individual scalar loads with
    # inttoptr(base + offset) and reconstruct via insertvalue.
    _decompose_composite_psb_accesses!(mod, dl)

    # 7. Hoist push constant loads into the entry block.
    # The LLVM SPIR-V backend correctly translates ConstantExpr GEPs on push
    # constants in the entry block but mishandles them in non-entry blocks
    # (producing byte-offset arithmetic instead of struct member access chains).
    # Push constants are uniform, so hoisting is semantically valid.
    _hoist_push_constant_loads!(mod)

    # 8. Check for unexpected constant globals in addrspace(1).
    # Julia captures constants (e.g. polynomial tables for pow/log) as global
    # variables in addrspace(1) (CrossWorkgroup). These are invalid in Vulkan SPIR-V
    # because CrossWorkgroup maps to PhysicalStorageBuffer, and OpVariable cannot
    # use that storage class. The proper fix is to override the functions that create
    # these globals (e.g. ^(Float64,Float64)) with intrinsic-based implementations.
    # If any slip through, warn so we can add the missing override.
    _warn_constant_globals!(mod)
end

"""
Replace unreachable control flow with branches to an exit block.

Ported from Metal.jl's `replace_unreachable!` (GPUCompiler/src/metal.jl).
After GPUCompiler's `lower_throw!` and `rm_trap!`, error paths end with
`unreachable` terminators. These create dead-end blocks that the LLVM SPIR-V
backend cannot structure into valid Vulkan SPIR-V. We replace them with
branches to a unified return block, which the backend handles correctly.
"""
function _replace_unreachable!(mod::LLVM.Module)
    # Process all functions in the module (entry + any remaining internal functions)
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        # Find unreachable instructions and exit blocks
        unreachables = LLVM.Instruction[]
        exit_blocks = LLVM.BasicBlock[]
        for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
            if inst isa LLVM.UnreachableInst
                push!(unreachables, inst)
            end
            if inst isa LLVM.RetInst
                push!(exit_blocks, bb)
            end
        end
        isempty(unreachables) && continue

        # If no exit block exists, create one with `ret void`
        if isempty(exit_blocks)
            ret_type = LLVM.return_type(LLVM.function_type(f))
            return_block = LLVM.BasicBlock(f, "ret")
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, return_block)
                if ret_type == LLVM.VoidType()
                    LLVM.ret!(builder)
                else
                    LLVM.ret!(builder, LLVM.null(ret_type))
                end
            end
            push!(exit_blocks, return_block)
        end

        LLVM.@dispose builder=LLVM.IRBuilder() begin
            # Use the last exit block (mirrors Metal's heuristic)
            exit_block = last(exit_blocks)
            ret = LLVM.terminator(exit_block)

            # Create a return block with only the return instruction
            if first(LLVM.instructions(exit_block)) == ret
                return_block = exit_block
            else
                return_block = LLVM.BasicBlock(f, "ret")
                LLVM.API.LLVMMoveBasicBlockAfter(return_block, exit_block)

                LLVM.position!(builder, ret)
                LLVM.br!(builder, return_block)

                LLVM.API.LLVMInstructionRemoveFromParent(ret)
                LLVM.position!(builder, return_block)
                LLVM.API.LLVMInsertIntoBuilder(builder, ret)
            end

            # When returning a value, add a phi node
            ret_ops = LLVM.operands(ret)
            if !isempty(ret_ops)
                LLVM.position!(builder, ret)
                val = ret_ops[1]
                phi = LLVM.phi!(builder, LLVM.value_type(val))
                for pred in LLVM.predecessors(return_block)
                    push!(LLVM.incoming(phi), (val, pred))
                end
                LLVM.operands(ret)[1] = phi
            end

            # Replace unreachable terminators with branches to return block
            for unreachable in unreachables
                bb = LLVM.parent(unreachable)

                LLVM.position!(builder, unreachable)
                LLVM.br!(builder, return_block)
                LLVM.erase!(unreachable)

                # Patch up phi nodes in the return block
                for inst in LLVM.instructions(return_block)
                    if inst isa LLVM.PHIInst
                        undef = LLVM.UndefValue(LLVM.value_type(inst))
                        push!(LLVM.incoming(inst), (undef, bb))
                    end
                end
            end
        end
    end
end

"""
Strip `noreturn` attribute from all functions in the module.

After `_replace_unreachable!` converts error paths to normal returns,
the functions are no longer truly noreturn. But LLVM's SimplifyCFG
sees the `noreturn` attribute and reintroduces `unreachable` terminators
after calls to these functions, undoing our fix. Stripping the attribute
prevents this.
"""
function _strip_noreturn!(mod::LLVM.Module)
    noreturn_kind = LLVM.kind(LLVM.EnumAttribute("noreturn"))
    for f in LLVM.functions(mod)
        # Strip from function definition attributes
        for attr in collect(LLVM.function_attributes(f))
            if LLVM.kind(attr) == noreturn_kind
                delete!(LLVM.function_attributes(f), attr)
            end
        end
        # Strip from call-site attributes on call instructions
        for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
            if inst isa LLVM.CallInst
                for attr in collect(LLVM.function_attributes(inst))
                    if LLVM.kind(attr) == noreturn_kind
                        delete!(LLVM.function_attributes(inst), attr)
                    end
                end
            end
        end
    end
end

"""
Strip `llvm.assume` intrinsic calls from the module.

These are optimization hints that have no effect on correctness.
The LLVM SPIR-V backend can misplace them relative to `OpSelectionMerge`
instructions, causing validation failures.
"""
function _strip_assume!(mod::LLVM.Module)
    if haskey(LLVM.functions(mod), "llvm.assume")
        assume_fn = LLVM.functions(mod)["llvm.assume"]
        for use in collect(LLVM.uses(assume_fn))
            val = LLVM.user(use)
            if val isa LLVM.CallInst
                LLVM.erase!(val)
            end
        end
        if isempty(LLVM.uses(assume_fn))
            LLVM.erase!(assume_fn)
        end
    end
end

"""
Replace `freeze` instructions with their operands.

The LLVM SPIR-V backend does not support the `freeze` instruction
(crashes in IRTranslator::translateFreeze). `freeze` makes poison/undef
values deterministic, but SPIR-V has no poison/undef semantics, so we
can safely replace `freeze %x` with `%x`.
"""
function _replace_freeze!(mod::LLVM.Module)
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        to_erase = LLVM.Instruction[]
        for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
            inst isa LLVM.FreezeInst || continue
            LLVM.replace_uses!(inst, LLVM.operands(inst)[1])
            push!(to_erase, inst)
        end
        for inst in to_erase
            LLVM.erase!(inst)
        end
    end
end

"""
    _fixup_structured_cfg!(mod::LLVM.Module)

Post-StructurizeCFG fixup pass, analogous to clspv's `FixupStructuredCFGPass`.

The LLVM SPIR-V backend assigns merge targets to structured constructs:
- Loop headers get `OpLoopMerge` targeting the loop exit block
- Conditional branches (non-latch) get `OpSelectionMerge` targeting the post-dominator

If a block is targeted by multiple conditional branches from different nesting
levels (e.g., both an outer selection and an inner loop target the same exit),
the SPIR-V backend produces duplicate merge targets, which is illegal in Vulkan.

Fix: detect blocks that are the successor of multiple conditional branches and
insert trampoline blocks so each construct gets its own unique merge target.
"""
function _fixup_structured_cfg!(mod::LLVM.Module)
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        _isolate_shared_merge_targets!(f)
    end
end

function _isolate_shared_merge_targets!(f::LLVM.Function)
    # Find blocks that are successors of multiple conditional branches.
    # Each such block would become a merge target of multiple SPIR-V constructs.
    # Map: target_block → [source_blocks that conditionally branch to it]
    cond_sources = Dict{LLVM.BasicBlock, Vector{LLVM.BasicBlock}}()
    for bb in LLVM.blocks(f)
        term = LLVM.terminator(bb)
        term isa LLVM.BrInst || continue
        LLVM.isconditional(term) || continue
        for succ in LLVM.successors(term)
            # Skip self-loops (loop back-edges). These are handled correctly by
            # the SPIR-V backend as OpLoopMerge, not OpSelectionMerge. Including
            # them creates dead trampoline blocks that break PHI node invariants.
            succ == bb && continue
            sources = get!(cond_sources, succ, LLVM.BasicBlock[])
            # Avoid counting the same source twice (both branches to same target)
            bb in sources || push!(sources, bb)
        end
    end

    for (target, sources) in cond_sources
        length(sources) <= 1 && continue

        # Keep the first source unchanged, redirect all others through trampolines.
        # The first source (in block order) keeps the direct edge; later ones get
        # trampolines. This matches clspv's isolateContinue() strategy.
        for src in sources[2:end]
            _insert_cfg_trampoline!(f, src, target)
        end
    end
end

"""Insert a trampoline block isolating `src`'s construct from `target`.

When `src` is a conditional branch header whose construct (all blocks reachable
from `src` without passing through `target`) also has blocks that branch to
`target`, ALL those branches must be redirected through the trampoline. Otherwise
the SPIR-V backend creates a construct where inner blocks exit "not via structured
exit". This mirrors clspv's `isolateContinue()`.
"""
function _insert_cfg_trampoline!(f::LLVM.Function, src::LLVM.BasicBlock,
                                  target::LLVM.BasicBlock)
    # Step 1: find all blocks "inside" src's construct.
    # = blocks reachable from src without passing through target.
    # In structured CFG, the only exit from a construct is through its merge block.
    inside = Set{LLVM.BasicBlock}()
    worklist = LLVM.BasicBlock[src]
    while !isempty(worklist)
        bb = pop!(worklist)
        bb in inside && continue    # already visited
        bb == target && continue     # don't enter target
        push!(inside, bb)
        term = LLVM.terminator(bb)
        for succ in LLVM.successors(term)
            push!(worklist, succ)
        end
    end

    # Step 2: create trampoline block (just branches to target)
    tramp = LLVM.BasicBlock(f, "cfg_fixup")
    LLVM.@dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, tramp)
        LLVM.br!(builder, target)
    end

    # Step 3: redirect ALL branches from inside blocks to target → trampoline
    for bb in inside
        term = LLVM.terminator(bb)
        succs = LLVM.successors(term)
        for i in 1:length(succs)
            if succs[i] == target
                succs[i] = tramp
            end
        end
    end

    # Step 4: update PHI nodes in target.
    # Multiple inside blocks may have been predecessors of target with different
    # values. They all now arrive via trampoline, so we need a PHI in trampoline
    # to merge their values, and a single entry in target's PHI from trampoline.
    for inst in collect(LLVM.instructions(target))
        inst isa LLVM.PHIInst || break  # PHIs are always at block start

        inc = LLVM.incoming(inst)
        # Partition incoming into inside vs outside
        inside_pairs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
        outside_pairs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
        for k in 1:length(inc)
            val, blk = inc[k]
            if blk in inside
                push!(inside_pairs, (val, blk))
            else
                push!(outside_pairs, (val, blk))
            end
        end

        isempty(inside_pairs) && continue

        # Determine the value arriving from the trampoline
        tramp_val = if length(inside_pairs) == 1
            # Single inside predecessor: use its value directly
            inside_pairs[1][1]
        elseif all(p -> p[1] == inside_pairs[1][1], inside_pairs)
            # All inside predecessors contribute the same value
            inside_pairs[1][1]
        else
            # Multiple different values: create a PHI in the trampoline
            LLVM.@dispose builder=LLVM.IRBuilder() begin
                # Position before the terminator (branch) in trampoline
                LLVM.position!(builder, LLVM.terminator(tramp))
                phi = LLVM.phi!(builder, LLVM.value_type(inst), "tramp.phi")
                append!(LLVM.incoming(phi), inside_pairs)
                phi
            end
        end

        # Rebuild target's PHI with trampoline as single predecessor for inside blocks
        new_pairs = copy(outside_pairs)
        push!(new_pairs, (tramp_val, tramp))

        LLVM.@dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, inst)
            new_phi = LLVM.phi!(builder, LLVM.value_type(inst), LLVM.name(inst) * ".fix")
            append!(LLVM.incoming(new_phi), new_pairs)
            LLVM.replace_uses!(inst, new_phi)
            LLVM.erase!(inst)
        end
    end
end

"""
Fix GEPs on addrspace(3) array globals for SPIR-V backend compatibility.

Three fixups:
1. Rewrite flat instruction GEPs (`getelementptr T, ptr @arr, idx`) to structured
   two-index form (`getelementptr [N x T], ptr @arr, 0, idx`). The LLVM SPIR-V backend
   silently drops the index in flat form.
2. Fix ConstantExpr GEP loads/stores on shared globals. The SPIR-V backend converts
   ConstantExpr GEPs to byte-offset bitcast+AccessChain, which is invalid.
3. Fix bare global loads/stores (`load float, ptr addrspace(3) @arr`) where InstCombine
   has constant-folded `GEP @arr, 0, 0` away. The SPIR-V backend crashes in its
   "legalize bitcast" pass because it needs an OpAccessChain to go from the array
   variable to element 0, but without a GEP there's no access chain to emit.

For fixups 2+3, we need non-constant element indices to prevent the IRBuilder's
constant folder from collapsing the GEP back to a ConstantExpr or bare global.
We find an existing i32 SSA value in the function (e.g. the thread ID from
extractelement) and use `sub %val, %val` / `sext` to create a non-constant i64
zero. The first GEP index (array-of-arrays offset) MUST remain constant `i64 0`
— the SPIR-V backend crashes if it's non-constant.
"""
function _fix_shared_geps!(mod::LLVM.Module)
    # Collect addrspace(3) globals with array type
    shared_globals = Dict{LLVM.Value, LLVM.LLVMType}()
    for gv in LLVM.globals(mod)
        pointee_ty = LLVM.global_value_type(gv)
        ptr_ty = LLVM.value_type(gv)
        if ptr_ty isa LLVM.PointerType && LLVM.addrspace(ptr_ty) == 3 &&
           pointee_ty isa LLVM.ArrayType
            shared_globals[gv] = pointee_ty
        end
    end

    isempty(shared_globals) && return

    T_i32 = LLVM.Int32Type()
    T_i64 = LLVM.Int64Type()

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue

        # Lazily created non-constant i64 zero (one per function).
        # Uses `sub %val, %val` on an existing i32 instruction to produce 0
        # without being recognized as a constant by the IRBuilder's folder.
        nonconst_zero = Ref{Union{Nothing, LLVM.Value}}(nothing)

        function _get_nonconst_zero!()
            nonconst_zero[] !== nothing && return nonconst_zero[]::LLVM.Value
            entry_bb = first(LLVM.blocks(f))
            # Find an existing i32 instruction in the entry block to derive zero from
            donor = nothing
            for inst in LLVM.instructions(entry_bb)
                if LLVM.value_type(inst) == T_i32 && inst != LLVM.terminator(entry_bb)
                    donor = inst
                    break
                end
            end
            if donor === nothing
                # Fallback: no i32 instruction found, create sub from first i32 param
                for p in LLVM.parameters(f)
                    if LLVM.value_type(p) == T_i32
                        donor = p
                        break
                    end
                end
            end
            @assert donor !== nothing "No i32 value found for non-constant zero"

            # Insert sub+sext right after the donor (or at entry start for params)
            ncz = LLVM.IRBuilder() do b
                insert_pt = if donor isa LLVM.Instruction
                    # Insert after the donor instruction
                    next = LLVM.API.LLVMGetNextInstruction(donor)
                    next == C_NULL ? LLVM.terminator(entry_bb) : LLVM.Instruction(next)
                else
                    first(LLVM.instructions(entry_bb))
                end
                LLVM.position!(b, insert_pt)
                z32 = LLVM.sub!(b, donor, donor, "shared_zero32")
                return LLVM.sext!(b, z32, T_i64, "shared_idx_zero")
            end
            nonconst_zero[] = ncz
            return ncz
        end

        for bb in LLVM.blocks(f)
            # Part 0: Fix negative-index GEPs on shared memory (addrspace 3).
            #
            # LLVM generates negative-index GEPs on shared memory pointers in
            # two patterns, both from Julia's 1-based `unsafe_load(ptr, i)`:
            #
            # Pattern A: ConstantExpr GEP base with negative array index
            #   GEP(T, ConstGEP([N x T], @shared, -1, N-1), %k)
            #   → GEP([N x T], @shared, 0, %k + (-1*N + N-1))
            #
            # Pattern B: GEP chain with constant -1 offset
            #   %a = GEP(T, @shared, %idx)
            #   %b = GEP(T, %a, -1)
            #   → GEP(T, @shared, %idx - 1)
            #
            # SPIR-V's OpAccessChain cannot represent negative constant indices.
            # Fix all patterns by folding into non-negative GEPs from the base.

            # Pattern A: ConstantExpr GEP bases
            constexpr_fixes = Tuple{LLVM.GetElementPtrInst, LLVM.Value, LLVM.LLVMType, Int}[]
            for inst in LLVM.instructions(bb)
                inst isa LLVM.GetElementPtrInst || continue
                ops = LLVM.operands(inst)
                length(ops) == 2 || continue
                ptr_op = ops[1]
                ptr_op isa LLVM.ConstantExpr || continue
                LLVM.API.LLVMGetConstOpcode(ptr_op) == LLVM.API.LLVMGetElementPtr || continue
                ce_ops = LLVM.operands(ptr_op)
                length(ce_ops) >= 3 || continue
                gv = ce_ops[1]
                haskey(shared_globals, gv) || continue
                arr_idx = ce_ops[2]
                elem_idx = ce_ops[3]
                arr_idx isa LLVM.ConstantInt || continue
                elem_idx isa LLVM.ConstantInt || continue
                a = convert(Int, arr_idx)
                b = convert(Int, elem_idx)
                arr_ty = shared_globals[gv]
                N = length(arr_ty)
                offset = a * N + b
                push!(constexpr_fixes, (inst, gv, arr_ty, offset))
            end

            if !isempty(constexpr_fixes)
                ncz = _get_nonconst_zero!()
                zero_const = LLVM.ConstantInt(T_i64, 0)
                LLVM.IRBuilder() do builder
                    for (gep, gv, arr_ty, offset) in constexpr_fixes
                        LLVM.position!(builder, gep)
                        dyn_idx = LLVM.operands(gep)[2]
                        adj_idx = if offset == 0
                            dyn_idx
                        else
                            LLVM.add!(builder, dyn_idx,
                                     LLVM.ConstantInt(LLVM.value_type(dyn_idx), offset),
                                     "shmem_adj_idx")
                        end
                        if LLVM.value_type(adj_idx) != T_i64
                            adj_idx = LLVM.sext!(builder, adj_idx, T_i64, "shmem_idx64")
                        end
                        new_gep = LLVM.gep!(builder, arr_ty, gv,
                                           LLVM.Value[zero_const, adj_idx],
                                           LLVM.name(gep) == "" ? "shmem_gep.fix" : LLVM.name(gep) * ".fix")
                        LLVM.replace_uses!(gep, new_gep)
                        LLVM.erase!(gep)
                    end
                end
            end

            # Pattern B: Flat GEP chains on addrspace(3) GEP results.
            # The SPIR-V backend cannot translate flat GEPs (element-type based)
            # on Workgroup pointers into valid OpAccessChain. Merge flat GEPs
            # into the preceding structured GEP by adding the offset to the
            # last index. Like clspv's SimplifyPointerBitcastPass::runOnGEPFromGEP.
            #
            # Before: %a = GEP [N x T], @shared, 0, %idx
            #         %b = GEP T, %a, %offset
            # After:  %b = GEP [N x T], @shared, 0, (%idx + %offset)
            #
            # Run in a loop since fixing one GEP may expose another chain.
            # Only collect GEPs whose base is NOT also being fixed (to avoid
            # invalidation when we erase replaced GEPs).
            changed = true
            while changed
                changed = false
                chain_fixes = Tuple{LLVM.GetElementPtrInst, LLVM.GetElementPtrInst, LLVM.Value}[]
                # First pass: collect all candidates
                candidates = Set{LLVM.GetElementPtrInst}()
                for inst in LLVM.instructions(bb)
                    inst isa LLVM.GetElementPtrInst || continue
                    ops = LLVM.operands(inst)
                    length(ops) == 2 || continue
                    ptr_op = ops[1]
                    ptr_op isa LLVM.GetElementPtrInst || continue
                    ptr_ty = LLVM.value_type(ptr_op)
                    ptr_ty isa LLVM.PointerType || continue
                    LLVM.addrspace(ptr_ty) == 3 || continue
                    push!(candidates, inst)
                end
                # Second pass: only fix GEPs whose base is NOT also a candidate
                # (process leaves first to avoid erasing bases)
                for inst in candidates
                    base = LLVM.operands(inst)[1]::LLVM.GetElementPtrInst
                    base in candidates && continue  # skip if base will also be fixed
                    push!(chain_fixes, (inst, base, LLVM.operands(inst)[2]))
                end

                if !isempty(chain_fixes)
                    changed = true
                    LLVM.IRBuilder() do builder
                        for (gep, base_gep, offset_val) in chain_fixes
                            LLVM.position!(builder, gep)
                            base_idx = LLVM.operands(base_gep)[end]
                            idx_ty = LLVM.value_type(base_idx)
                            off_ty = LLVM.value_type(offset_val)
                            if off_ty != idx_ty
                                if idx_ty == T_i64
                                    offset_val = LLVM.sext!(builder, offset_val, T_i64, "shmem_off64")
                                else
                                    base_idx = LLVM.sext!(builder, base_idx, T_i64, "shmem_base64")
                                    idx_ty = T_i64
                                end
                            end
                            adj_idx = LLVM.add!(builder, base_idx, offset_val,
                                               "shmem_chain_adj")
                            src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(base_gep))
                            base_ptr = LLVM.operands(base_gep)[1]
                            base_ops = LLVM.operands(base_gep)
                            new_indices = LLVM.Value[base_ops[i] for i in 2:length(base_ops)-1]
                            push!(new_indices, adj_idx)
                            new_gep = LLVM.gep!(builder, src_ty, base_ptr, new_indices,
                                               LLVM.name(gep) == "" ? "shmem_chain.fix" : LLVM.name(gep) * ".fix")
                            LLVM.replace_uses!(gep, new_gep)
                            LLVM.erase!(gep)
                        end
                    end
                end
            end

            # Part 1: Fix flat instruction GEPs
            to_fix = Tuple{LLVM.GetElementPtrInst, LLVM.LLVMType}[]
            for inst in LLVM.instructions(bb)
                inst isa LLVM.GetElementPtrInst || continue
                ops = LLVM.operands(inst)
                length(ops) == 2 || continue  # flat GEP: pointer + one index
                ptr_op = ops[1]
                haskey(shared_globals, ptr_op) || continue
                # Verify the GEP uses element type (flat), not array type (structured)
                src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                arr_ty = shared_globals[ptr_op]
                elem_ty = eltype(arr_ty)
                src_ty == elem_ty || continue
                push!(to_fix, (inst, arr_ty))
            end

            LLVM.IRBuilder() do builder
                for (gep, arr_ty) in to_fix
                    ops = LLVM.operands(gep)
                    ptr_op = ops[1]
                    idx = ops[2]
                    LLVM.position!(builder, gep)
                    zero = LLVM.ConstantInt(LLVM.value_type(idx), 0)
                    new_gep = LLVM.gep!(builder, arr_ty, ptr_op,
                                        LLVM.Value[zero, idx], LLVM.name(gep) * ".fix")
                    LLVM.replace_uses!(gep, new_gep)
                    LLVM.erase!(gep)
                end
            end

            # Parts 2+3: Fix constant-index shared memory accesses.
            # Collect loads/stores that use either:
            #   (A) bare shared array global as pointer (element index 0)
            #   (B) ConstantExpr GEP on shared array global (constant element index)
            # Both crash llc. We replace with instruction GEPs using non-constant indices.
            # Tuple: (instruction, operand_index, global, array_type, element_index)
            const_fixes = Tuple{LLVM.Instruction, Int, LLVM.Value, LLVM.LLVMType, Int}[]
            for inst in LLVM.instructions(bb)
                if inst isa LLVM.LoadInst
                    ptr_op = LLVM.operands(inst)[1]
                    op_idx = 0
                elseif inst isa LLVM.StoreInst
                    ptr_op = LLVM.operands(inst)[2]
                    op_idx = 1
                else
                    continue
                end

                # Case A: Bare global (constant-folded GEP @arr, 0, 0 → @arr)
                if haskey(shared_globals, ptr_op)
                    push!(const_fixes, (inst, op_idx, ptr_op, shared_globals[ptr_op], 0))
                    continue
                end

                # Case B: ConstantExpr GEP (e.g. GEP @arr, 0, 1)
                if ptr_op isa LLVM.ConstantExpr
                    LLVM.API.LLVMGetConstOpcode(ptr_op) == LLVM.API.LLVMGetElementPtr || continue
                    ce_ops = LLVM.operands(ptr_op)
                    gv = ce_ops[1]
                    haskey(shared_globals, gv) || continue
                    # Extract element index: GEP @arr, 0, idx → idx is ce_ops[3]
                    length(ce_ops) >= 3 || continue
                    idx_val = ce_ops[3]
                    idx_val isa LLVM.ConstantInt || continue
                    elem_idx = convert(Int, idx_val)
                    push!(const_fixes, (inst, op_idx, gv, shared_globals[gv], elem_idx))
                end
            end

            isempty(const_fixes) && continue

            ncz = _get_nonconst_zero!()

            zero_const = LLVM.ConstantInt(T_i64, 0)

            LLVM.IRBuilder() do builder
                for (inst, op_idx, gv, arr_ty, elem_idx) in const_fixes
                    LLVM.position!(builder, inst)
                    idx = if elem_idx == 0
                        ncz
                    else
                        LLVM.add!(builder, ncz, LLVM.ConstantInt(T_i64, elem_idx),
                                  "shared_idx.fix")
                    end
                    # First index MUST be constant 0 (array-of-arrays offset).
                    # The SPIR-V backend crashes if the first index is non-constant.
                    new_gep = LLVM.gep!(builder, arr_ty, gv,
                                       LLVM.Value[zero_const, idx], "shared_gep.fix")
                    LLVM.API.LLVMSetOperand(inst, op_idx, new_gep)
                end
            end
        end
    end
end

"""
Hoist loads from push constant globals (addrspace 2) into the entry block.

The LLVM SPIR-V backend correctly translates ConstantExpr GEPs on push constants
when they appear in the entry block (producing proper OpAccessChain with struct
member indices), but mishandles them in non-entry blocks (converting to byte-offset
arithmetic with OpAccessChain on uchar type). Since push constants are uniform
(constant for the entire shader invocation), hoisting all loads to the entry block
is semantically valid and avoids this backend bug.
"""
function _hoist_push_constant_loads!(mod::LLVM.Module)
    # Find push constant globals (addrspace 2)
    push_globals = Set{LLVM.Value}()
    for gv in LLVM.globals(mod)
        ptr_ty = LLVM.value_type(gv)
        if ptr_ty isa LLVM.PointerType && LLVM.addrspace(ptr_ty) == 2
            push!(push_globals, gv)
        end
    end
    isempty(push_globals) && return

    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        entry_bb = first(LLVM.blocks(f))

        # Find all loads from push constant globals in non-entry blocks
        to_hoist = LLVM.LoadInst[]
        for bb in LLVM.blocks(f)
            bb == entry_bb && continue
            for inst in LLVM.instructions(bb)
                inst isa LLVM.LoadInst || continue
                ptr_op = LLVM.operands(inst)[1]
                # Direct load from push constant global
                if ptr_op in push_globals
                    push!(to_hoist, inst)
                    continue
                end
                # Load via ConstantExpr GEP on push constant global
                if ptr_op isa LLVM.ConstantExpr
                    opcode = LLVM.API.LLVMGetConstOpcode(ptr_op)
                    if opcode == LLVM.API.LLVMGetElementPtr
                        ce_ops = LLVM.operands(ptr_op)
                        if ce_ops[1] in push_globals
                            push!(to_hoist, inst)
                        end
                    end
                end
            end
        end
        isempty(to_hoist) && continue

        # Find insertion point: just before the entry block's terminator
        entry_term = LLVM.terminator(entry_bb)

        LLVM.@dispose builder=LLVM.IRBuilder() begin
            for load_inst in to_hoist
                LLVM.API.LLVMInstructionRemoveFromParent(load_inst)
                LLVM.position!(builder, entry_term)
                LLVM.API.LLVMInsertIntoBuilder(builder, load_inst)
            end
        end
    end
end

"""
    _warn_constant_globals!(mod)

Check for constant global variables in addrspace(1) that would be invalid in Vulkan SPIR-V.
These indicate a missing device function override — the Julia stdlib function is using
lookup tables that produce global constant arrays.
"""
function _warn_constant_globals!(mod::LLVM.Module)
    for gv in LLVM.globals(mod)
        as = LLVM.API.LLVMGetPointerAddressSpace(LLVM.API.LLVMTypeOf(gv))
        as == 1 || continue
        LLVM.API.LLVMGetInitializer(gv) == C_NULL && continue
        name = LLVM.name(gv)
        @warn "Vulkan: constant global '$name' in addrspace(1) — needs device function override" maxlog=1
    end
end

"""
    _gep_byte_offset(dl, src_ty, indices) -> Int

Compute the total byte offset for a sequence of constant GEP indices starting
from `src_ty`. Returns `nothing` if any index is non-constant (runtime value).

For array types, each index selects an element (offset = idx * elem_size).
For struct types, each index selects a member (offset from DataLayout).
"""
function _gep_byte_offset(dl::LLVM.DataLayout, src_ty::LLVM.LLVMType,
                          indices::AbstractVector)
    offset = 0
    cur_ty = src_ty
    for idx_val in indices
        idx_val isa LLVM.ConstantInt || return nothing
        idx = convert(Int, idx_val)
        if cur_ty isa LLVM.ArrayType
            elem_ty = LLVM.eltype(cur_ty)
            elem_size = Int(LLVM.storage_size(dl, elem_ty))
            offset += idx * elem_size
            cur_ty = elem_ty
        elseif cur_ty isa LLVM.StructType
            offset += Int(LLVM.API.LLVMOffsetOfElement(dl, cur_ty, idx))
            cur_ty = LLVM.LLVMType(LLVM.API.LLVMStructGetTypeAtIndex(cur_ty, idx))
        else
            # Scalar — shouldn't happen in a valid multi-index GEP
            return nothing
        end
    end
    return offset
end

"""
Replace composite-typed GEPs in addrspace(1) with explicit pointer arithmetic.

Julia represents tuples as LLVM array types (e.g. [3 x float] for NTuple{3,Float32}).
The SPIR-V backend emits OpTypeArray without ArrayStride decoration, which Vulkan requires
for PhysicalStorageBuffer. Multi-level GEPs (e.g. `gep [3 x [1 x [3 x float]]], ptr, 0, 1, 0, 1`)
also crash the backend's bitcast legalization pass.

Solution: for any GEP on addrspace(1) with a composite source type and all-constant indices
starting with 0, compute the total byte offset and replace with ptrtoint→add→inttoptr.
This produces OpConvertUToPtr in SPIR-V and avoids problematic composite types entirely.
"""
function _flatten_bda_array_geps!(mod::LLVM.Module)
    i64 = LLVM.Int64Type()
    dl = LLVM.datalayout(mod)
    for f in LLVM.functions(mod)
        for bb in LLVM.blocks(f)
            to_fix = LLVM.GetElementPtrInst[]
            for inst in LLVM.instructions(bb)
                inst isa LLVM.GetElementPtrInst || continue

                # Check pointer operand is addrspace(1) (BDA / CrossWorkgroup → PhysicalStorageBuffer)
                ptr_op = LLVM.operands(inst)[1]
                ptr_ty = LLVM.value_type(ptr_op)
                ptr_ty isa LLVM.PointerType || continue
                LLVM.addrspace(ptr_ty) == 1 || continue

                # Source element type must be composite (array or struct)
                src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                (src_ty isa LLVM.ArrayType || src_ty isa LLVM.StructType) || continue

                ops = LLVM.operands(inst)
                length(ops) >= 3 || continue  # ptr, idx0, idx1, ...

                # First index must be constant 0 (no base pointer scaling)
                idx0 = ops[2]
                idx0 isa LLVM.ConstantInt || continue
                convert(Int, idx0) == 0 || continue

                # All remaining indices must be constant for compile-time offset
                remaining = @view ops[3:end]
                byte_off = _gep_byte_offset(dl, src_ty, remaining)
                byte_off === nothing && continue

                push!(to_fix, inst)
            end

            isempty(to_fix) && continue

            LLVM.IRBuilder() do builder
                for gep in to_fix
                    ops = LLVM.operands(gep)
                    ptr_op = ops[1]
                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep))
                    remaining = @view ops[3:end]
                    byte_off = _gep_byte_offset(dl, src_ty, remaining)

                    LLVM.position!(builder, gep)
                    # ptrtoint → add byte offset → inttoptr
                    base_int = LLVM.ptrtoint!(builder, ptr_op, i64, "bda.base")
                    if byte_off == 0
                        new_addr = base_int
                    else
                        off_const = LLVM.ConstantInt(i64, byte_off)
                        new_addr = LLVM.add!(builder, base_int, off_const, "bda.addr")
                    end
                    new_ptr = LLVM.inttoptr!(builder, new_addr, LLVM.value_type(ptr_op), LLVM.name(gep))

                    LLVM.replace_uses!(gep, new_ptr)
                    LLVM.erase!(gep)
                end
            end
        end
    end
end

function _emit_spirv(mod::LLVM.Module, target::GPUCompiler.SPIRVCompilerTarget)
    mktempdir() do dir
        bc_path = joinpath(dir, "kernel.bc")
        spv_path = joinpath(dir, "kernel.spv")

        open(bc_path, "w") do io
            write(io, mod)
        end

        # Debug: save a copy of the .bc file for inspection
        if get(ENV, "ABACUS_SAVE_IR", "") == "1"
            cp(bc_path, "/tmp/abacus_debug_kernel.bc"; force=true)
            open("/tmp/abacus_debug_kernel.ll", "w") do io
                write(io, string(mod))
            end
        end

        # -O0: prevent llc from collapsing StructurizeCFG's Flow blocks.
        # At -O2 (default), the SPIR-V backend simplifies empty blocks, merging
        # multiple OpSelectionMerge targets into one block (illegal in Vulkan).
        # We already run our own LLVM optimization passes, so -O0 is safe here.
        cmd = `$(SPIRV_LLVM_Backend_jll.llc()) $bc_path -filetype=obj -O0 -o $spv_path`
        if !isempty(target.extensions)
            str = join(map(ext -> "+$ext", target.extensions), ",")
            cmd = `$cmd -spirv-ext=$str`
        end
        # Redirect stderr to suppress LLVM stack dumps on llc crashes.
        # The crash info is not actionable by users; the Julia-level error is
        # sufficient for debugging.
        err_path = joinpath(dir, "llc_stderr.txt")
        try
            open(err_path, "w") do err_io
                run(pipeline(cmd; stderr=err_io))
            end
        catch e
            # Read stderr for diagnostic context
            llc_stderr = isfile(err_path) ? read(err_path, String) : ""
            # Save the failing .bc for debugging when env var is set
            if get(ENV, "ABACUS_SAVE_IR", "") == "1" || occursin("PLEASE submit a bug report", llc_stderr)
                debug_bc = "/tmp/abacus_failed_kernel.bc"
                cp(bc_path, debug_bc; force=true)
                @debug "Saved failing kernel IR to $debug_bc"
            end
            error("SPIR-V backend (llc) failed to compile kernel. " *
                  "Set ABACUS_SAVE_IR=1 to save the failing IR for debugging." *
                  (occursin("PLEASE submit", llc_stderr) ? " (llc crashed internally)" : ""))
        end

        return read(spv_path)
    end
end
