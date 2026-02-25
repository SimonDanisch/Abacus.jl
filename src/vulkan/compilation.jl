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
              push_size = linked.push_size)
end

# --- Cached compilation pipeline (mirrors AMDGPU's hipfunction/hipcompile/hiplink) ---

"""Compiled Vulkan kernel: holds the pipeline and metadata needed for dispatch."""
struct VkKernel{F, TT}
    fun::F
    pipeline::VkComputePipeline
    entry_name::String
    push_size::Int
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
        k = VkKernel{F, TT}(f, linked.pipeline, linked.entry_name, linked.push_size)
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
                  push_size = push_info.push_size)
    end
end

"""Linker function for `GPUCompiler.cached_compilation`.

Takes compiled SPIR-V and creates the Vulkan compute pipeline.
"""
function vklink(@nospecialize(job::GPUCompiler.CompilerJob), compiled)
    pipeline = get_pipeline(compiled.spirv_bytes, compiled.entry_name,
                            compiled.push_size)
    return (; compiled.spirv_bytes, compiled.entry_name, compiled.push_size, pipeline)
end

"""
Push constant layout info returned by `_wrap_entry_for_vulkan!`.
"""
struct PushConstantInfo
    wrapper_name::String
    push_size::Int
    # Vector of (offset, size) for each struct member
    member_layout::Vector{Pair{Int,Int}}
end

"""
    _wrap_entry_for_vulkan!(mod, entry) -> PushConstantInfo

Transform the LLVM module so the entry point is a void() function that loads
kernel arguments from a push constant global variable (addrspace 2).

- Original entry: internal + alwaysinline (becomes a callee)
- Wrapper: void(), hlsl.shader=compute, hlsl.numthreads=64,1,1
- Pointer args (addrspace 1) → stored as i64 (BDA), loaded via inttoptr
- Scalar args → stored directly in push constant struct
"""
function _wrap_entry_for_vulkan!(mod::LLVM.Module, entry::LLVM.Function;
                                 workgroup_size::NTuple{3,Int} = (64, 1, 1))
    entry_name = LLVM.name(entry)
    ft = LLVM.function_type(entry)
    param_types = LLVM.parameters(ft)

    # If no parameters, just ensure the hlsl attributes are present
    if isempty(param_types)
        return PushConstantInfo(entry_name, 0, Pair{Int,Int}[])
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

    # Build push constant struct type
    # - Pointer params without byval → i64 BDA (convert to pointer)
    # - Pointer params with byval(struct_type) → flatten struct fields into scalars
    # - Non-pointer scalar params → as-is
    T_i64 = LLVM.Int64Type()
    push_fields = LLVM.LLVMType[]
    # param_info entries:
    #   (:ptr, start_idx)                    — BDA pointer (i64 → inttoptr)
    #   (:scalar, start_idx)                 — scalar (load directly)
    #   (:byval, start_idx, byval_type)      — byval struct (alloca + store fields + pass ptr)
    param_info = []

    for (i, pt) in enumerate(param_types)
        start_idx = length(push_fields)
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
                # Byval struct: flatten all leaf fields into push constant scalars
                _flatten_struct_fields!(push_fields, byval_type)
                push!(param_info, (:byval, start_idx, byval_type))
            else
                # Regular pointer (BDA): store as i64
                push!(push_fields, T_i64)
                push!(param_info, (:ptr, start_idx))
            end
        else
            push!(push_fields, pt)
            push!(param_info, (:scalar, start_idx))
        end
    end
    T_push = LLVM.StructType(push_fields)

    # Compute member layout (offsets and sizes)
    member_layout = Pair{Int,Int}[]
    offset = 0
    for field in push_fields
        sz = _llvm_type_size(field)
        align = sz  # natural alignment
        offset = (offset + align - 1) & ~(align - 1)
        push!(member_layout, offset => sz)
        offset += sz
    end
    push_size = offset

    # Create global in addrspace 2 (maps to UniformConstant; spirv_fixup converts to PushConstant)
    # NOTE: addrspace 13 would map directly to PushConstant but requires LLVM 23+
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

        # Load entire push constant struct, then use extractvalue per field.
        # This avoids struct GEPs which the LLVM SPIR-V backend converts to
        # raw byte-offset arithmetic that RADV can't handle.
        # extractvalue maps to OpCompositeExtract in SPIR-V which works correctly.
        push_struct = LLVM.load!(builder, T_push, gv)

        args = LLVM.Value[]
        for (i, pt) in enumerate(param_types)
            info = param_info[i]
            if info[1] == :ptr
                bda = LLVM.extract_value!(builder, push_struct, info[2])
                ptr_val = LLVM.inttoptr!(builder, bda, pt)
                push!(args, ptr_val)
            elseif info[1] == :scalar
                val = LLVM.extract_value!(builder, push_struct, info[2])
                push!(args, val)
            elseif info[1] == :byval
                # Byval struct: alloca on stack, load flattened fields from
                # push constants via extractvalue, store into alloca, pass pointer.
                byval_type = info[3]
                alloca = LLVM.alloca!(builder, byval_type, "byval_arg_$i")
                flat_idx = Ref(info[2])
                _store_flattened_fields_ev!(builder, push_struct, T_push, alloca, byval_type, flat_idx)
                push!(args, alloca)
            end
        end

        LLVM.call!(builder, ft, entry, args)
        LLVM.ret!(builder)
    end

    return PushConstantInfo(wrapper_name, push_size, member_layout)
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

function _prepare_module_for_vulkan!(mod::LLVM.Module, entry_name::String)
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

    # 5. Hoist push constant loads into the entry block.
    # The LLVM SPIR-V backend correctly translates ConstantExpr GEPs on push
    # constants in the entry block but mishandles them in non-entry blocks
    # (producing byte-offset arithmetic instead of struct member access chains).
    # Push constants are uniform, so hoisting is semantically valid.
    _hoist_push_constant_loads!(mod)
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
Replace array-typed GEPs in addrspace(1) with explicit pointer arithmetic.

Julia represents tuples as LLVM array types (e.g. [3 x float] for NTuple{3,Float32}).
The SPIR-V backend emits OpTypeArray without ArrayStride decoration, which Vulkan requires
for PhysicalStorageBuffer. The backend also drops indices from flat GEPs on raw pointers.

Solution: replace `getelementptr [N x T], ptr, 0, idx` with ptrtoint→add→inttoptr,
which produces OpConvertUToPtr in SPIR-V and avoids both array types and dropped indices.
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

                # Check it's a structured GEP on an array type: [N x T], 0, idx
                src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(inst))
                src_ty isa LLVM.ArrayType || continue

                ops = LLVM.operands(inst)
                length(ops) == 3 || continue  # ptr, 0, idx

                # First index must be constant 0
                idx0 = ops[2]
                idx0 isa LLVM.ConstantInt || continue
                convert(Int, idx0) == 0 || continue

                push!(to_fix, inst)
            end

            isempty(to_fix) && continue

            LLVM.IRBuilder() do builder
                for gep in to_fix
                    ops = LLVM.operands(gep)
                    ptr_op = ops[1]
                    idx = ops[3]  # the element index
                    src_ty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(gep))
                    elem_size = LLVM.storage_size(dl, eltype(src_ty))

                    LLVM.position!(builder, gep)
                    # ptrtoint → compute byte offset → add → inttoptr
                    base_int = LLVM.ptrtoint!(builder, ptr_op, i64, "bda.base")
                    idx64 = if LLVM.value_type(idx) == i64
                        idx
                    else
                        LLVM.zext!(builder, idx, i64, "bda.idx")
                    end
                    stride = LLVM.ConstantInt(i64, elem_size)
                    byte_off = LLVM.mul!(builder, idx64, stride, "bda.off")
                    new_addr = LLVM.add!(builder, base_int, byte_off, "bda.addr")
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

        # -O0: prevent llc from collapsing StructurizeCFG's Flow blocks.
        # At -O2 (default), the SPIR-V backend simplifies empty blocks, merging
        # multiple OpSelectionMerge targets into one block (illegal in Vulkan).
        # We already run our own LLVM optimization passes, so -O0 is safe here.
        cmd = `$(SPIRV_LLVM_Backend_jll.llc()) $bc_path -filetype=obj -O0 -o $spv_path`
        if !isempty(target.extensions)
            str = join(map(ext -> "+$ext", target.extensions), ",")
            cmd = `$cmd -spirv-ext=$str`
        end
        run(cmd)

        return read(spv_path)
    end
end
