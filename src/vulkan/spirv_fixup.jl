# SPIR-V binary post-processor for Vulkan compliance
#
# Parses the SPIR-V binary into a structured SPVModule (header + instruction list),
# applies targeted fixup passes, then serializes back to bytes.
#
# Each fixup pass is a small, focused function operating on SPVModule.
# This separation makes it easy to debug, skip, or reorder passes.
#
# Reference: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html

using SPIRV_Tools_jll

# ── SPIR-V opcodes ──────────────────────────────────────────────────────
const OP_EXTENSION         = UInt16(10)
const OP_EXT_INST_IMPORT   = UInt16(11)
const OP_EXT_INST          = UInt16(12)
const OP_MEMORY_MODEL      = UInt16(14)
const OP_ENTRY_POINT       = UInt16(15)
const OP_EXECUTION_MODE    = UInt16(16)
const OP_CAPABILITY        = UInt16(17)
const OP_TYPE_VOID         = UInt16(19)
const OP_TYPE_BOOL         = UInt16(20)
const OP_TYPE_INT          = UInt16(21)
const OP_TYPE_FLOAT        = UInt16(22)
const OP_TYPE_VECTOR       = UInt16(23)
const OP_TYPE_ARRAY        = UInt16(28)
const OP_TYPE_STRUCT       = UInt16(30)
const OP_TYPE_POINTER      = UInt16(32)
const OP_CONSTANT          = UInt16(43)
const OP_FUNCTION          = UInt16(54)
const OP_FUNCTION_END      = UInt16(56)
const OP_VARIABLE          = UInt16(59)
const OP_ACCESS_CHAIN      = UInt16(65)
const OP_DECORATE          = UInt16(71)
const OP_MEMBER_DECORATE   = UInt16(72)
const OP_BITCAST           = UInt16(124)
const OP_IS_NAN            = UInt16(156)
const OP_IS_INF            = UInt16(157)
const OP_ORDERED           = UInt16(162)
const OP_UNORDERED         = UInt16(163)
const OP_LOGICAL_OR        = UInt16(166)
const OP_LOGICAL_NOT       = UInt16(168)
const OP_CONTROL_BARRIER   = UInt16(224)
const OP_PHI               = UInt16(245)
const OP_LOOP_MERGE        = UInt16(246)
const OP_SELECTION_MERGE   = UInt16(247)
const OP_LABEL             = UInt16(248)
const OP_BRANCH            = UInt16(249)
const OP_BRANCH_COND       = UInt16(250)
const OP_SWITCH            = UInt16(251)

# ── SPIR-V enums ────────────────────────────────────────────────────────
const MEM_SEQUENTIALLY_CONSISTENT = UInt32(0x10)
const MEM_ACQUIRE_RELEASE         = UInt32(0x8)
const MEM_WORKGROUP_MEMORY        = UInt32(0x100)

const SC_UNIFORM_CONSTANT        = UInt32(0)
const SC_CROSS_WORKGROUP         = UInt32(5)
const SC_PUSH_CONSTANT           = UInt32(9)
const SC_PHYSICAL_STORAGE_BUFFER = UInt32(5349)

const CAP_ADDRESSES                          = UInt32(4)
const CAP_SHADER                             = UInt32(1)
const CAP_LINKAGE                            = UInt32(5)
const CAP_VARIABLE_POINTERS_STORAGE_BUFFER   = UInt32(4441)
const CAP_VARIABLE_POINTERS                  = UInt32(4442)
const CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES  = UInt32(5347)

const AM_PHYSICAL_STORAGE_BUFFER_64 = UInt32(5348)

const DEC_BLOCK              = UInt32(2)
const DEC_CONSTANT           = UInt32(22)  # Requires Kernel capability — not valid in Vulkan
const DEC_OFFSET             = UInt32(35)
const DEC_FP_FAST_MATH_MODE  = UInt32(40)  # Requires Kernel/FloatControls2 — not valid in Vulkan
const DEC_LINKAGE_ATTRIBUTES = UInt32(41)
const DEC_ALIGNMENT          = UInt32(44)

const SC_PRIVATE                     = UInt32(6)

const OP_COPY_MEMORY         = UInt16(63)
const OP_COPY_MEMORY_SIZED   = UInt16(64)

const EXEC_GL_COMPUTE = UInt32(5)

const SC_FUNCTION                = UInt32(7)

const CAP_UNTYPED_POINTERS_KHR  = UInt32(4473)

const OP_UNTYPED_ACCESS_CHAIN_KHR = UInt16(4419)
const OP_TYPE_UNTYPED_POINTER_KHR = UInt16(4417)
const OP_IN_BOUNDS_ACCESS_CHAIN = UInt16(66)


# ══════════════════════════════════════════════════════════════════════════
# Structured SPIR-V representation
# ══════════════════════════════════════════════════════════════════════════

"""A single SPIR-V instruction, stored as its raw word sequence."""
struct SPVInst
    words::Vector{UInt32}
end

@inline opcode(inst::SPVInst)    = (inst.words[1] & 0xFFFF) % UInt16
@inline wordcount(inst::SPVInst) = length(inst.words)

"""Construct an instruction from opcode + operand words."""
function SPVInst(op::UInt16, operands::UInt32...)
    wc = 1 + length(operands)
    SPVInst(UInt32[(UInt32(wc) << 16) | UInt32(op), operands...])
end

function Base.show(io::IO, inst::SPVInst)
    print(io, "SPVInst(op=", Int(opcode(inst)), ", wc=", wordcount(inst), ")")
end

"""Parsed SPIR-V module: 5-word header + ordered instruction list."""
mutable struct SPVModule
    header::Vector{UInt32}   # [magic, version, generator, bound, reserved]
    insts::Vector{SPVInst}
end

"""Allocate a fresh SPIR-V ID (increments the bound in the header)."""
function alloc_id!(mod::SPVModule)::UInt32
    id = mod.header[4]
    mod.header[4] += 1
    return id
end


# ── Parsing & serialization ─────────────────────────────────────────────

@inline _spv_wordcount(word::UInt32) = (word >> 16) % Int

function parse_spirv(spv_bytes::Vector{UInt8})
    @assert length(spv_bytes) % 4 == 0
    parse_spirv(collect(reinterpret(UInt32, spv_bytes)))
end

function parse_spirv(words::Vector{UInt32})
    @assert words[1] == 0x07230203 "Not a SPIR-V binary"
    header = words[1:5]
    insts = SPVInst[]
    pos = 6
    while pos <= length(words)
        wc = _spv_wordcount(words[pos])
        wc == 0 && break
        push!(insts, SPVInst(words[pos:pos+wc-1]))
        pos += wc
    end
    SPVModule(header, insts)
end

function serialize(mod::SPVModule)::Vector{UInt32}
    out = copy(mod.header)
    for inst in mod.insts
        append!(out, inst.words)
    end
    out
end

function to_bytes(mod::SPVModule)::Vector{UInt8}
    collect(reinterpret(UInt8, serialize(mod)))
end


# ── String helpers ───────────────────────────────────────────────────────

"Encode a string as null-terminated SPIR-V words."
function spv_encode_string(s::String)
    bytes = Vector{UInt8}(s)
    push!(bytes, 0x00)
    while length(bytes) % 4 != 0
        push!(bytes, 0x00)
    end
    return reinterpret(UInt32, bytes) |> collect
end

"Read a null-terminated UTF-8 string from SPIR-V words starting at index `i`."
function spv_read_string(words, i::Int)
    buf = UInt8[]
    while i <= length(words)
        w = words[i]
        for shift in (0, 8, 16, 24)
            b = (w >> shift) % UInt8
            b == 0x00 && return (String(buf), i)
            push!(buf, b)
        end
        i += 1
    end
    return (String(buf), i)
end


# ══════════════════════════════════════════════════════════════════════════
# Fixup passes — each is a focused function operating on SPVModule
# ══════════════════════════════════════════════════════════════════════════

# ── 1. Remap ExtInstImport "OpenCL.std" → "GLSL.std.450" ────────────────
# The LLVM SPIR-V backend emits @builtin_ccall functions as OpExtInst with
# GLSL.std.450 instruction numbers, but labels the import as "OpenCL.std".

function fix_ext_inst_import!(mod::SPVModule)
    opencl_set_id = UInt32(0)
    for (i, inst) in enumerate(mod.insts)
        if opcode(inst) == OP_EXT_INST_IMPORT
            name, _ = spv_read_string(inst.words, 3)
            if name == "OpenCL.std"
                opencl_set_id = inst.words[2]
                str_words = spv_encode_string("GLSL.std.450")
                wc = 2 + length(str_words)
                mod.insts[i] = SPVInst(UInt32[
                    (UInt32(wc) << 16) | UInt32(OP_EXT_INST_IMPORT),
                    opencl_set_id, str_words...
                ])
            end
        end
    end
    # Remap OpenCL.std instruction numbers → GLSL.std.450 equivalents
    opencl_set_id == 0 && return
    _remap_opencl_ext_insts!(mod, opencl_set_id)
end

# OpenCL.std instruction number → GLSL.std.450 instruction number
const _OPENCL_TO_GLSL = Dict{UInt32, UInt32}(
    0  => 17,  # acos → Acos
    1  => 23,  # acosh → Acosh
    3  => 16,  # asin → Asin
    4  => 22,  # asinh → Asinh
    6  => 18,  # atan → Atan
    7  => 25,  # atan2 → Atan2
    8  => 24,  # atanh → Atanh
    12 => 9,   # ceil → Ceil
    14 => 14,  # cos → Cos
    15 => 20,  # cosh → Cosh
    19 => 27,  # exp → Exp
    20 => 29,  # exp2 → Exp2
    23 => 4,   # fabs → FAbs
    25 => 8,   # floor → Floor
    26 => 50,  # fma → Fma
    27 => 40,  # fmax → FMax
    28 => 37,  # fmin → FMin
    29 => 1,   # fmod → (no direct equivalent, but keeping for now)
    48 => 26,  # pow → Pow
    56 => 13,  # sin → Sin
    58 => 19,  # sinh → Sinh
    60 => 31,  # sqrt → Sqrt
    61 => 15,  # tan → Tan
    62 => 21,  # tanh → Tanh
    63 => 3,   # tanpi → (approx Trunc — placeholder)
    # Integer operations (same semantics, different numbering)
    # s_abs → SAbs, etc. — add as needed
)

function _remap_opencl_ext_insts!(mod::SPVModule, set_id::UInt32)
    for (i, inst) in enumerate(mod.insts)
        if opcode(inst) == OP_EXT_INST && wordcount(inst) >= 5
            # OpExtInst: [opcode|wc, result_type, result_id, set, instruction, operands...]
            if inst.words[4] == set_id
                opencl_num = inst.words[5]
                glsl_num = get(_OPENCL_TO_GLSL, opencl_num, nothing)
                if glsl_num !== nothing
                    inst.words[5] = glsl_num
                else
                    @warn "Unmapped OpenCL.std instruction $opencl_num — passing through unchanged"
                end
            end
        end
    end
end

# ── 2. Fix capabilities & extensions ─────────────────────────────────────
# Remove Linkage capability, add PhysicalStorageBufferAddresses + extension.

function fix_capabilities!(mod::SPVModule)
    # Collect capabilities we want to keep from the source (e.g. Float64, Int64)
    # and skip OpenCL-only ones. Then prepend the Vulkan-required set.
    keep_caps = Set{UInt32}()
    # Vulkan-invalid capabilities to strip (emitted by llvm-spirv translator)
    skip_caps = Set{UInt32}([
        CAP_ADDRESSES,          # 4  — OpenCL only
        CAP_LINKAGE,            # 5  — Linkage (OpenCL only)
        UInt32(6),              # Kernel — OpenCL only
        UInt32(38),             # GenericPointer — not allowed in Vulkan 1.3
    ])

    for inst in mod.insts
        if opcode(inst) == OP_CAPABILITY
            cap = inst.words[2]
            if cap ∉ skip_caps
                push!(keep_caps, cap)
            end
        end
    end

    # Build the canonical capability + extension section
    new_insts = SPVInst[]

    # Required Vulkan capabilities
    required_caps = UInt32[
        CAP_SHADER,
        CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES,
        CAP_VARIABLE_POINTERS_STORAGE_BUFFER,
        CAP_VARIABLE_POINTERS,
    ]

    all_caps = Set(required_caps)
    union!(all_caps, keep_caps)

    for cap in sort(collect(all_caps))
        push!(new_insts, SPVInst(OP_CAPABILITY, cap))
    end

    # Required extensions
    for ext_name in ["SPV_KHR_physical_storage_buffer",
                     "SPV_KHR_variable_pointers",
                     "SPV_KHR_storage_buffer_storage_class"]
        str = spv_encode_string(ext_name)
        wc = 1 + length(str)
        push!(new_insts, SPVInst(UInt32[(UInt32(wc) << 16) | UInt32(OP_EXTENSION), str...]))
    end

    # Append everything after the original capability/extension section
    for inst in mod.insts
        op = opcode(inst)
        op == OP_CAPABILITY && continue
        op == OP_EXTENSION && continue
        push!(new_insts, inst)
    end

    empty!(mod.insts)
    append!(mod.insts, new_insts)
end

# ── 3. Fix memory model ──────────────────────────────────────────────────
# Set addressing model to PhysicalStorageBuffer64.

function fix_memory_model!(mod::SPVModule)
    for inst in mod.insts
        if opcode(inst) == OP_MEMORY_MODEL && wordcount(inst) == 3
            inst.words[2] = AM_PHYSICAL_STORAGE_BUFFER_64
            # Memory model: OpenCL(2) → GLSL450(1) for Vulkan
            if inst.words[3] == UInt32(2)  # OpenCL
                inst.words[3] = UInt32(1)  # GLSL450
            end
        end
    end
end

# ── 4. Fix entry points — keep only the target entry ─────────────────────

function fix_entry_points!(mod::SPVModule, entry_name::String)
    isempty(entry_name) && return

    # Check if there's already an entry point for our function
    has_entry = false
    for inst in mod.insts
        if opcode(inst) == OP_ENTRY_POINT && wordcount(inst) >= 4
            name, _ = spv_read_string(inst.words, 4)
            if name == entry_name
                has_entry = true
            end
        end
    end

    if !has_entry
        # llvm-spirv doesn't create OpEntryPoint — we need to add one.
        # Find the function ID by matching OpName.
        func_id = UInt32(0)
        for inst in mod.insts
            if opcode(inst) == UInt16(5) # OpName
                name, _ = spv_read_string(inst.words, 3)
                if name == entry_name
                    func_id = inst.words[2]
                    break
                end
            end
        end
        func_id == 0 && error("fix_entry_points!: cannot find function '$entry_name' in OpName declarations")

        # In SPIR-V 1.4+, all module-scope variables used by the entry point
        # must be listed. Collect all non-Function storage class variables.
        # OpVariable: [wc|op, result_type, result_id, storage_class]
        interface_ids = UInt32[]
        for inst in mod.insts
            if opcode(inst) == OP_VARIABLE && wordcount(inst) >= 4
                sc = inst.words[4]
                # Include all module-scope variables (everything except Function=7)
                if sc != SC_FUNCTION
                    push!(interface_ids, inst.words[3])  # result_id
                end
            end
        end

        # Build OpEntryPoint: [wc|op, GLCompute, func_id, name..., interface_ids...]
        name_words = spv_encode_string(entry_name)
        ep_words = UInt32[UInt32(0), EXEC_GL_COMPUTE, func_id, name_words..., interface_ids...]
        wc = length(ep_words)
        ep_words[1] = (UInt32(wc) << 16) | UInt32(OP_ENTRY_POINT)
        ep_inst = SPVInst(ep_words)

        # Insert after capabilities/extensions/ext_inst_import/memory_model
        insert_idx = 1
        for (i, inst) in enumerate(mod.insts)
            op = opcode(inst)
            if op == OP_CAPABILITY || op == OP_EXTENSION || op == OP_EXT_INST_IMPORT || op == OP_MEMORY_MODEL
                insert_idx = i + 1
            else
                break
            end
        end
        insert!(mod.insts, insert_idx, ep_inst)
    end

    # Remove any entry points that don't match our target
    filter!(mod.insts) do inst
        if opcode(inst) == OP_ENTRY_POINT && wordcount(inst) >= 4
            name, _ = spv_read_string(inst.words, 4)
            return name == entry_name
        end
        true
    end
end

# ── 5. Fix execution modes — keep only for the target entry ──────────────

function fix_execution_modes!(mod::SPVModule, entry_id::UInt32;
                               workgroup_size::NTuple{3,Int}=(64,1,1))
    entry_id == 0 && return
    filter!(mod.insts) do inst
        opcode(inst) == OP_EXECUTION_MODE && wordcount(inst) >= 3 ?
            inst.words[2] == entry_id : true
    end

    # Check if LocalSize execution mode exists for our entry point
    has_local_size = false
    for inst in mod.insts
        if opcode(inst) == OP_EXECUTION_MODE && wordcount(inst) >= 3
            if inst.words[2] == entry_id && inst.words[3] == UInt32(17)  # LocalSize
                has_local_size = true
                break
            end
        end
    end

    if !has_local_size
        # Add OpExecutionMode LocalSize after OpEntryPoint
        # OpExecutionMode entry_id LocalSize x y z
        local_size_inst = SPVInst(UInt32[
            (UInt32(6) << 16) | UInt32(OP_EXECUTION_MODE),
            entry_id,
            UInt32(17),  # LocalSize
            UInt32(workgroup_size[1]),
            UInt32(workgroup_size[2]),
            UInt32(workgroup_size[3])
        ])
        # Insert after last OpEntryPoint
        insert_idx = 1
        for (i, inst) in enumerate(mod.insts)
            if opcode(inst) == OP_ENTRY_POINT
                insert_idx = i + 1
            end
        end
        insert!(mod.insts, insert_idx, local_size_inst)
    end
end

# ── 6. Fix decorations ───────────────────────────────────────────────────
# Remove Linkage/Alignment decorations, insert Block + MemberOffset for
# the push constant struct (before any type declarations).

function fix_decorations!(mod::SPVModule, push_struct_type_id::UInt32,
                           push_struct_info::Vector{Pair{Int,Int}})
    new_insts = SPVInst[]
    decorations_added = false

    for inst in mod.insts
        op = opcode(inst)

        # Remove decorations not valid in Vulkan Shader model:
        # - Linkage, Alignment (existing)
        # - Constant (requires Kernel capability)
        # - FPFastMathMode (requires Kernel or FloatControls2)
        if op == OP_DECORATE && wordcount(inst) >= 3
            dec = inst.words[3]
            (dec == DEC_LINKAGE_ATTRIBUTES || dec == DEC_ALIGNMENT ||
             dec == DEC_CONSTANT || dec == DEC_FP_FAST_MATH_MODE) && continue
        end

        # Insert push constant decorations before first type/function declaration
        if !decorations_added && (OP_TYPE_VOID <= op <= OP_TYPE_POINTER || op == OP_FUNCTION)
            append!(new_insts, _make_push_decorations(push_struct_type_id, push_struct_info))
            decorations_added = true
        end

        push!(new_insts, inst)
    end

    empty!(mod.insts)
    append!(mod.insts, new_insts)
end

function _make_push_decorations(struct_id::UInt32, members::Vector{Pair{Int,Int}})
    struct_id == 0 && return SPVInst[]
    insts = SPVInst[]
    # OpDecorate struct_id Block
    push!(insts, SPVInst(OP_DECORATE, struct_id, DEC_BLOCK))
    # OpMemberDecorate struct_id member Offset value
    for (i, (offset, _)) in enumerate(members)
        push!(insts, SPVInst(UInt32[
            (UInt32(5) << 16) | UInt32(OP_MEMBER_DECORATE),
            struct_id, UInt32(i - 1), DEC_OFFSET, UInt32(offset)
        ]))
    end
    insts
end

# ── 7. Fix storage classes ────────────────────────────────────────────────
# CrossWorkgroup → PhysicalStorageBuffer.
# UniformConstant: push constant struct pointers → PushConstant,
#                  other constants (lookup tables) → Private.

function fix_storage_classes!(mod::SPVModule)
    # Identify which UniformConstant pointer types point to structs (= push constants)
    struct_ids = Set{UInt32}()
    for inst in mod.insts
        opcode(inst) == OP_TYPE_STRUCT && push!(struct_ids, inst.words[2])
    end

    push_ptr_type_ids = Set{UInt32}()
    for inst in mod.insts
        if opcode(inst) == OP_TYPE_POINTER && wordcount(inst) == 4
            if inst.words[3] == SC_UNIFORM_CONSTANT && inst.words[4] in struct_ids
                push!(push_ptr_type_ids, inst.words[2])
            end
        end
    end

    # Collect non-struct UniformConstant pointer types and their pointee types
    # We'll decide the target storage class (Private vs PushConstant) based on usage
    uc_non_struct_ptrs = Dict{UInt32, UInt32}()  # ptr_type_id → pointee_type_id
    for inst in mod.insts
        if opcode(inst) == OP_TYPE_POINTER && wordcount(inst) == 4
            if inst.words[3] == SC_UNIFORM_CONSTANT && !(inst.words[2] in push_ptr_type_ids)
                uc_non_struct_ptrs[inst.words[2]] = inst.words[4]
            end
        end
    end

    # Identify PushConstant variables (after remapping)
    push_var_ids = Set{UInt32}()
    for inst in mod.insts
        if opcode(inst) == OP_VARIABLE && wordcount(inst) >= 4
            if inst.words[4] == SC_UNIFORM_CONSTANT && inst.words[2] in push_ptr_type_ids
                push!(push_var_ids, inst.words[3])
            end
        end
    end

    # Find which non-struct UC pointer types are used as OpAccessChain/OpBitcast result types
    # on PushConstant base variables — these need PushConstant, not Private.
    # OpBitcast from push constant var is common: LLVM emits `%x = OpBitcast %member_ptr %pc_var`
    # and spirv-opt may convert it to `OpAccessChain %member_ptr %pc_var %0`.
    needs_push_version = Set{UInt32}()
    needs_private_version = Set{UInt32}()
    for inst in mod.insts
        op = opcode(inst)
        if op in (OP_ACCESS_CHAIN, UInt16(66))  # 66 = OpInBoundsAccessChain
            result_type = inst.words[2]
            base_id = inst.words[4]
            if haskey(uc_non_struct_ptrs, result_type)
                if base_id in push_var_ids
                    push!(needs_push_version, result_type)
                else
                    push!(needs_private_version, result_type)
                end
            end
        elseif op == OP_BITCAST && wordcount(inst) == 4
            # OpBitcast: [header, result_type, result_id, operand]
            result_type = inst.words[2]
            operand_id = inst.words[4]
            if haskey(uc_non_struct_ptrs, result_type)
                if operand_id in push_var_ids
                    push!(needs_push_version, result_type)
                else
                    push!(needs_private_version, result_type)
                end
            end
        end
    end

    # For non-struct UC pointers used ONLY in push constant access chains,
    # remap the existing type to PushConstant directly.
    # For those used in both contexts or only non-push contexts, create Private copies.
    # For those used in push context AND other contexts, create both.
    uc_to_private = Dict{UInt32, UInt32}()
    uc_to_push = Dict{UInt32, UInt32}()
    new_type_insts = SPVInst[]

    for (ptr_id, pointee_id) in uc_non_struct_ptrs
        in_push = ptr_id in needs_push_version
        in_other = ptr_id in needs_private_version

        if in_push && !in_other
            # Only used in push constant access chains — remap in-place to PushConstant
            uc_to_push[ptr_id] = ptr_id  # identity mapping, will change SC in-place
        elseif !in_push && in_other
            # Only used in non-push contexts — create Private copy
            new_id = alloc_id!(mod)
            uc_to_private[ptr_id] = new_id
            push!(new_type_insts, SPVInst(UInt32[
                (UInt32(4) << 16) | UInt32(OP_TYPE_POINTER),
                new_id, SC_PRIVATE, pointee_id
            ]))
        elseif in_push && in_other
            # Used in both — create new types for both, keep original for other uses
            push_id = alloc_id!(mod)
            private_id = alloc_id!(mod)
            uc_to_push[ptr_id] = push_id
            uc_to_private[ptr_id] = private_id
            push!(new_type_insts, SPVInst(UInt32[
                (UInt32(4) << 16) | UInt32(OP_TYPE_POINTER),
                push_id, SC_PUSH_CONSTANT, pointee_id
            ]))
            push!(new_type_insts, SPVInst(UInt32[
                (UInt32(4) << 16) | UInt32(OP_TYPE_POINTER),
                private_id, SC_PRIVATE, pointee_id
            ]))
        else
            # Not used in any access chain — still needs Private for variables
            new_id = alloc_id!(mod)
            uc_to_private[ptr_id] = new_id
            push!(new_type_insts, SPVInst(UInt32[
                (UInt32(4) << 16) | UInt32(OP_TYPE_POINTER),
                new_id, SC_PRIVATE, pointee_id
            ]))
        end
    end

    # Apply storage class remappings to pointer types and variables
    for inst in mod.insts
        op = opcode(inst)
        if op == OP_TYPE_POINTER && wordcount(inst) == 4
            if inst.words[3] == SC_CROSS_WORKGROUP
                inst.words[3] = SC_PHYSICAL_STORAGE_BUFFER
            elseif inst.words[3] == SC_UNIFORM_CONSTANT && inst.words[2] in push_ptr_type_ids
                inst.words[3] = SC_PUSH_CONSTANT
            elseif inst.words[3] == SC_UNIFORM_CONSTANT && haskey(uc_to_push, inst.words[2]) && uc_to_push[inst.words[2]] == inst.words[2]
                # In-place remap to PushConstant (only-push-use case)
                inst.words[3] = SC_PUSH_CONSTANT
            end
        elseif op == OP_VARIABLE && wordcount(inst) >= 4
            if inst.words[4] == SC_CROSS_WORKGROUP
                inst.words[4] = SC_PHYSICAL_STORAGE_BUFFER
            elseif inst.words[4] == SC_UNIFORM_CONSTANT
                type_id = inst.words[2]
                if type_id in push_ptr_type_ids
                    inst.words[4] = SC_PUSH_CONSTANT
                elseif haskey(uc_to_private, type_id)
                    inst.words[2] = uc_to_private[type_id]
                    inst.words[4] = SC_PRIVATE
                end
            end
        end
    end

    # Update OpAccessChain/OpInBoundsAccessChain/OpBitcast result types
    for inst in mod.insts
        op = opcode(inst)
        if op in (OP_ACCESS_CHAIN, UInt16(66))  # 66 = OpInBoundsAccessChain
            result_type = inst.words[2]
            base_id = inst.words[4]
            if base_id in push_var_ids && haskey(uc_to_push, result_type)
                inst.words[2] = uc_to_push[result_type]
            elseif haskey(uc_to_private, result_type)
                inst.words[2] = uc_to_private[result_type]
            end
        elseif op == OP_BITCAST && wordcount(inst) == 4
            result_type = inst.words[2]
            operand_id = inst.words[4]
            if operand_id in push_var_ids && haskey(uc_to_push, result_type)
                inst.words[2] = uc_to_push[result_type]
            elseif haskey(uc_to_private, result_type)
                inst.words[2] = uc_to_private[result_type]
            end
        end
    end

    # Insert new pointer types before the first global OpVariable or OpFunction
    # (SPIR-V layout: types must appear before variables that reference them)
    if !isempty(new_type_insts)
        new_insts = SPVInst[]
        inserted = false
        for inst in mod.insts
            op = opcode(inst)
            if !inserted && (op == OP_VARIABLE || op == OP_FUNCTION)
                append!(new_insts, new_type_insts)
                inserted = true
            end
            push!(new_insts, inst)
        end
        empty!(mod.insts)
        append!(mod.insts, new_insts)
    end
end

# ── 7b. Strip disallowed OpVariable initializers ─────────────────────────
# In Vulkan SPIR-V, OpVariable with an Initializer operand is only allowed in
# Output(3), Private(6), Function(7), and Workgroup(4) with ZeroInitialize.
# The LLVM SPIR-V backend sometimes emits initializers for PhysicalStorageBuffer
# or PushConstant variables (from Julia constants captured in closures).
# Strip the initializer word to make these valid.

function fix_variable_initializers!(mod::SPVModule)
    # Storage classes that allow initializers
    allowed_init_sc = Set{UInt32}([
        UInt32(3),   # Output
        UInt32(6),   # Private
        UInt32(7),   # Function
    ])

    for inst in mod.insts
        if opcode(inst) == OP_VARIABLE && wordcount(inst) == 5
            sc = inst.words[4]
            if sc ∉ allowed_init_sc
                # Strip the initializer (5th word) by truncating + fixing word count
                pop!(inst.words)
                inst.words[1] = (UInt32(4) << 16) | UInt32(OP_VARIABLE)
            end
        end
    end
end


# ── 7b. Add structured control flow annotations ──────────────────────────
# llvm-spirv (SPIRV-LLVM-Translator) emits unstructured SPIR-V (OpenCL style)
# without OpSelectionMerge/OpLoopMerge. Vulkan requires structured control flow.
# Since we run StructurizeCFG on the LLVM IR, the CFG is already structured —
# we just need to add the merge annotations.
#
# Algorithm: build CFG, compute immediate post-dominators, insert merge
# instructions before OpBranchConditional/OpSwitch that lack them.

function fix_structured_control_flow!(mod::SPVModule)
    # 1. Find all functions and add merge instructions
    func_ranges = Tuple{Int,Int}[]
    func_start = 0
    for (i, inst) in enumerate(mod.insts)
        if opcode(inst) == OP_FUNCTION
            func_start = i
        elseif opcode(inst) == OP_FUNCTION_END && func_start > 0
            push!(func_ranges, (func_start, i))
            func_start = 0
        end
    end

    # Process in reverse to keep indices valid after insertions
    for (fstart, fend) in reverse(func_ranges)
        _add_structured_merges!(mod, fstart, fend)
    end

    # 2. Fix block positions: move misplaced merge/continue blocks so that
    #    every construct's merge block appears within its parent construct.
    #    Re-scan function ranges since insertions shifted indices.
    func_ranges2 = Tuple{Int,Int}[]
    func_start = 0
    for (i, inst) in enumerate(mod.insts)
        if opcode(inst) == OP_FUNCTION
            func_start = i
        elseif opcode(inst) == OP_FUNCTION_END && func_start > 0
            push!(func_ranges2, (func_start, i))
            func_start = 0
        end
    end
    for (fstart, fend) in reverse(func_ranges2)
        _fix_merge_block_positions!(mod, fstart, fend)
    end

    # 3. Fix construct block ordering: ensure all blocks belonging to a
    #    selection/loop construct appear BEFORE the merge block in linear order.
    #    StructurizeCFG can place some construct blocks (e.g. "err" blocks reached
    #    via the true branch) after the merge in block order.
    func_ranges3 = Tuple{Int,Int}[]
    func_start = 0
    for (i, inst) in enumerate(mod.insts)
        if opcode(inst) == OP_FUNCTION
            func_start = i
        elseif opcode(inst) == OP_FUNCTION_END && func_start > 0
            push!(func_ranges3, (func_start, i))
            func_start = 0
        end
    end
    for (fstart, fend) in reverse(func_ranges3)
        _fix_construct_block_order!(mod, fstart, fend)
    end
end

"""
Move blocks placed after the function's exit (between OpReturn and
OpFunctionEnd) to their correct structural position. llvm-spirv's
StructurizeCFG can place "Flow"/"cfg_fixup" blocks at the end of the function,
outside the loop constructs that reference them as merge targets.

Each misplaced block is moved to right before its branch successor.
Only blocks AFTER the exit block are considered misplaced.
"""
function _fix_merge_block_positions!(mod::SPVModule, fstart::Int, fend::Int)
    for _iter in 1:20
        # Find the actual function end (may have shifted from previous iterations)
        actual_fend = fend
        for i in fstart:length(mod.insts)
            if opcode(mod.insts[i]) == OP_FUNCTION_END
                actual_fend = i
                break
            end
        end

        # Find exit block (block with OpReturn/OpReturnValue/OpUnreachable)
        exit_term_idx = 0
        for i in fstart:actual_fend
            op = opcode(mod.insts[i])
            if op == OP_RETURN || op == OP_RETURN_VALUE || op == OP_UNREACHABLE
                exit_term_idx = i
                break
            end
        end
        exit_term_idx == 0 && return

        # Collect blocks after the exit terminator and before OpFunctionEnd
        post_blocks = Tuple{UInt32, Int, Int}[]  # (label, start_idx, end_idx)
        cur_label = UInt32(0)
        cur_start = 0
        for i in (exit_term_idx + 1):actual_fend
            inst = mod.insts[i]
            op = opcode(inst)
            if op == OP_LABEL && wordcount(inst) >= 2
                cur_label = inst.words[2]
                cur_start = i
            elseif (op == OP_BRANCH || op == OP_BRANCH_COND || op == OP_SWITCH) && cur_label != 0
                push!(post_blocks, (cur_label, cur_start, i))
                cur_label = UInt32(0)
            elseif op == OP_FUNCTION_END
                break
            end
        end

        isempty(post_blocks) && return

        # Move ALL post-exit blocks to right before the exit block, preserving
        # their relative order. This ensures they end up after all function body
        # blocks (satisfying dominator ordering). The subsequent merge-blocks pass
        # will merge them with adjacent predecessors where possible.

        # Collect all post-exit instructions
        all_post_insts = SPVInst[]
        for (_, bs, be) in post_blocks
            append!(all_post_insts, mod.insts[bs:be])
        end

        # Remove all post-exit blocks (in reverse order to keep indices valid)
        for k in length(post_blocks):-1:1
            _, bs, be = post_blocks[k]
            deleteat!(mod.insts, bs:be)
        end

        # Find the exit block start (after deletions)
        exit_insert = 0
        for i in fstart:length(mod.insts)
            op = opcode(mod.insts[i])
            if op == OP_RETURN || op == OP_RETURN_VALUE || op == OP_UNREACHABLE
                # Walk back to find OpLabel
                for j in i:-1:fstart
                    if opcode(mod.insts[j]) == OP_LABEL
                        exit_insert = j
                        break
                    end
                end
                break
            end
        end

        exit_insert == 0 && return

        # Insert before the exit block
        splice!(mod.insts, exit_insert:exit_insert-1, all_post_insts)
    end
end

"""
Fix construct block ordering: for each selection/loop construct, ensure all blocks
that belong to the construct appear BEFORE the merge block in linear order.

StructurizeCFG can place some construct blocks (e.g. "err" blocks reached via
the true branch of a selection) after the merge block in block order. This
violates the SPIR-V rule that the merge block must be the first block after
the construct.

Strategy: for each selection/loop merge, collect blocks that are reachable from
the header WITHOUT going through the merge (= construct body). If any of these
blocks appear after the merge in block order, move them to right before the merge.
"""
function _fix_construct_block_order!(mod::SPVModule, fstart::Int, fend::Int)
    for _iter in 1:20
        # Rebuild block map each iteration since indices shift
        blocks = Dict{UInt32, @NamedTuple{start_idx::Int, end_idx::Int}}()
        block_order = UInt32[]
        successors = Dict{UInt32, Vector{UInt32}}()
        current_label = UInt32(0)
        block_start = 0

        actual_fend = fend
        for i in fstart:length(mod.insts)
            if opcode(mod.insts[i]) == OP_FUNCTION_END
                actual_fend = i
                break
            end
        end

        for i in fstart:actual_fend
            inst = mod.insts[i]
            op = opcode(inst)
            if op == OP_LABEL
                current_label = inst.words[2]
                block_start = i
                push!(block_order, current_label)
                successors[current_label] = UInt32[]
            elseif op == OP_BRANCH && current_label != 0
                push!(successors[current_label], inst.words[2])
                blocks[current_label] = (; start_idx=block_start, end_idx=i)
                current_label = UInt32(0)
            elseif op == OP_BRANCH_COND && current_label != 0
                push!(successors[current_label], inst.words[3])
                push!(successors[current_label], inst.words[4])
                blocks[current_label] = (; start_idx=block_start, end_idx=i)
                current_label = UInt32(0)
            elseif op == OP_SWITCH && current_label != 0
                push!(successors[current_label], inst.words[3])
                for j in 4:2:wordcount(inst)
                    j + 1 <= wordcount(inst) && push!(successors[current_label], inst.words[j+1])
                end
                blocks[current_label] = (; start_idx=block_start, end_idx=i)
                current_label = UInt32(0)
            elseif (op == UInt16(253) || op == UInt16(254) || op == UInt16(255) || op == OP_FUNCTION_END) && current_label != 0
                blocks[current_label] = (; start_idx=block_start, end_idx=i)
                current_label = UInt32(0)
            end
        end

        all_labels = Set(block_order)
        label_pos = Dict(l => i for (i, l) in enumerate(block_order))

        # Collect all merge targets with their headers
        merges = Tuple{UInt32, UInt32}[]  # (header, merge_target)
        for i in fstart:actual_fend
            inst = mod.insts[i]
            op = opcode(inst)
            if (op == OP_SELECTION_MERGE || op == OP_LOOP_MERGE) && wordcount(inst) >= 3
                # Find header label (walk back to OpLabel)
                header_label = UInt32(0)
                for j in i:-1:fstart
                    if opcode(mod.insts[j]) == OP_LABEL
                        header_label = mod.insts[j].words[2]
                        break
                    end
                end
                header_label != 0 && push!(merges, (header_label, inst.words[2]))
            end
        end

        moved = false
        for (header, merge_target) in merges
            haskey(label_pos, header) || continue
            haskey(label_pos, merge_target) || continue
            merge_pos = label_pos[merge_target]

            # BFS from header, not going through merge_target, to find construct blocks
            construct_blocks = Set{UInt32}()
            worklist = [header]
            visited = Set{UInt32}([header])
            while !isempty(worklist)
                node = popfirst!(worklist)
                push!(construct_blocks, node)
                for succ in get(successors, node, UInt32[])
                    succ == merge_target && continue
                    succ ∉ all_labels && continue
                    succ ∈ visited && continue
                    push!(visited, succ)
                    push!(worklist, succ)
                end
            end

            # Find construct blocks that appear AFTER the merge in block order
            misplaced = UInt32[]
            for blk in construct_blocks
                blk == header && continue  # header is fine where it is
                haskey(label_pos, blk) || continue
                if label_pos[blk] > merge_pos
                    push!(misplaced, blk)
                end
            end

            isempty(misplaced) && continue

            # Sort misplaced blocks by their current position (maintain relative order)
            sort!(misplaced; by=l -> label_pos[l])

            # Move each misplaced block to right before the merge block
            for blk_label in misplaced
                haskey(blocks, blk_label) || continue
                blk = blocks[blk_label]
                blk_insts = mod.insts[blk.start_idx:blk.end_idx]

                # Find current merge block position (may have shifted)
                merge_insert = 0
                for i in fstart:length(mod.insts)
                    if opcode(mod.insts[i]) == OP_LABEL && mod.insts[i].words[2] == merge_target
                        merge_insert = i
                        break
                    end
                end
                merge_insert == 0 && continue

                # Find current block position
                blk_start_cur = 0
                blk_end_cur = 0
                for i in fstart:length(mod.insts)
                    if opcode(mod.insts[i]) == OP_LABEL && mod.insts[i].words[2] == blk_label
                        blk_start_cur = i
                        # Find end of this block
                        for j in i+1:length(mod.insts)
                            op = opcode(mod.insts[j])
                            if op == OP_BRANCH || op == OP_BRANCH_COND || op == OP_SWITCH ||
                               op == UInt16(253) || op == UInt16(254) || op == UInt16(255) ||
                               op == OP_LABEL
                                blk_end_cur = (op == OP_LABEL) ? j - 1 : j
                                break
                            end
                        end
                        break
                    end
                end
                (blk_start_cur == 0 || blk_end_cur == 0) && continue

                # Extract block instructions
                extracted = mod.insts[blk_start_cur:blk_end_cur]
                deleteat!(mod.insts, blk_start_cur:blk_end_cur)

                # Re-find merge position after deletion
                merge_insert_new = 0
                for i in fstart:length(mod.insts)
                    if opcode(mod.insts[i]) == OP_LABEL && mod.insts[i].words[2] == merge_target
                        merge_insert_new = i
                        break
                    end
                end
                merge_insert_new == 0 && continue

                # Insert before merge block
                splice!(mod.insts, merge_insert_new:merge_insert_new-1, extracted)
                moved = true
                break  # restart outer loop since indices shifted
            end
            moved && break
        end
        moved || return  # no more moves needed
    end
end

# Reorder blocks within each function so every block appears after its dominator.
# Must run AFTER all passes that create new blocks (fix_duplicate_merge_targets!,
# fix_structured_control_flow!, etc.) to ensure trampoline blocks are also correctly ordered.
function fix_block_ordering!(mod::SPVModule)
    func_ranges = Tuple{Int,Int}[]
    func_start = 0
    for (i, inst) in enumerate(mod.insts)
        if opcode(inst) == OP_FUNCTION
            func_start = i
        elseif opcode(inst) == OP_FUNCTION_END && func_start > 0
            push!(func_ranges, (func_start, i))
            func_start = 0
        end
    end

    for (fstart, fend) in reverse(func_ranges)
        _reorder_blocks_rpo!(mod, fstart, fend)
    end
end

# Reorder blocks within a function so every block appears after its dominator.
# Uses the Cooper-Harvey-Kennedy immediate dominator algorithm, then BFS of
# the dominator tree to produce a valid ordering for SPIR-V.
function _reorder_blocks_rpo!(mod::SPVModule, fstart::Int, fend::Int)
    blocks = Dict{UInt32, @NamedTuple{start_idx::Int, end_idx::Int}}()
    block_order = UInt32[]
    successors = Dict{UInt32, Vector{UInt32}}()
    predecessors = Dict{UInt32, Vector{UInt32}}()

    current_label = UInt32(0)
    block_start = 0

    for i in fstart:fend
        inst = mod.insts[i]
        op = opcode(inst)

        if op == OP_LABEL
            current_label = inst.words[2]
            block_start = i
            push!(block_order, current_label)
            successors[current_label] = UInt32[]
            predecessors[current_label] = UInt32[]
        elseif current_label != 0
            if op == OP_BRANCH
                push!(successors[current_label], inst.words[2])
                blocks[current_label] = (; start_idx=block_start, end_idx=i)
                current_label = UInt32(0)
            elseif op == OP_BRANCH_COND
                push!(successors[current_label], inst.words[3])
                push!(successors[current_label], inst.words[4])
                blocks[current_label] = (; start_idx=block_start, end_idx=i)
                current_label = UInt32(0)
            elseif op == OP_SWITCH
                push!(successors[current_label], inst.words[3])
                j = 4
                while j + 1 <= wordcount(inst)
                    push!(successors[current_label], inst.words[j+1])
                    j += 2
                end
                blocks[current_label] = (; start_idx=block_start, end_idx=i)
                current_label = UInt32(0)
            elseif op == UInt16(253) || op == UInt16(254) || op == UInt16(255) || op == OP_FUNCTION_END
                blocks[current_label] = (; start_idx=block_start, end_idx=i)
                current_label = UInt32(0)
            end
        end
    end

    isempty(block_order) && return
    length(block_order) <= 1 && return
    all_labels = Set(block_order)

    # Build predecessor map
    for (label, succs) in successors
        for s in succs
            haskey(predecessors, s) && push!(predecessors[s], label)
        end
    end

    entry = block_order[1]
    n = length(block_order)
    label_to_idx = Dict{UInt32, Int}()
    for (i, l) in enumerate(block_order)
        label_to_idx[l] = i
    end

    # Compute RPO via iterative DFS
    # rpo_labels[k] = label of the k-th block in RPO order (k=1 is entry)
    rpo_labels = UInt32[]
    visited = Set{UInt32}()
    dfs_stack = Tuple{UInt32, Int}[(entry, 1)]
    push!(visited, entry)
    while !isempty(dfs_stack)
        label, si = dfs_stack[end]
        succs = get(successors, label, UInt32[])
        found_next = false
        while si <= length(succs)
            s = succs[si]; si += 1
            if s in all_labels && s ∉ visited
                dfs_stack[end] = (label, si)
                push!(visited, s)
                push!(dfs_stack, (s, 1))
                found_next = true
                break
            end
        end
        if !found_next
            dfs_stack[end] = (label, si)
            pop!(dfs_stack)
            push!(rpo_labels, label)
        end
    end
    reverse!(rpo_labels)

    # Build RPO numbering: label → RPO index (1-based, entry = 1)
    label_to_rpo = Dict{UInt32, Int}()
    for (rpo_idx, label) in enumerate(rpo_labels)
        label_to_rpo[label] = rpo_idx
    end
    n_rpo = length(rpo_labels)

    # Compute immediate dominators using Cooper-Harvey-Kennedy algorithm.
    # idom[rpo_idx] = RPO index of immediate dominator.
    # The intersect function requires RPO numbering: dominator has LOWER RPO number.
    idom = zeros(Int, n_rpo)
    idom[1] = 1  # entry dominates itself

    @inline function intersect_idom(b1::Int, b2::Int)
        f1, f2 = b1, b2
        while f1 != f2
            while f1 > f2; f1 = idom[f1]; f1 == 0 && return b2; end
            while f2 > f1; f2 = idom[f2]; f2 == 0 && return b1; end
        end
        return f1
    end

    changed = true
    while changed
        changed = false
        for rpo_idx in 2:n_rpo
            label = rpo_labels[rpo_idx]
            preds = get(predecessors, label, UInt32[])
            new_idom = 0
            for p in preds
                haskey(label_to_rpo, p) || continue
                pi = label_to_rpo[p]
                idom[pi] == 0 && continue  # not yet processed
                new_idom = new_idom == 0 ? pi : intersect_idom(new_idom, pi)
            end
            if new_idom != 0 && new_idom != idom[rpo_idx]
                idom[rpo_idx] = new_idom
                changed = true
            end
        end
    end

    # Build dominator tree and BFS for dominator-first ordering.
    # Children sorted by RPO index to preserve topological order.
    dom_children = [Int[] for _ in 1:n_rpo]
    for i in 2:n_rpo
        if idom[i] != 0 && idom[i] != i
            push!(dom_children[idom[i]], i)
        end
    end

    new_order = UInt32[]
    queue = Int[1]
    while !isempty(queue)
        idx = popfirst!(queue)
        push!(new_order, rpo_labels[idx])
        # Children already in RPO order (lower index = closer to entry)
        append!(queue, sort(dom_children[idx]))
    end

    # Add unreachable blocks at the end (not reachable from entry)
    reachable = Set(new_order)
    for label in block_order
        if label ∉ reachable
            push!(new_order, label)
        end
    end

    new_order == block_order && return

    # Collect pre-block instructions (OpFunction, OpFunctionParameter, etc.)
    first_block_start = blocks[block_order[1]].start_idx
    pre_block_insts = mod.insts[fstart:first_block_start-1]

    # Collect post-block instructions (OpFunctionEnd)
    last_block_end = blocks[block_order[end]].end_idx
    post_block_insts = mod.insts[last_block_end+1:fend]

    new_insts = SPVInst[]
    append!(new_insts, pre_block_insts)
    for label in new_order
        haskey(blocks, label) || continue
        blk = blocks[label]
        append!(new_insts, mod.insts[blk.start_idx:blk.end_idx])
    end
    append!(new_insts, post_block_insts)

    splice!(mod.insts, fstart:fend, new_insts)
end

function _add_structured_merges!(mod::SPVModule, fstart::Int, fend::Int)
    # Build block map: label_id → (start_idx, end_idx)
    # and successor/predecessor maps
    blocks = Dict{UInt32, @NamedTuple{start_idx::Int, end_idx::Int}}()
    block_order = UInt32[]  # labels in order of appearance
    successors = Dict{UInt32, Vector{UInt32}}()
    predecessors = Dict{UInt32, Vector{UInt32}}()
    has_merge = Dict{UInt32, Bool}()  # block already has a merge instruction

    current_label = UInt32(0)
    block_start = 0

    for i in fstart:fend
        inst = mod.insts[i]
        op = opcode(inst)

        if op == OP_LABEL
            current_label = inst.words[2]
            block_start = i
            push!(block_order, current_label)
            successors[current_label] = UInt32[]
            predecessors[current_label] = UInt32[]
            has_merge[current_label] = false
        elseif op == OP_SELECTION_MERGE || op == OP_LOOP_MERGE
            has_merge[current_label] = true
        elseif op == OP_BRANCH && current_label != 0
            target = inst.words[2]
            push!(successors[current_label], target)
            blocks[current_label] = (; start_idx=block_start, end_idx=i)
            current_label = UInt32(0)
        elseif op == OP_BRANCH_COND && current_label != 0
            # OpBranchConditional: words = [opcode|wc, condition, true_label, false_label]
            true_target = inst.words[3]
            false_target = inst.words[4]
            push!(successors[current_label], true_target)
            push!(successors[current_label], false_target)
            blocks[current_label] = (; start_idx=block_start, end_idx=i)
            current_label = UInt32(0)
        elseif op == OP_SWITCH && current_label != 0
            # OpSwitch %selector %default [literal target ...]*
            push!(successors[current_label], inst.words[3])  # default
            j = 4
            while j + 1 <= wordcount(inst)
                push!(successors[current_label], inst.words[j+1])
                j += 2
            end
            blocks[current_label] = (; start_idx=block_start, end_idx=i)
            current_label = UInt32(0)
        elseif (op == UInt16(253) || op == UInt16(254) || op == UInt16(255) || op == OP_FUNCTION_END) && current_label != 0
            blocks[current_label] = (; start_idx=block_start, end_idx=i)
            current_label = UInt32(0)
        end
    end

    # Build predecessor map
    for (label, succs) in successors
        for s in succs
            if haskey(predecessors, s)
                push!(predecessors[s], label)
            end
        end
    end

    isempty(block_order) && return

    entry_label = block_order[1]
    all_labels = Set(block_order)

    # ── Compute FORWARD DOMINATORS ──
    # dom[B] = set of blocks that dominate B
    # Used to detect back-edges: edge (A → B) is a back-edge iff B dominates A
    dom = Dict{UInt32, Set{UInt32}}()
    dom[entry_label] = Set([entry_label])
    for label in block_order[2:end]
        dom[label] = Set(block_order)
    end

    changed = true
    while changed
        changed = false
        for label in block_order[2:end]
            preds = get(predecessors, label, UInt32[])
            valid_preds = filter(p -> p in all_labels && haskey(dom, p), preds)
            isempty(valid_preds) && continue
            new_dom = intersect([dom[p] for p in valid_preds]...)
            push!(new_dom, label)
            if new_dom != dom[label]
                dom[label] = new_dom
                changed = true
            end
        end
    end

    # ── Detect back-edges using dominators ──
    # Edge (src → tgt) is a back-edge if tgt dominates src
    back_edges = Tuple{UInt32, UInt32}[]  # (latch, header)
    for (src, succs) in successors
        for tgt in succs
            if tgt in all_labels && haskey(dom, src) && tgt in dom[src]
                push!(back_edges, (src, tgt))
            end
        end
    end

    # ── Group back-edges by loop header ──
    loop_latches = Dict{UInt32, Vector{UInt32}}()  # header → [latch blocks]
    for (latch, header) in back_edges
        push!(get!(Vector{UInt32}, loop_latches, header), latch)
    end

    # ── Compute natural loop bodies ──
    # The loop body = header + all blocks that can reach a latch without going through header
    loop_bodies = Dict{UInt32, Set{UInt32}}()
    for (header, latches) in loop_latches
        body = Set([header])
        worklist = UInt32[]
        for latch in latches
            if latch != header && latch ∉ body
                push!(body, latch)
                push!(worklist, latch)
            end
        end
        while !isempty(worklist)
            node = pop!(worklist)
            for pred in get(predecessors, node, UInt32[])
                if pred ∉ body && pred in all_labels
                    push!(body, pred)
                    push!(worklist, pred)
                end
            end
        end
        loop_bodies[header] = body
    end

    # ── Compute POST-DOMINATORS ──
    exit_labels = UInt32[]
    for label in block_order
        if haskey(blocks, label)
            term_idx = blocks[label].end_idx
            term_op = opcode(mod.insts[term_idx])
            if term_op == UInt16(253) || term_op == UInt16(254) || term_op == UInt16(255)
                push!(exit_labels, label)
            elseif isempty(filter(s -> s in all_labels, get(successors, label, UInt32[])))
                push!(exit_labels, label)
            end
        end
    end

    isempty(exit_labels) && return

    pdom_sets = Dict{UInt32, Set{UInt32}}()
    for label in block_order
        pdom_sets[label] = Set(block_order)
    end
    for exit_label in exit_labels
        pdom_sets[exit_label] = Set([exit_label])
    end

    changed = true
    iterations = 0
    while changed && iterations < 100
        changed = false
        iterations += 1
        for label in reverse(block_order)
            label in exit_labels && continue
            succs = get(successors, label, UInt32[])
            valid_succs = filter(s -> s in all_labels, succs)
            isempty(valid_succs) && continue
            new_set = intersect([pdom_sets[s] for s in valid_succs]...)
            push!(new_set, label)
            if new_set != pdom_sets[label]
                pdom_sets[label] = new_set
                changed = true
            end
        end
    end

    # Extract immediate post-dominator: closest post-dominator (largest pdom_set)
    ipdom = Dict{UInt32, UInt32}()
    for label in block_order
        pdoms = setdiff(pdom_sets[label], Set([label]))
        if !isempty(pdoms)
            best = first(pdoms)
            best_size = length(pdom_sets[best])
            for p in pdoms
                if length(pdom_sets[p]) > best_size
                    best = p
                    best_size = length(pdom_sets[p])
                end
            end
            ipdom[label] = best
        end
    end

    # DEBUG: dump ipdom and block names
    _dbg_names = Dict{UInt32, String}()
    for inst in mod.insts
        if opcode(inst) == UInt16(5) && wordcount(inst) >= 3  # OpName
            _dbg_names[inst.words[2]] = String(reinterpret(UInt8, inst.words[3:end]))
        end
    end
    _dbg_name(id) = get(_dbg_names, id, "?$(id)")

    # ── Scan existing OpLoopMerge instructions for actual continue targets ──
    # The LLVM SPIR-V backend may already have placed OpLoopMerge instructions.
    # The continue target declared there may differ from the latch blocks found
    # by back-edge analysis. We must use the ACTUAL continue targets.
    existing_loop_merges = Dict{UInt32, @NamedTuple{merge_target::UInt32, continue_target::UInt32}}()
    for label in block_order
        haskey(blocks, label) || continue
        blk = blocks[label]
        for i in blk.start_idx:blk.end_idx
            inst = mod.insts[i]
            if opcode(inst) == OP_LOOP_MERGE && wordcount(inst) >= 4
                existing_loop_merges[label] = (merge_target=inst.words[2], continue_target=inst.words[3])
                break
            end
        end
    end

    # ── Collect continue targets (used by loop merges) ──
    # SPIR-V allows bare OpBranchConditional in continue targets without
    # OpSelectionMerge. For loops with existing OpLoopMerge, use the declared
    # continue target. For loops without, use back-edge analysis with an exit
    # check: a latch that conditionally exits the loop body needs
    # OpSelectionMerge and should NOT be treated as a continue target.
    continue_targets = Set{UInt32}()
    for (header, latches) in loop_latches
        if haskey(existing_loop_merges, header)
            # Use the actual continue target from the existing OpLoopMerge
            push!(continue_targets, existing_loop_merges[header].continue_target)
        else
            # Pick the latch that will become the continue target (same logic
            # as used later in the loop merge insertion). This latch is always
            # added to continue_targets — even if it conditionally exits the
            # loop — because the SPIR-V spec explicitly allows a continue
            # target to have a bare OpBranchConditional (branching to the loop
            # merge for break, and to the header for back-edge).
            chosen_latch = if length(latches) == 1
                latches[1]
            else
                latches[argmax(l -> something(findfirst(==(l), block_order), 0), latches)]
            end
            push!(continue_targets, chosen_latch)
            # Non-chosen latches that don't conditionally exit are also continue-like
            body = get(loop_bodies, header, Set{UInt32}())
            for l in latches
                l == chosen_latch && continue
                l_succs = get(successors, l, UInt32[])
                l_term_op = haskey(blocks, l) ? opcode(mod.insts[blocks[l].end_idx]) : UInt16(0)
                if !(l_term_op == OP_BRANCH_COND && any(s -> s ∉ body, l_succs))
                    push!(continue_targets, l)
                end
            end
        end
    end

    # ── Build loop membership: for each block, which loops contain it? ──
    # Used to clamp selection merge targets to stay within the innermost loop.
    # A selection inside a loop body must have its merge target within that loop
    # body (or be the loop's own merge target).

    # ── Fix existing incorrect selection merge targets ──
    # The LLVM SPIR-V backend (StructurizeCFG) may produce selection merges with
    # incorrect targets (e.g., shifted by one ipdom step). Verify existing selection
    # merges against computed ipdom and fix any mismatches.
    for label in block_order
        haskey(blocks, label) || continue
        has_merge[label] || continue
        haskey(loop_latches, label) && continue

        correct_merge = get(ipdom, label, UInt32(0))
        correct_merge == 0 && continue

        # Apply loop body constraint (same as for new merges below)
        for (loop_header, body) in loop_bodies
            label in body && label != loop_header || continue
            loop_continue = UInt32(0)
            loop_merge_target = UInt32(0)
            if haskey(existing_loop_merges, loop_header)
                loop_continue = existing_loop_merges[loop_header].continue_target
                loop_merge_target = existing_loop_merges[loop_header].merge_target
            else
                for (lh, latches) in loop_latches
                    if lh == loop_header
                        loop_continue = length(latches) == 1 ? latches[1] : latches[argmax(l -> something(findfirst(==(l), block_order), 0), latches)]
                        break
                    end
                end
            end
            if correct_merge == loop_continue
                correct_merge = UInt32(0)  # skip fixing — trampoline logic is complex
            elseif correct_merge ∉ body && correct_merge != loop_merge_target
                correct_merge = UInt32(0)  # outside body and not loop merge — skip
            end
            break
        end
        correct_merge == 0 && continue

        # Find and fix the existing OpSelectionMerge in this block
        blk = blocks[label]
        for i in blk.start_idx:blk.end_idx
            inst = mod.insts[i]
            if opcode(inst) == OP_SELECTION_MERGE && wordcount(inst) >= 3
                if inst.words[2] != correct_merge
                    inst.words[2] = correct_merge
                end
                break
            end
        end
    end

    # ── Insert merge instructions ──
    insertions = Tuple{Int, SPVInst}[]

    # Track which continue targets have already been trampolined.
    # When two blocks in the same loop both need the trampoline (e.g. block X
    # whose ipdom is the continue target, and Flow23/latch whose ipdom is the
    # loop merge), the second invocation must reuse the first's new continue ID
    # instead of creating another (unreachable) block.
    trampolined_continues = Dict{UInt32, UInt32}()  # original_continue → new_continue_id

    for label in block_order
        haskey(blocks, label) || continue
        has_merge[label] && continue

        blk = blocks[label]
        term_idx = blk.end_idx
        term_op = opcode(mod.insts[term_idx])

        is_loop_header = haskey(loop_latches, label)

        if is_loop_header
            # Loop header — insert OpLoopMerge before terminator
            # (terminator can be OpBranch, OpBranchConditional, or OpSwitch)
            (term_op == OP_BRANCH || term_op == OP_BRANCH_COND || term_op == OP_SWITCH) || continue

            # Merge target = ipdom that is OUTSIDE the loop body
            merge_target = get(ipdom, label, UInt32(0))
            merge_target == 0 && continue
            body = loop_bodies[label]
            safety = 0
            while merge_target in body && haskey(ipdom, merge_target) && safety < 50
                merge_target = ipdom[merge_target]
                safety += 1
            end

            # Continue target = latch block (source of back-edge to this header)
            latches = loop_latches[label]
            if length(latches) == 1
                continue_target = latches[1]
            else
                # Multiple latches: pick the one that appears last in block order
                continue_target = latches[1]
                best_pos = 0
                for l in latches
                    pos = findfirst(==(l), block_order)
                    if pos !== nothing && pos > best_pos
                        continue_target = l
                        best_pos = pos
                    end
                end
            end

            merge_inst = SPVInst(OP_LOOP_MERGE, merge_target, continue_target, UInt32(0))
            push!(insertions, (term_idx, merge_inst))

        elseif term_op == OP_BRANCH_COND || term_op == OP_SWITCH
            # Skip adding OpSelectionMerge to continue targets.
            # SPIR-V spec 2.11: "If the block has an OpBranchConditional as its
            # terminator, then either it must be a Continue target of a structured
            # loop or it must be immediately preceded by an OpSelectionMerge."
            # Continue targets are allowed to have bare OpBranchConditional.
            label in continue_targets && continue

            # Selection header — insert OpSelectionMerge
            merge_target = get(ipdom, label, UInt32(0))
            merge_target == 0 && continue

            # Check if this block is inside any loop body (not as the header).
            # If so, constrain the merge target:
            #  - Must be within the loop body
            #  - Must NOT be the loop's continue target (continue starts a
            #    separate construct; SPIR-V forbids selection merges pointing there)
            # If the ipdom violates either rule, insert a trampoline block.
            needs_trampoline = false
            for (loop_header, body) in loop_bodies
                label in body && label != loop_header || continue

                # Find this loop's continue target and merge target
                loop_continue = UInt32(0)
                loop_merge = UInt32(0)
                if haskey(existing_loop_merges, loop_header)
                    loop_continue = existing_loop_merges[loop_header].continue_target
                    loop_merge = existing_loop_merges[loop_header].merge_target
                else
                    for (lh, latches) in loop_latches
                        if lh == loop_header
                            loop_continue = length(latches) == 1 ? latches[1] : latches[argmax(l -> something(findfirst(==(l), block_order), 0), latches)]
                            break
                        end
                    end
                    lm = get(ipdom, loop_header, UInt32(0))
                    safety = 0
                    while lm in body && haskey(ipdom, lm) && safety < 50
                        lm = ipdom[lm]
                        safety += 1
                    end
                    loop_merge = lm
                end

                # Merge target constraints:
                #  - Must NOT be the continue target (can't overlap constructs)
                #  - Must NOT be the loop's merge target (would duplicate merge)
                #  - If outside the loop body, redirect to continue target + trampoline
                if merge_target == loop_continue
                    needs_trampoline = true
                    merge_target = loop_continue
                elseif merge_target == loop_merge
                    # "Conditional break" pattern: selection branches to loop merge
                    # (break) and back to header (continue). Can't reuse loop merge
                    # as selection merge — use the continue target instead.
                    needs_trampoline = true
                    merge_target = loop_continue
                elseif merge_target ∉ body && merge_target != loop_merge
                    needs_trampoline = true
                    merge_target = loop_continue
                end
                break  # innermost loop only
            end

            if needs_trampoline && merge_target != 0
                original_continue = merge_target  # = the continue target we can't merge to

                if haskey(trampolined_continues, original_continue)
                    # This continue target was already trampolined by a prior block
                    # in the same loop. Reuse the existing new continue ID — don't
                    # create another (unreachable) block.
                    new_continue_id = trampolined_continues[original_continue]
                    sel_merge_target = (label == original_continue) ? new_continue_id : original_continue
                    merge_inst = SPVInst(OP_SELECTION_MERGE, sel_merge_target, UInt32(0))
                    push!(insertions, (term_idx, merge_inst))
                else
                    # First trampoline for this continue target.
                    # Split the continue target by inserting a NEW block as the
                    # actual continue target. The original continue target then
                    # becomes a regular loop body block and CAN be used as the
                    # selection merge.
                    #
                    # Before: header → ... → original_continue → header
                    # After:  header → ... → original_continue → NEW(continue) → header
                    new_continue_id = alloc_id!(mod)
                    trampolined_continues[original_continue] = new_continue_id

                    # Find the OpLoopMerge that references this continue target and update it
                    for inst in mod.insts
                        if opcode(inst) == OP_LOOP_MERGE && wordcount(inst) >= 4
                            if inst.words[3] == original_continue
                                inst.words[3] = new_continue_id
                            end
                        end
                    end

                    # Also update any already-queued OpLoopMerge insertions
                    for k in eachindex(insertions)
                        mi = insertions[k][2]
                        if opcode(mi) == OP_LOOP_MERGE && wordcount(mi) >= 4 && mi.words[3] == original_continue
                            mi.words[3] = new_continue_id
                        end
                    end

                    # Find the loop header for this continue target
                    this_loop_header = UInt32(0)
                    for (lh, latches) in loop_latches
                        if original_continue in latches
                            this_loop_header = lh
                            break
                        end
                    end

                    if this_loop_header != 0
                        # Find the end of the original_continue block
                        cont_blk = get(blocks, original_continue, nothing)
                        if cont_blk !== nothing
                            cont_end = cont_blk.end_idx
                            cont_term = mod.insts[cont_end]
                            ct_op = opcode(cont_term)

                            # Redirect back-edge: original_continue → header  becomes
                            #                     original_continue → new_continue
                            if ct_op == OP_BRANCH_COND
                                for j in 3:min(4, wordcount(cont_term))
                                    if cont_term.words[j] == this_loop_header
                                        cont_term.words[j] = new_continue_id
                                    end
                                end
                            elseif ct_op == OP_BRANCH
                                if cont_term.words[2] == this_loop_header
                                    cont_term.words[2] = new_continue_id
                                end
                            end

                            # Insert new continue block right after original_continue
                            new_insts = [SPVInst(OP_LABEL, new_continue_id), SPVInst(OP_BRANCH, this_loop_header)]
                            splice!(mod.insts, cont_end+1:cont_end, new_insts)

                            # Update OpPhi instructions in the loop header:
                            # original_continue used to branch to header, now new_continue does
                            in_header = false
                            for inst in mod.insts
                                op = opcode(inst)
                                if op == OP_LABEL && wordcount(inst) >= 2
                                    in_header = (inst.words[2] == this_loop_header)
                                elseif in_header && op == OP_PHI
                                    for j in 5:2:wordcount(inst)
                                        if inst.words[j] == original_continue
                                            inst.words[j] = new_continue_id
                                        end
                                    end
                                elseif in_header && op != OP_PHI
                                    in_header = false
                                end
                            end

                            # Adjust indices — the splice shifted everything after cont_end by 2
                            for k in eachindex(insertions)
                                if insertions[k][1] > cont_end
                                    insertions[k] = (insertions[k][1] + 2, insertions[k][2])
                                end
                            end
                            if term_idx > cont_end
                                term_idx += 2
                            end
                            for (blk_label, blk_range) in blocks
                                new_s = blk_range.start_idx > cont_end ? blk_range.start_idx + 2 : blk_range.start_idx
                                new_e = blk_range.end_idx > cont_end ? blk_range.end_idx + 2 : blk_range.end_idx
                                if new_s != blk_range.start_idx || new_e != blk_range.end_idx
                                    blocks[blk_label] = (start_idx=new_s, end_idx=new_e)
                                end
                            end
                        end
                    end

                    sel_merge_target = (label == original_continue) ? new_continue_id : original_continue
                    merge_inst = SPVInst(OP_SELECTION_MERGE, sel_merge_target, UInt32(0))
                    push!(insertions, (term_idx, merge_inst))
                end
            else
                merge_target == 0 && continue
                merge_inst = SPVInst(OP_SELECTION_MERGE, merge_target, UInt32(0))
                push!(insertions, (term_idx, merge_inst))
            end
        end
    end

    # Sort insertions by index in reverse order and apply
    sort!(insertions; by=first, rev=true)
    for (idx, merge_inst) in insertions
        insert!(mod.insts, idx, merge_inst)
    end
end

# ── 8. Fix merge placement ───────────────────────────────────────────────
# The LLVM SPIR-V backend at -O0 may emit instructions between
# OpSelectionMerge/OpLoopMerge and the subsequent branch. SPIR-V requires
# the merge to immediately precede the branch. Fix: reorder.

function fix_merge_placement!(mod::SPVModule)
    is_merge(op) = op == OP_SELECTION_MERGE || op == OP_LOOP_MERGE
    is_branch(op) = op == OP_BRANCH || op == OP_BRANCH_COND || op == OP_SWITCH

    new_insts = SPVInst[]
    i = 1
    while i <= length(mod.insts)
        inst = mod.insts[i]
        if is_merge(opcode(inst))
            # Collect instructions until the next branch
            merge_inst = inst
            between = SPVInst[]
            j = i + 1
            while j <= length(mod.insts) && !is_branch(opcode(mod.insts[j]))
                push!(between, mod.insts[j])
                j += 1
            end
            if j <= length(mod.insts) && is_branch(opcode(mod.insts[j]))
                # Emit: intervening instructions, then merge, then branch
                append!(new_insts, between)
                push!(new_insts, merge_inst)
                push!(new_insts, mod.insts[j])
                i = j + 1
                continue
            end
            # No branch found — emit as-is (shouldn't happen in valid SPIR-V)
            push!(new_insts, merge_inst)
            append!(new_insts, between)
            i = j
        else
            push!(new_insts, inst)
            i += 1
        end
    end

    empty!(mod.insts)
    append!(mod.insts, new_insts)
end

# ── 8b. Remove dead (orphan) blocks ──────────────────────────────────────
# The trampoline insertion + merge placement reordering can leave orphan blocks
# in the instruction stream: blocks whose OpLabel is unreachable (no branch,
# conditional, or switch targets them). These orphan blocks still have outgoing
# edges (OpBranch) which create phantom predecessors in downstream blocks,
# causing OpPhi predecessor-count mismatches in spirv-val.
#
# Fix: within each function, compute the set of branch targets and remove any
# block (OpLabel … next OpLabel) that is not the entry block and not a target.

function fix_dead_blocks!(mod::SPVModule)
    # Collect function boundaries: each function starts at OpFunction and ends at OpFunctionEnd
    func_ranges = UnitRange{Int}[]
    func_start = 0
    for (i, inst) in enumerate(mod.insts)
        op = opcode(inst)
        if op == OP_FUNCTION
            func_start = i
        elseif op == OP_FUNCTION_END && func_start > 0
            push!(func_ranges, func_start:i)
            func_start = 0
        end
    end

    dead_indices = Set{Int}()

    for frange in func_ranges
        # Find all block labels and their index ranges within this function
        block_starts = Int[]  # indices of OpLabel instructions
        block_labels = UInt32[]
        for i in frange
            if opcode(mod.insts[i]) == OP_LABEL && wordcount(mod.insts[i]) >= 2
                push!(block_starts, i)
                push!(block_labels, mod.insts[i].words[2])
            end
        end
        isempty(block_labels) && continue

        # Entry block = first block in the function (right after OpFunction + OpFunctionParameter*)
        entry_label = block_labels[1]

        # Collect all branch targets within this function
        targets = Set{UInt32}()
        for i in frange
            inst = mod.insts[i]
            op = opcode(inst)
            if op == OP_BRANCH && wordcount(inst) >= 2
                push!(targets, inst.words[2])
            elseif op == OP_BRANCH_COND && wordcount(inst) >= 4
                push!(targets, inst.words[3])
                push!(targets, inst.words[4])
            elseif op == OP_SWITCH
                # Default target at word 3, then (literal, label) pairs
                wordcount(inst) >= 3 && push!(targets, inst.words[3])
                for j in 5:2:wordcount(inst)
                    push!(targets, inst.words[j])
                end
            elseif op == OP_SELECTION_MERGE && wordcount(inst) >= 3
                push!(targets, inst.words[2])
            elseif op == OP_LOOP_MERGE && wordcount(inst) >= 4
                push!(targets, inst.words[2])
                push!(targets, inst.words[3])
            end
        end

        # Mark dead blocks: not the entry, not a branch target
        for (bi, label) in enumerate(block_labels)
            label == entry_label && continue
            label in targets && continue

            # This block is dead — mark all instructions from its OpLabel
            # to just before the next OpLabel (or end of function)
            blk_start = block_starts[bi]
            blk_end = bi < length(block_starts) ? block_starts[bi+1] - 1 : last(frange) - 1  # -1 for OpFunctionEnd
            for i in blk_start:blk_end
                push!(dead_indices, i)
            end
        end
    end

    if !isempty(dead_indices)
        new_insts = SPVInst[]
        for (i, inst) in enumerate(mod.insts)
            i in dead_indices || push!(new_insts, inst)
        end
        empty!(mod.insts)
        append!(mod.insts, new_insts)
        @debug "fix_dead_blocks!: removed $(length(dead_indices)) instructions from dead blocks"
    end
end

# ── 9. Fix barrier semantics ─────────────────────────────────────────────
# OpControlBarrier with SequentiallyConsistent (0x10) is not valid in Vulkan
# [VUID-StandaloneSpirv-MemorySemantics-10866]. Replace with
# AcquireRelease|WorkgroupMemory (0x108).
#
# The semantics constant may be shared with other uses (e.g. `lid < 16`),
# so we create a new constant rather than modifying the existing one.

function fix_barrier_semantics!(mod::SPVModule)
    # Find OpControlBarrier semantics IDs
    barrier_sem_ids = Set{UInt32}()
    for inst in mod.insts
        if opcode(inst) == OP_CONTROL_BARRIER && wordcount(inst) == 4
            push!(barrier_sem_ids, inst.words[4])
        end
    end
    isempty(barrier_sem_ids) && return

    # Check if any referenced constant has SequentiallyConsistent
    sem_type_id = UInt32(0)
    needs_fix = false
    for inst in mod.insts
        if opcode(inst) == OP_CONSTANT && wordcount(inst) >= 4
            if inst.words[3] in barrier_sem_ids
                sem_type_id = inst.words[2]
                if inst.words[4] & MEM_SEQUENTIALLY_CONSISTENT != 0
                    needs_fix = true
                end
            end
        end
    end
    !needs_fix && return

    # Create new constant with correct semantics
    new_id = alloc_id!(mod)
    new_value = MEM_ACQUIRE_RELEASE | MEM_WORKGROUP_MEMORY
    new_const = SPVInst(OP_CONSTANT, sem_type_id, new_id, new_value)

    # Insert before first OpFunction
    idx = findfirst(inst -> opcode(inst) == OP_FUNCTION, mod.insts)
    idx !== nothing && insert!(mod.insts, idx, new_const)

    # Update all barriers to use the new constant
    for inst in mod.insts
        if opcode(inst) == OP_CONTROL_BARRIER && wordcount(inst) == 4
            inst.words[4] = new_id
        end
    end
end

# ── 10. Fix duplicate merge targets ──────────────────────────────────────
# SPIR-V requires each block to be the merge target of at most one header.
# StructurizeCFG + the LLVM SPIR-V backend can produce cases where both
# OpSelectionMerge and OpLoopMerge target the same block.
#
# Fix: redirect the second merge to a new trampoline block (OpLabel +
# OpBranch to original target), inserted before OpFunctionEnd.

function fix_duplicate_merge_targets!(mod::SPVModule)
    # Count merge references per block
    merge_counts = Dict{UInt32, Int}()
    for inst in mod.insts
        op = opcode(inst)
        if (op == OP_SELECTION_MERGE || op == OP_LOOP_MERGE) && wordcount(inst) >= 3
            target = inst.words[2]
            merge_counts[target] = get(merge_counts, target, 0) + 1
        end
    end

    duplicates = Set(k for (k, v) in merge_counts if v > 1)
    isempty(duplicates) && return

    # Redirect second+ merge references to new trampoline IDs
    seen = Set{UInt32}()
    trampolines = Dict{UInt32, Vector{UInt32}}()  # original → [new_ids...]

    for inst in mod.insts
        op = opcode(inst)
        if (op == OP_SELECTION_MERGE || op == OP_LOOP_MERGE) && wordcount(inst) >= 3
            target = inst.words[2]
            if target in duplicates
                if target in seen
                    new_id = alloc_id!(mod)
                    inst.words[2] = new_id
                    push!(get!(trampolines, target, UInt32[]), new_id)
                else
                    push!(seen, target)
                end
            end
        end
    end

    # Redirect branches inside each redirected construct
    for (orig, new_ids) in trampolines, new_id in new_ids
        inside = false
        for inst in mod.insts
            op = opcode(inst)
            if (op == OP_LOOP_MERGE || op == OP_SELECTION_MERGE) &&
                    wordcount(inst) >= 3 && inst.words[2] == new_id
                inside = true
            end
            if inside
                # Stop at the (not yet existing) trampoline label
                op == OP_LABEL && wordcount(inst) == 2 && inst.words[2] == new_id && break
                if op == OP_BRANCH_COND && wordcount(inst) >= 4
                    for j in 3:4
                        inst.words[j] == orig && (inst.words[j] = new_id)
                    end
                elseif op == OP_BRANCH && wordcount(inst) == 2
                    inst.words[2] == orig && (inst.words[2] = new_id)
                end
            end
        end
    end

    # Update OpPhi predecessors in original merge blocks
    for (orig, new_ids) in trampolines, new_id in new_ids
        # Find the loop header containing the merge targeting new_id
        loop_header_id = UInt32(0)
        for (idx, inst) in enumerate(mod.insts)
            if opcode(inst) == OP_LOOP_MERGE && wordcount(inst) >= 4 && inst.words[2] == new_id
                for k in (idx-1):-1:1
                    if opcode(mod.insts[k]) == OP_LABEL && wordcount(mod.insts[k]) == 2
                        loop_header_id = mod.insts[k].words[2]
                        break
                    end
                end
                break
            end
        end
        loop_header_id == 0 && continue

        # In the original target block, update OpPhi: predecessor header → trampoline
        in_block = false
        for inst in mod.insts
            op = opcode(inst)
            if op == OP_LABEL && wordcount(inst) == 2
                in_block = (inst.words[2] == orig)
            elseif in_block && op == OP_PHI
                for j in 5:2:wordcount(inst)
                    inst.words[j] == loop_header_id && (inst.words[j] = new_id)
                end
            elseif in_block && op != OP_PHI
                in_block = false
            end
        end
    end

    # Insert each trampoline block right before the original target block it
    # branches to. This ensures the trampoline appears before its target in
    # the instruction stream, satisfying SPIR-V's dominator ordering requirement.
    # Process in reverse to keep indices stable during insertion.
    for (target, new_ids) in trampolines
        # Find the OpLabel for the original target block
        target_idx = findfirst(inst -> opcode(inst) == OP_LABEL && wordcount(inst) == 2 &&
                                       inst.words[2] == target, mod.insts)
        target_idx === nothing && continue
        for new_id in reverse(new_ids)
            tramp_insts = [SPVInst(OP_LABEL, new_id), SPVInst(OP_BRANCH, target)]
            splice!(mod.insts, target_idx:target_idx-1, tramp_insts)
        end
    end
end

# ── 10b. Merge single-predecessor blocks ─────────────────────────────────
# When block A ends with OpBranch to B, and B has exactly one predecessor (A),
# merge B into A. This is equivalent to spirv-opt's --merge-blocks pass.
# Needed because StructurizeCFG creates intermediate "Flow" blocks that can
# be merged with their successors. For example, merging a %Flow block with
# the continue target %L629_i makes %Flow itself the continue target,
# allowing the loop header to branch to body/continue (valid SPIR-V) instead
# of body/intermediate-block (which spirv-val rejects as unstructured).
function fix_merge_single_pred_blocks!(mod::SPVModule)
    changed = true
    while changed
        changed = false

        # Build predecessor counts: for each label, how many blocks branch to it
        pred_count = Dict{UInt32, Int}()
        for inst in mod.insts
            op = opcode(inst)
            if op == OP_BRANCH && wordcount(inst) >= 2
                target = inst.words[2]
                pred_count[target] = get(pred_count, target, 0) + 1
            elseif op == OP_BRANCH_COND && wordcount(inst) >= 4
                for j in 3:min(4, wordcount(inst))
                    target = inst.words[j]
                    pred_count[target] = get(pred_count, target, 0) + 1
                end
            elseif op == OP_SWITCH
                # Switch default + cases
                for j in 2:wordcount(inst)
                    if isodd(j) || j == 2  # default at word[2], targets at even indices
                        target = inst.words[j]
                        pred_count[target] = get(pred_count, target, 0) + 1
                    end
                end
            end
        end

        # Find a mergeable pair: A ends with OpBranch B, B has exactly 1 predecessor
        for i in 1:length(mod.insts)
            inst = mod.insts[i]
            opcode(inst) == OP_BRANCH && wordcount(inst) >= 2 || continue

            target_label = inst.words[2]
            get(pred_count, target_label, 0) == 1 || continue

            # Find the source block's label (walk back to OpLabel)
            src_label = UInt32(0)
            for j in i-1:-1:1
                if opcode(mod.insts[j]) == OP_LABEL && wordcount(mod.insts[j]) >= 2
                    src_label = mod.insts[j].words[2]
                    break
                end
            end
            src_label == 0 && continue

            # Don't merge if source == target (degenerate)
            src_label == target_label && continue

            # Don't merge if A has a merge instruction right before the OpBranch.
            # The OpBranch after OpLoopMerge/OpSelectionMerge is structurally
            # required and removing it would break the merge placement.
            if i >= 2
                prev_op = opcode(mod.insts[i - 1])
                (prev_op == OP_LOOP_MERGE || prev_op == OP_SELECTION_MERGE) && continue
            end

            # Target block must be immediately after A (adjacent blocks only)
            i + 1 <= length(mod.insts) || continue
            next_inst = mod.insts[i + 1]
            (opcode(next_inst) == OP_LABEL && wordcount(next_inst) >= 2 &&
             next_inst.words[2] == target_label) || continue
            target_label_idx = i + 1

            # Remove the OpBranch from A (index i)
            deleteat!(mod.insts, i)

            # Adjust target_label_idx
            if target_label_idx > i
                target_label_idx -= 1
            end

            # Remove B's OpLabel
            deleteat!(mod.insts, target_label_idx)

            # Remove any OpPhi instructions at the start of B (single pred = no phi needed)
            # Replace phi results with the value from the single predecessor
            while target_label_idx <= length(mod.insts)
                pinst = mod.insts[target_label_idx]
                opcode(pinst) == OP_PHI || break
                # OpPhi: result_type result_id (value parent)...
                # Single predecessor: just one (value, parent) pair
                # Replace all uses of phi result with the value
                phi_result = pinst.words[2]
                phi_value = pinst.words[3]  # the value from the single predecessor
                # Replace all uses of phi_result with phi_value
                for k in 1:length(mod.insts)
                    kinst = mod.insts[k]
                    for w in 1:wordcount(kinst)
                        if kinst.words[w] == phi_result
                            # Don't replace the definition itself (result ID)
                            # Result ID is word[2] for most instructions
                            (w == 2 && k == target_label_idx) && continue
                            kinst.words[w] = phi_value
                        end
                    end
                end
                deleteat!(mod.insts, target_label_idx)
            end

            # Replace all references to target_label with src_label
            for inst in mod.insts
                for w in 1:wordcount(inst)
                    if inst.words[w] == target_label
                        inst.words[w] = src_label
                    end
                end
            end

            changed = true
            break  # restart scan after modification
        end
    end
end

# ── 11. Fix PushConstant bitcast (post spirv-opt) ────────────────────────
# spirv-opt may optimize `OpAccessChain %ptr %struct %0` into
# `OpBitcast %ptr %struct` for the first member. RADV crashes on this.

function fix_pushconstant_bitcast!(mod::SPVModule)
    pc_var_id = UInt32(0)
    uint32_type_id = UInt32(0)
    zero_const_id = UInt32(0)

    for inst in mod.insts
        op = opcode(inst)
        if op == OP_VARIABLE && wordcount(inst) >= 4 && inst.words[4] == SC_PUSH_CONSTANT
            pc_var_id = inst.words[3]
        elseif op == OP_TYPE_INT && wordcount(inst) == 4 && inst.words[3] == 32 && inst.words[4] == 0
            uint32_type_id = inst.words[2]
        elseif op == OP_CONSTANT && wordcount(inst) == 4 && uint32_type_id != 0
            if inst.words[2] == uint32_type_id && inst.words[4] == 0
                zero_const_id = inst.words[3]
            end
        end
    end

    pc_var_id == 0 && return

    # Create uint32 type and zero constant if they don't exist
    if uint32_type_id == 0
        uint32_type_id = alloc_id!(mod)
        idx = findfirst(inst -> opcode(inst) == OP_FUNCTION, mod.insts)
        idx !== nothing && insert!(mod.insts, idx, SPVInst(OP_TYPE_INT, uint32_type_id, UInt32(32), UInt32(0)))
    end
    if zero_const_id == 0
        zero_const_id = alloc_id!(mod)
        idx = findfirst(inst -> opcode(inst) == OP_FUNCTION, mod.insts)
        idx !== nothing && insert!(mod.insts, idx, SPVInst(OP_CONSTANT, uint32_type_id, zero_const_id, UInt32(0)))
    end

    for (i, inst) in enumerate(mod.insts)
        if opcode(inst) == OP_BITCAST && wordcount(inst) == 4 && inst.words[4] == pc_var_id
            mod.insts[i] = SPVInst(UInt32[
                (UInt32(5) << 16) | UInt32(OP_ACCESS_CHAIN),
                inst.words[2],   # result_type
                inst.words[3],   # result_id
                pc_var_id,
                zero_const_id
            ])
        end
    end
end


# ── OpOrdered/OpUnordered → OpIsNan lowering ─────────────────────────────
#
# SPIR-V Shader model does not support OpOrdered/OpUnordered (requires Kernel
# capability). Julia generates these from `fcmp ord`/`fcmp uno` for isnan checks.
#
# Following clspv's approach (SPIRVProducerPass.cpp:6073-6119), we lower:
#   OpOrdered   %bool %x %y  →  NOT (IsNan(%x) OR IsNan(%y))
#   OpUnordered %bool %x %y  →  IsNan(%x) OR IsNan(%y)
#
# OpIsNan is available in Shader model (no extra capability needed).

function fix_ordered_unordered!(mod::SPVModule)
    # Find all OpOrdered/OpUnordered instructions
    indices = Int[]
    for (i, inst) in enumerate(mod.insts)
        op = opcode(inst)
        if op == OP_ORDERED || op == OP_UNORDERED
            push!(indices, i)
        end
    end
    isempty(indices) && return

    # Process in reverse order so insertion indices stay valid
    for idx in reverse(indices)
        inst = mod.insts[idx]
        op = opcode(inst)
        # OpOrdered/OpUnordered format: [header, result_type, result_id, operand1, operand2]
        result_type = inst.words[2]  # %bool
        result_id   = inst.words[3]
        op_x        = inst.words[4]
        op_y        = inst.words[5]

        new_insts = SPVInst[]

        # OpIsNan %bool %x
        isnan_x_id = alloc_id!(mod)
        push!(new_insts, SPVInst(OP_IS_NAN, result_type, isnan_x_id, op_x))

        # OpIsNan %bool %y
        isnan_y_id = alloc_id!(mod)
        push!(new_insts, SPVInst(OP_IS_NAN, result_type, isnan_y_id, op_y))

        # OpLogicalOr %bool isnan_x isnan_y  (= "either is NaN" = unordered)
        or_id = alloc_id!(mod)
        push!(new_insts, SPVInst(OP_LOGICAL_OR, result_type, or_id, isnan_x_id, isnan_y_id))

        if op == OP_ORDERED
            # OpOrdered = NOT unordered
            not_id = alloc_id!(mod)
            push!(new_insts, SPVInst(OP_LOGICAL_NOT, result_type, not_id, or_id))
            # Rewrite: original result_id now comes from the NOT
            # We replace the instruction and patch references
            # Simpler: give the NOT the original result_id
            new_insts[end] = SPVInst(OP_LOGICAL_NOT, result_type, result_id, or_id)
        else
            # OpUnordered: result is the OR directly
            # Give the OR the original result_id
            new_insts[end] = SPVInst(OP_LOGICAL_OR, result_type, result_id, isnan_x_id, isnan_y_id)
        end

        # Replace the original instruction with the new sequence
        splice!(mod.insts, idx, new_insts)
    end
end


# ── spirv-opt wrapper ────────────────────────────────────────────────────

function _spirv_opt(spv_bytes::Vector{UInt8})
    mktempdir() do dir
        in_path = joinpath(dir, "in.spv")
        out_path = joinpath(dir, "out.spv")
        write(in_path, spv_bytes)
        # Debug: save pre-opt SPIR-V
        _cnt = Ref(0); _cnt[] += 1
        cp(in_path, "/tmp/spirv_debug_preopt.spv"; force=true)
        err = IOBuffer()
        ok = success(pipeline(`$(SPIRV_Tools_jll.spirv_opt()) --target-env=vulkan1.3
             --eliminate-dead-functions
             -o $out_path $in_path`; stderr=err))
        if !ok
            error("spirv-opt failed:\n", String(take!(err)))
        end
        return read(out_path)
    end
end


# ── Eliminate dead functions ─────────────────────────────────────────────
# Julia's GPU codegen can emit dead runtime stubs (ijl_box_int64, etc.)
# that contain invalid SPIR-V (function-type pointers, ConvertUToPtr on
# non-PSB pointers). spirv-opt validates before eliminating, so we must
# remove them first. Keeps the entry function and anything reachable from it.
const OP_TYPE_FUNCTION   = UInt16(33)
const OP_CONSTANT_NULL   = UInt16(46)
const OP_CONSTANT_COMPOSITE = UInt16(44)
const OP_FUNCTION_CALL   = UInt16(57)
const OP_RETURN          = UInt16(253)
const OP_RETURN_VALUE    = UInt16(254)
const OP_UNREACHABLE     = UInt16(255)

function fix_dead_functions!(mod::SPVModule)
    # 1. Find the entry point function ID
    entry_func_id = UInt32(0)
    for inst in mod.insts
        if opcode(inst) == OP_ENTRY_POINT && wordcount(inst) >= 3
            entry_func_id = inst.words[3]
            break
        end
    end
    entry_func_id == 0 && return

    # 2. Parse function boundaries and call graph
    # Map function_id → (start_index, end_index) in mod.insts
    func_ranges = Dict{UInt32, Tuple{Int,Int}}()
    # Map function_id → set of called function IDs
    call_graph = Dict{UInt32, Set{UInt32}}()

    current_func_id = UInt32(0)
    current_func_start = 0
    for (i, inst) in enumerate(mod.insts)
        op = opcode(inst)
        if op == OP_FUNCTION && wordcount(inst) >= 3
            current_func_id = inst.words[3]
            current_func_start = i
            call_graph[current_func_id] = Set{UInt32}()
        elseif op == OP_FUNCTION_END && current_func_id != 0
            func_ranges[current_func_id] = (current_func_start, i)
            current_func_id = UInt32(0)
        elseif op == OP_FUNCTION_CALL && wordcount(inst) >= 4 && current_func_id != 0
            callee_id = inst.words[4]
            push!(call_graph[current_func_id], callee_id)
        end
    end

    # 3. BFS from entry point to find all reachable functions
    reachable = Set{UInt32}()
    queue = UInt32[entry_func_id]
    while !isempty(queue)
        fid = popfirst!(queue)
        fid in reachable && continue
        push!(reachable, fid)
        for callee in get(call_graph, fid, Set{UInt32}())
            callee in reachable || push!(queue, callee)
        end
    end

    # 4. Remove unreachable functions
    dead_funcs = setdiff(keys(func_ranges), reachable)
    if !isempty(dead_funcs)
        dead_indices = Set{Int}()
        for fid in dead_funcs
            start_idx, end_idx = func_ranges[fid]
            for i in start_idx:end_idx
                push!(dead_indices, i)
            end
        end

        new_insts = SPVInst[]
        for (i, inst) in enumerate(mod.insts)
            i in dead_indices || push!(new_insts, inst)
        end
        empty!(mod.insts)
        append!(mod.insts, new_insts)
    end

    # 5. Fix invalid OpTypePointer whose pointee is an OpTypeFunction.
    func_type_ids = Set{UInt32}()
    uint8_type_id = UInt32(0)
    for inst in mod.insts
        op = opcode(inst)
        if op == OP_TYPE_FUNCTION
            push!(func_type_ids, inst.words[2])
        elseif op == OP_TYPE_INT && wordcount(inst) == 4
            if inst.words[3] == UInt32(8) && inst.words[4] == UInt32(0)
                uint8_type_id = inst.words[2]
            end
        end
    end
    if !isempty(func_type_ids) && uint8_type_id != 0
        for inst in mod.insts
            if opcode(inst) == OP_TYPE_POINTER && wordcount(inst) == 4
                if inst.words[4] in func_type_ids
                    inst.words[4] = uint8_type_id
                end
            end
        end
    end

    # 6. Stub out functions that contain invalid Vulkan instructions.
    # Error-handling paths (InexactError, BoundsError) from Julia may contain
    # OpConvertUToPtr/OpConvertPtrToU on non-PhysicalStorageBuffer pointers,
    # OpBitcast on function-pointer types, etc. Replace their bodies with
    # just OpLabel + OpReturn/OpReturnValue + OpFunctionEnd.
    _stub_invalid_functions!(mod, func_ranges, entry_func_id)

    @debug "fix_dead_functions!: removed $(length(dead_funcs)) dead functions"
end

const OP_CONVERT_U_TO_PTR = UInt16(120)
const OP_CONVERT_PTR_TO_U = UInt16(117)

function _stub_invalid_functions!(mod::SPVModule, func_ranges, entry_func_id)
    # Build pointer storage class map
    ptr_sc = Dict{UInt32, UInt32}()
    for inst in mod.insts
        if opcode(inst) == OP_TYPE_POINTER && wordcount(inst) == 4
            ptr_sc[inst.words[2]] = inst.words[3]
        end
    end

    # Find functions with invalid instructions (skip entry point)
    invalid_funcs = Set{UInt32}()
    current_fid = UInt32(0)
    for inst in mod.insts
        op = opcode(inst)
        if op == OP_FUNCTION && wordcount(inst) >= 3
            current_fid = inst.words[3]
        elseif op == OP_FUNCTION_END
            current_fid = UInt32(0)
        elseif current_fid != 0 && current_fid != entry_func_id
            # Check for ConvertUToPtr/ConvertPtrToU on non-PSB pointers
            if op == OP_CONVERT_U_TO_PTR && wordcount(inst) >= 3
                result_type = inst.words[2]
                sc = get(ptr_sc, result_type, SC_PHYSICAL_STORAGE_BUFFER)
                if sc != SC_PHYSICAL_STORAGE_BUFFER
                    push!(invalid_funcs, current_fid)
                end
            end
        end
    end
    isempty(invalid_funcs) && return

    # Find void type for OpReturn
    void_type_id = UInt32(0)
    for inst in mod.insts
        if opcode(inst) == OP_TYPE_VOID
            void_type_id = inst.words[2]
            break
        end
    end

    # Stub out invalid functions: keep OpFunction + OpFunctionParameter, then
    # replace body with OpLabel + OpReturn + OpFunctionEnd
    n_stubbed = 0
    new_insts = SPVInst[]
    current_fid = UInt32(0)
    in_invalid_func = false
    emitted_label = false

    for inst in mod.insts
        op = opcode(inst)

        if op == OP_FUNCTION && wordcount(inst) >= 3
            current_fid = inst.words[3]
            in_invalid_func = current_fid in invalid_funcs
            emitted_label = false
            push!(new_insts, inst)
        elseif op == OP_FUNCTION_END
            if in_invalid_func && !emitted_label
                # Emit a label + return
                label_id = alloc_id!(mod)
                push!(new_insts, SPVInst(UInt32[(UInt32(2) << 16) | UInt32(OP_LABEL), label_id]))
                push!(new_insts, SPVInst(UInt32[(UInt32(1) << 16) | UInt32(OP_RETURN)]))
                n_stubbed += 1
            end
            push!(new_insts, inst)
            in_invalid_func = false
            current_fid = UInt32(0)
        elseif in_invalid_func
            # Keep OpFunctionParameter instructions only, skip everything else
            if wordcount(inst) >= 3 && op == UInt16(55)  # OpFunctionParameter
                push!(new_insts, inst)
            elseif op == OP_LABEL && !emitted_label
                push!(new_insts, inst)
                emitted_label = true
                # Emit return right after label
                push!(new_insts, SPVInst(UInt32[(UInt32(1) << 16) | UInt32(OP_RETURN)]))
                n_stubbed += 1
            end
            # Skip all other instructions in the invalid function
        else
            push!(new_insts, inst)
        end
    end

    empty!(mod.insts)
    append!(mod.insts, new_insts)

    @debug "_stub_invalid_functions!: stubbed $n_stubbed functions"
end


# ── 12. Replace OpCopyMemorySized with OpCopyMemory ──────────────────────
#
# OpCopyMemorySized requires the Addresses capability (Kernel-only).
# RADV's ACO cannot handle any deref-cast patterns (store_deref unimplemented),
# so we must avoid ALL pointer-level OpBitcasts in the copy chain.
#
# Strategy: when source comes from OpBitcast of a Private variable, retype the
# variable itself to match the target's pointee type, eliminate the OpBitcast,
# and use the variable directly as OpCopyMemory source. This avoids any
# nir_deref_cast in RADV's spirv-to-nir path.
#
# Type mismatch example (from detect_camera_medium kernel):
#   %249 = OpVariable %3201 Private %183   ; Private [128 x i8]*, null-initialized
#   %648 = OpBitcast  %3202 %249           ; Private i8* (element pointer)
#   OpCopyMemorySized %526 %648 128        ; target is Function [16 x i64]*
# Fix: retype %249 to Private [16 x i64]*, remove the OpBitcast, use %249 directly.

const OP_LOAD  = UInt16(61)
const OP_STORE = UInt16(62)

function fix_copy_memory_sized!(mod::SPVModule)
    # Build maps
    ptr_pointee = Dict{UInt32, UInt32}()   # ptr_type_id → pointee_type_id
    ptr_sc = Dict{UInt32, UInt32}()        # ptr_type_id → storage_class
    result_type_of = Dict{UInt32, UInt32}() # result_id → result_type_id
    bitcast_operand = Dict{UInt32, UInt32}() # bitcast_result_id → operand_id
    var_type = Dict{UInt32, UInt32}()       # var_result_id → ptr_type_id
    var_sc = Dict{UInt32, UInt32}()         # var_result_id → storage_class

    for inst in mod.insts
        op = opcode(inst)
        if op == OP_TYPE_POINTER && wordcount(inst) == 4
            ptr_pointee[inst.words[2]] = inst.words[4]
            ptr_sc[inst.words[2]] = inst.words[3]
        elseif op == OP_VARIABLE && wordcount(inst) >= 4
            result_type_of[inst.words[3]] = inst.words[2]
            var_type[inst.words[3]] = inst.words[2]
            var_sc[inst.words[3]] = inst.words[4]
        elseif op == OP_BITCAST && wordcount(inst) >= 4
            result_type_of[inst.words[3]] = inst.words[2]
            bitcast_operand[inst.words[3]] = inst.words[4]
        elseif op == OP_ACCESS_CHAIN && wordcount(inst) >= 4
            result_type_of[inst.words[3]] = inst.words[2]
        end
    end

    # Find OpCopyMemorySized instructions
    cms_indices = Int[]
    for (i, inst) in enumerate(mod.insts)
        opcode(inst) == OP_COPY_MEMORY_SIZED && push!(cms_indices, i)
    end
    isempty(cms_indices) && return

    new_type_insts = SPVInst[]    # new pointer types + null constants
    bitcasts_to_remove = Set{UInt32}()  # bitcast result IDs to delete

    for idx in cms_indices
        inst = mod.insts[idx]
        target_id = inst.words[2]
        source_id = inst.words[3]

        target_type = get(result_type_of, target_id, UInt32(0))
        source_type = get(result_type_of, source_id, UInt32(0))
        target_pointee = get(ptr_pointee, target_type, UInt32(0))
        source_pointee = get(ptr_pointee, source_type, UInt32(0))

        # Check if source comes from an OpBitcast of a variable
        orig_var_id = get(bitcast_operand, source_id, UInt32(0))
        orig_var_type = get(var_type, orig_var_id, UInt32(0))
        orig_var_sc_val = get(var_sc, orig_var_id, UInt32(0))

        if target_pointee != 0 && orig_var_id != 0 && orig_var_type != 0
            # Source is OpBitcast of a variable. Retype the variable to match target.
            target_sc = orig_var_sc_val  # keep the variable's storage class

            # Find or create pointer type: <var_sc> <target_pointee>*
            new_ptr_type_id = UInt32(0)
            for (pid, ptee) in ptr_pointee
                if ptee == target_pointee && get(ptr_sc, pid, UInt32(0)) == target_sc
                    new_ptr_type_id = pid
                    break
                end
            end
            if new_ptr_type_id == 0
                new_ptr_type_id = alloc_id!(mod)
                push!(new_type_insts, SPVInst(UInt32[
                    (UInt32(4) << 16) | UInt32(OP_TYPE_POINTER),
                    new_ptr_type_id, target_sc, target_pointee
                ]))
                ptr_pointee[new_ptr_type_id] = target_pointee
                ptr_sc[new_ptr_type_id] = target_sc
            end

            # Create OpConstantNull of target_pointee type (for variable initializer)
            null_const_id = UInt32(0)
            for jinst in mod.insts
                if opcode(jinst) == OP_CONSTANT_NULL && wordcount(jinst) == 3 && jinst.words[2] == target_pointee
                    null_const_id = jinst.words[3]
                    break
                end
            end
            if null_const_id == 0
                null_const_id = alloc_id!(mod)
                push!(new_type_insts, SPVInst(UInt32[
                    (UInt32(3) << 16) | UInt32(OP_CONSTANT_NULL),
                    target_pointee, null_const_id
                ]))
            end

            # Retype the variable: change its pointer type and initializer
            for jinst in mod.insts
                if opcode(jinst) == OP_VARIABLE && wordcount(jinst) >= 4 && jinst.words[3] == orig_var_id
                    jinst.words[2] = new_ptr_type_id
                    # Replace or add initializer
                    if wordcount(jinst) >= 5
                        jinst.words[5] = null_const_id
                    end
                    break
                end
            end

            # Mark the OpBitcast for removal
            push!(bitcasts_to_remove, source_id)

            # Rewrite OpCopyMemorySized: source becomes the variable directly
            inst.words[3] = orig_var_id
        end

        # Convert OpCopyMemorySized → OpCopyMemory (drop size operand)
        wc = wordcount(inst)
        new_words = UInt32[UInt32(0), inst.words[2], inst.words[3]]
        for j in 5:wc
            push!(new_words, inst.words[j])
        end
        new_words[1] = (UInt32(length(new_words)) << 16) | UInt32(OP_COPY_MEMORY)
        mod.insts[idx] = SPVInst(new_words)
    end

    # Remove dead OpBitcast instructions
    if !isempty(bitcasts_to_remove)
        filter!(mod.insts) do inst
            !(opcode(inst) == OP_BITCAST && wordcount(inst) >= 4 && inst.words[3] in bitcasts_to_remove)
        end
    end

    # Insert new types/constants before first OpVariable or OpFunction
    if !isempty(new_type_insts)
        new_insts = SPVInst[]
        inserted = false
        for inst in mod.insts
            op = opcode(inst)
            if !inserted && (op == OP_VARIABLE || op == OP_FUNCTION)
                append!(new_insts, new_type_insts)
                inserted = true
            end
            push!(new_insts, inst)
        end
        empty!(mod.insts)
        append!(mod.insts, new_insts)
    end

    # Strip Addresses capability (no longer needed after conversion)
    filter!(mod.insts) do inst
        !(opcode(inst) == OP_CAPABILITY && inst.words[2] == CAP_ADDRESSES)
    end
end

# ── Fix Function/Private pointer bitcasts via UntypedAccessChainKHR ──────
# RADV ACO cannot handle nir_deref_cast from SPIR-V OpBitcast on
# Function/Private pointers (crashes in store_deref instruction selection).
# Replace pointer-level OpBitcast with OpUntypedAccessChainKHR using
# OpTypeUntypedPointerKHR result types, and cascade to any OpAccessChain
# that consumes the untyped results.
# Requires VK_KHR_shader_untyped_pointers (Mesa 25.3+).

function fix_function_pointer_bitcasts!(mod::SPVModule)
    # 1. Build maps
    ptr_info = Dict{UInt32, Tuple{UInt32, UInt32}}()  # ptr_type_id → (sc, pointee_type_id)
    result_type_of = Dict{UInt32, UInt32}()            # result_id → type_id

    for inst in mod.insts
        op = opcode(inst)
        if op == OP_TYPE_POINTER && wordcount(inst) == 4
            ptr_info[inst.words[2]] = (inst.words[3], inst.words[4])
        end
        if wordcount(inst) >= 3
            if op == OP_VARIABLE
                result_type_of[inst.words[3]] = inst.words[2]
            elseif op in (OP_ACCESS_CHAIN, OP_IN_BOUNDS_ACCESS_CHAIN, OP_BITCAST) && wordcount(inst) >= 4
                result_type_of[inst.words[3]] = inst.words[2]
            end
        end
    end

    # 2. Create OpTypeUntypedPointerKHR for Function and Private storage classes
    untyped_ptr_types = Dict{UInt32, UInt32}()  # storage_class → untyped_ptr_type_id
    new_type_insts = SPVInst[]
    for sc in (SC_FUNCTION, SC_PRIVATE)
        id = alloc_id!(mod)
        untyped_ptr_types[sc] = id
        push!(new_type_insts, SPVInst(UInt32[
            (UInt32(3) << 16) | UInt32(OP_TYPE_UNTYPED_POINTER_KHR),
            id,
            sc,
        ]))
    end

    # 3. Replace OpBitcast on Function/Private pointers → OpUntypedAccessChainKHR
    untyped_ids = Set{UInt32}()  # IDs that now have untyped pointer type
    n_replaced = 0
    for (i, inst) in enumerate(mod.insts)
        op = opcode(inst)

        # Handle OpBitcast on Function/Private pointers
        if op == OP_BITCAST && wordcount(inst) >= 4
            result_type_id = inst.words[2]
            result_id = inst.words[3]
            source_id = inst.words[4]

            haskey(ptr_info, result_type_id) || continue
            result_sc, _ = ptr_info[result_type_id]
            (result_sc == SC_FUNCTION || result_sc == SC_PRIVATE) || continue

            haskey(result_type_of, source_id) || continue
            source_type_id = result_type_of[source_id]
            haskey(ptr_info, source_type_id) || continue
            source_sc, source_pointee = ptr_info[source_type_id]
            (source_sc == SC_FUNCTION || source_sc == SC_PRIVATE) || continue

            # If source is itself untyped, look up the pointee from the original typed pointer
            # that was tracked before it became untyped
            if source_id in untyped_ids
                # source_pointee came from the typed version — use original pointee
            end

            untyped_rt = untyped_ptr_types[result_sc]
            mod.insts[i] = SPVInst(UInt32[
                (UInt32(5) << 16) | UInt32(OP_UNTYPED_ACCESS_CHAIN_KHR),
                untyped_rt,
                result_id,
                source_id,
                source_pointee,
            ])
            push!(untyped_ids, result_id)
            result_type_of[result_id] = untyped_rt
            n_replaced += 1
        end
    end

    # 4. Convert OpAccessChain/OpInBoundsAccessChain consuming untyped pointers
    #    to OpUntypedAccessChainKHR.
    n_cascaded = 0
    for (i, inst) in enumerate(mod.insts)
        op = opcode(inst)
        (op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) || continue
        wordcount(inst) < 4 && continue

        result_type_id = inst.words[2]
        result_id = inst.words[3]
        base_id = inst.words[4]

        # Only cascade if base is an untyped pointer
        base_id in untyped_ids || continue

        # The result type is a typed pointer — look up its pointee as the base type
        # (the base type tells the instruction how to interpret the memory for indexing)
        haskey(ptr_info, result_type_id) || continue
        result_sc, result_pointee = ptr_info[result_type_id]

        # Use untyped pointer as result type
        untyped_rt = untyped_ptr_types[result_sc]

        # Build OpUntypedAccessChainKHR: result_type result base base_type [indices...]
        new_words = UInt32[UInt32(0), untyped_rt, result_id, base_id, result_pointee]
        for j in 5:wordcount(inst)
            push!(new_words, inst.words[j])
        end
        new_words[1] = (UInt32(length(new_words)) << 16) | UInt32(OP_UNTYPED_ACCESS_CHAIN_KHR)
        mod.insts[i] = SPVInst(new_words)
        push!(untyped_ids, result_id)
        result_type_of[result_id] = untyped_rt
        n_cascaded += 1
    end

    # 5. Insert new untyped pointer types before first OpVariable or OpFunction
    if !isempty(new_type_insts) && n_replaced > 0
        new_insts = SPVInst[]
        inserted = false
        for inst in mod.insts
            op = opcode(inst)
            if !inserted && (op == OP_VARIABLE || op == OP_FUNCTION)
                append!(new_insts, new_type_insts)
                inserted = true
            end
            push!(new_insts, inst)
        end
        empty!(mod.insts)
        append!(mod.insts, new_insts)
    end

    (n_replaced + n_cascaded) > 0 && @debug "fix_function_pointer_bitcasts!: replaced $n_replaced bitcasts, cascaded $n_cascaded access chains"
end


# ══════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════

"""
    spirv_fixup(spv_bytes::Vector{UInt8}, push_struct_info::Vector{Pair{Int,Int}};
                entry_name::String="", workgroup_size::NTuple{3,Int}=(64,1,1)) -> Vector{UInt8}

Apply Vulkan-compliance fixups to a SPIR-V binary.

`push_struct_info` is a vector of `(offset, size)` pairs for each push constant member.
`entry_name` is the actual entry point name; spurious entry points are removed.
`workgroup_size` is used to create the LocalSize execution mode if missing.
"""
function spirv_fixup(spv_bytes::Vector{UInt8}, push_struct_info::Vector{Pair{Int,Int}};
                     entry_name::String="", workgroup_size::NTuple{3,Int}=(64,1,1))
    mod = parse_spirv(spv_bytes)

    # DEBUG: save pre-fixup SPIR-V
    open("/tmp/debug_pre_fixup.spv", "w") do io
        write(io, spv_bytes)
    end

    # Ensure SPIR-V version is at least 1.4 — we need it for:
    #  - All module-scope variables in OpEntryPoint interface (not just Input/Output)
    #  - PhysicalStorageBufferAddresses capability
    # llvm-spirv may output 1.0–1.3 for simple kernels.
    SPIRV_1_4 = UInt32(0x00010400)
    if mod.header[2] < SPIRV_1_4
        mod.header[2] = SPIRV_1_4
    end

    # Analysis: collect IDs needed by fixup passes
    push_struct_type_id = _find_push_struct_type_id(mod)

    # Pre-opt fixup passes (order matters: capabilities → layout → types → code)
    fix_ext_inst_import!(mod)
    fix_capabilities!(mod)
    fix_memory_model!(mod)
    fix_entry_points!(mod, entry_name)
    # Find entry ID after fix_entry_points! (which may create the OpEntryPoint)
    entry_id = _find_entry_id(mod, entry_name)
    fix_execution_modes!(mod, entry_id; workgroup_size)
    fix_decorations!(mod, push_struct_type_id, push_struct_info)
    fix_storage_classes!(mod)
    fix_variable_initializers!(mod)
    fix_structured_control_flow!(mod)
    fix_merge_placement!(mod)
    fix_dead_blocks!(mod)
    fix_barrier_semantics!(mod)
    fix_ordered_unordered!(mod)
    fix_duplicate_merge_targets!(mod)
    fix_merge_single_pred_blocks!(mod)
    fix_duplicate_merge_targets!(mod)  # re-run: merge_single_pred can reintroduce duplicates
    # Remove dead functions (Julia runtime stubs with invalid SPIR-V) before spirv-opt
    fix_dead_functions!(mod)
    # Convert OpCopyMemorySized → OpCopyMemory PRE-opt.
    # This retypes source variables to match target types, producing type-safe OpCopyMemory.
    # Must run before spirv-opt, which would otherwise convert OpCopyMemorySized to
    # OpCopyMemory with pointer bitcasts that RADV ACO cannot handle (store_deref).
    fix_copy_memory_sized!(mod)

    # spirv-opt skipped — it only ran --eliminate-dead-functions which our
    # fix_dead_functions! already handles. Skipping avoids spirv-opt's strict
    # validation which rejects intermediate SPIR-V from llvm-spirv (e.g.
    # Addresses capability before our fixup removes it).
    fix_pushconstant_bitcast!(mod)
    # NOTE: fix_function_pointer_bitcasts! is disabled — SPV_KHR_untyped_pointers
    # cannot be used with Function/Private storage classes in Vulkan (no explicit layout).
    # The MArray quirks.jl override handles the main source of these bitcasts instead.
    # Keep the pass for future use on StorageBuffer/PhysicalStorageBuffer if needed.
    # fix_function_pointer_bitcasts!(mod)

    # DEBUG: save intermediate SPIR-V for analysis
    let bytes = to_bytes(mod)
        open("/tmp/debug_fixup_output.spv", "w") do io
            write(io, bytes)
        end
    end

    return to_bytes(mod)
end

# ── Analysis helpers ─────────────────────────────────────────────────────

function _find_entry_id(mod::SPVModule, entry_name::String)::UInt32
    isempty(entry_name) && return UInt32(0)
    for inst in mod.insts
        if opcode(inst) == OP_ENTRY_POINT && wordcount(inst) >= 4
            if inst.words[2] == EXEC_GL_COMPUTE
                name, _ = spv_read_string(inst.words, 4)
                name == entry_name && return inst.words[3]
            end
        end
    end
    return UInt32(0)
end

function _find_push_struct_type_id(mod::SPVModule)::UInt32
    # Collect all struct type IDs so we can verify the push constant type
    struct_ids = Set{UInt32}()
    for inst in mod.insts
        opcode(inst) == OP_TYPE_STRUCT && push!(struct_ids, inst.words[2])
    end
    # Find the UniformConstant pointer whose pointee is a struct
    for inst in mod.insts
        if opcode(inst) == OP_TYPE_POINTER && wordcount(inst) == 4
            if inst.words[3] == SC_UNIFORM_CONSTANT && inst.words[4] in struct_ids
                return inst.words[4]
            end
        end
    end
    return UInt32(0)
end
