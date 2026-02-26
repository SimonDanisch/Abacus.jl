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

const CAP_SHADER                             = UInt32(1)
const CAP_LINKAGE                            = UInt32(5)
const CAP_VARIABLE_POINTERS_STORAGE_BUFFER   = UInt32(4441)
const CAP_VARIABLE_POINTERS                  = UInt32(4442)
const CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES  = UInt32(5347)

const AM_PHYSICAL_STORAGE_BUFFER_64 = UInt32(5348)

const DEC_BLOCK              = UInt32(2)
const DEC_OFFSET             = UInt32(35)
const DEC_LINKAGE_ATTRIBUTES = UInt32(41)
const DEC_ALIGNMENT          = UInt32(44)

const EXEC_GL_COMPUTE = UInt32(5)


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
    for (i, inst) in enumerate(mod.insts)
        if opcode(inst) == OP_EXT_INST_IMPORT
            name, _ = spv_read_string(inst.words, 3)
            if name == "OpenCL.std"
                result_id = inst.words[2]
                str_words = spv_encode_string("GLSL.std.450")
                wc = 2 + length(str_words)
                mod.insts[i] = SPVInst(UInt32[
                    (UInt32(wc) << 16) | UInt32(OP_EXT_INST_IMPORT),
                    result_id, str_words...
                ])
            end
        end
    end
end

# ── 2. Fix capabilities & extensions ─────────────────────────────────────
# Remove Linkage capability, add PhysicalStorageBufferAddresses + extension.

function fix_capabilities!(mod::SPVModule)
    has_psb = any(mod.insts) do inst
        opcode(inst) == OP_CAPABILITY && inst.words[2] == CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES
    end
    caps_added = false
    ext_added = false
    new_insts = SPVInst[]

    for inst in mod.insts
        op = opcode(inst)

        # Remove Linkage capability
        op == OP_CAPABILITY && inst.words[2] == CAP_LINKAGE && continue

        # After Shader capability, insert required capabilities
        if op == OP_CAPABILITY && inst.words[2] == CAP_SHADER && !caps_added
            push!(new_insts, inst)
            if !has_psb
                push!(new_insts, SPVInst(OP_CAPABILITY, CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES))
            end
            # VariablePointers enables OpSelect/OpPhi on pointer types (needed for
            # closures, conditional array accesses, loop-dependent pointers, etc.)
            push!(new_insts, SPVInst(OP_CAPABILITY, CAP_VARIABLE_POINTERS_STORAGE_BUFFER))
            push!(new_insts, SPVInst(OP_CAPABILITY, CAP_VARIABLE_POINTERS))
            caps_added = true
            continue
        end

        # Before first non-capability, insert required extensions
        if !ext_added && op != OP_CAPABILITY
            for ext_name in ["SPV_KHR_physical_storage_buffer",
                             "SPV_KHR_variable_pointers",
                             "SPV_KHR_storage_buffer_storage_class"]
                str = spv_encode_string(ext_name)
                wc = 1 + length(str)
                push!(new_insts, SPVInst(UInt32[(UInt32(wc) << 16) | UInt32(OP_EXTENSION), str...]))
            end
            ext_added = true
        end

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
        end
    end
end

# ── 4. Fix entry points — keep only the target entry ─────────────────────

function fix_entry_points!(mod::SPVModule, entry_name::String)
    isempty(entry_name) && return
    filter!(mod.insts) do inst
        if opcode(inst) == OP_ENTRY_POINT && wordcount(inst) >= 4
            name, _ = spv_read_string(inst.words, 4)
            return name == entry_name
        end
        true
    end
end

# ── 5. Fix execution modes — keep only for the target entry ──────────────

function fix_execution_modes!(mod::SPVModule, entry_id::UInt32)
    entry_id == 0 && return
    filter!(mod.insts) do inst
        opcode(inst) == OP_EXECUTION_MODE && wordcount(inst) >= 3 ?
            inst.words[2] == entry_id : true
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

        # Remove Linkage and Alignment decorations
        if op == OP_DECORATE && wordcount(inst) >= 3
            dec = inst.words[3]
            (dec == DEC_LINKAGE_ATTRIBUTES || dec == DEC_ALIGNMENT) && continue
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
# CrossWorkgroup → PhysicalStorageBuffer, UniformConstant → PushConstant.

function fix_storage_classes!(mod::SPVModule)
    # Simple remapping:
    # - CrossWorkgroup → PhysicalStorageBuffer (BDA array pointers)
    # - UniformConstant → PushConstant
    #
    # Constant globals (polynomial tables etc.) are moved from addrspace(1) to
    # addrspace(0) at the LLVM IR level by _move_constant_globals_to_private!,
    # so they never appear as CrossWorkgroup in the SPIR-V binary.
    for inst in mod.insts
        op = opcode(inst)
        if op == OP_TYPE_POINTER && wordcount(inst) == 4
            if inst.words[3] == SC_CROSS_WORKGROUP
                inst.words[3] = SC_PHYSICAL_STORAGE_BUFFER
            elseif inst.words[3] == SC_UNIFORM_CONSTANT
                inst.words[3] = SC_PUSH_CONSTANT
            end
        elseif op == OP_VARIABLE && wordcount(inst) >= 4
            if inst.words[4] == SC_CROSS_WORKGROUP
                inst.words[4] = SC_PHYSICAL_STORAGE_BUFFER
            elseif inst.words[4] == SC_UNIFORM_CONSTANT
                inst.words[4] = SC_PUSH_CONSTANT
            end
        end
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

    # Insert trampoline blocks before the last OpFunctionEnd
    last_func_end = findlast(inst -> opcode(inst) == OP_FUNCTION_END, mod.insts)
    if last_func_end !== nothing
        tramp_insts = SPVInst[]
        for (target, new_ids) in trampolines, new_id in new_ids
            push!(tramp_insts, SPVInst(OP_LABEL, new_id))
            push!(tramp_insts, SPVInst(OP_BRANCH, target))
        end
        splice!(mod.insts, last_func_end:last_func_end-1, tramp_insts)
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

    (pc_var_id == 0 || zero_const_id == 0) && return

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


# ══════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════

"""
    spirv_fixup(spv_bytes::Vector{UInt8}, push_struct_info::Vector{Pair{Int,Int}};
                entry_name::String="") -> Vector{UInt8}

Apply Vulkan-compliance fixups to a SPIR-V binary.

`push_struct_info` is a vector of `(offset, size)` pairs for each push constant member.
`entry_name` is the actual entry point name; spurious entry points are removed.
"""
function spirv_fixup(spv_bytes::Vector{UInt8}, push_struct_info::Vector{Pair{Int,Int}};
                     entry_name::String="")
    mod = parse_spirv(spv_bytes)

    # Analysis: collect IDs needed by fixup passes
    entry_id = _find_entry_id(mod, entry_name)
    push_struct_type_id = _find_push_struct_type_id(mod)

    # Pre-opt fixup passes (order matters: capabilities → layout → types → code)
    fix_ext_inst_import!(mod)
    fix_capabilities!(mod)
    fix_memory_model!(mod)
    fix_entry_points!(mod, entry_name)
    fix_execution_modes!(mod, entry_id)
    fix_decorations!(mod, push_struct_type_id, push_struct_info)
    fix_storage_classes!(mod)
    fix_variable_initializers!(mod)
    fix_merge_placement!(mod)
    fix_barrier_semantics!(mod)
    fix_ordered_unordered!(mod)
    fix_duplicate_merge_targets!(mod)

    # Dead function elimination via spirv-opt
    bytes = _spirv_opt(to_bytes(mod))

    # Post-opt fixups (spirv-opt can introduce new patterns)
    mod = parse_spirv(bytes)
    fix_pushconstant_bitcast!(mod)

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
