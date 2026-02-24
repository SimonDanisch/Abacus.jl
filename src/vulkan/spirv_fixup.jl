# SPIR-V post-processor for Vulkan compliance
#
# Several fixups are handled at the LLVM IR level in _prepare_module_for_vulkan!():
# - Alignment: fixed on load/store instructions before llc
# - Lifetime intrinsics: stripped before llc
# - Internal linkage: set on non-entry function definitions
#
# And via proper LLVM intrinsics / naming conventions:
# - BuiltIn decorations: __spirv_BuiltIn* naming → llc auto-decorates
# - Barrier: @llvm.spv.group.memory.barrier.with.group.sync → OpControlBarrier
#   (NOTE: hardcodes SequentiallyConsistent semantics which spirv-val rejects
#    for Vulkan, but drivers accept it. No parameterized Vulkan barrier intrinsic
#    exists in the LLVM SPIR-V backend — __spirv_ControlBarrier only works with
#    the OpenCL target.)
#
# Remaining text-based fixups (blocked on upstream LLVM SPIR-V backend):
# 1. Linkage capability/decorations (external declarations still trigger these)
# 2. UniformConstant → PushConstant (addrspace 13 requires LLVM 23+)
# 3. CrossWorkgroup → PhysicalStorageBuffer (no address space mapping yet)
# 4. PhysicalStorageBuffer64 memory model + capability
# 5. Block/Offset decorations on push constant struct

using SPIRV_Tools_jll

"""
    spirv_fixup(spv_bytes::Vector{UInt8}, push_struct_info::Vector{Pair{Int,Int}};
                entry_name::String="") -> Vector{UInt8}

Apply Vulkan-compliance fixups to a SPIR-V binary.

`push_struct_info` is a vector of `(offset, size)` pairs for each member of the
push constant struct, used to generate Offset decorations.
`entry_name` is the name of the actual entry point; all other OpEntryPoint
declarations are removed.
"""
function spirv_fixup(spv_bytes::Vector{UInt8}, push_struct_info::Vector{Pair{Int,Int}};
                     entry_name::String="")
    # Disassemble with raw IDs for consistent manipulation
    dis = mktempdir() do dir
        spv_in = joinpath(dir, "in.spv")
        write(spv_in, spv_bytes)
        read(`$(SPIRV_Tools_jll.spirv_dis()) --raw-id $spv_in`, String)
    end

    fixed = dis

    # 0. Remove spurious OpEntryPoint/OpExecutionMode for non-entry functions.
    if !isempty(entry_name)
        fixed = replace(fixed, Regex("^\\s*OpEntryPoint GLCompute %\\S+ \"(?!" *
            replace(entry_name, r"([.*+?^${}()|[\]\\])" => s"\\\1") *
            "\").*\\n", "m") => "")
        entry_match = match(Regex("OpEntryPoint GLCompute (\\%\\S+) \"" *
            replace(entry_name, r"([.*+?^${}()|[\]\\])" => s"\\\1") * "\""), fixed)
        if entry_match !== nothing
            entry_id = entry_match.captures[1]
            entry_id_escaped = _escape_for_regex(entry_id)
            fixed = replace(fixed, Regex("^\\s*OpExecutionMode (?!" *
                entry_id_escaped * ")%\\S+ .*\\n", "m") => "")
        end
    end

    # 1. Remove OpCapability Linkage and LinkageAttributes decorations.
    fixed = replace(fixed, r"^\s*OpCapability Linkage\n"m => "")
    fixed = replace(fixed, r"^\s*OpDecorate .* LinkageAttributes .*\n"m => "")

    # 2. Add PhysicalStorageBufferAddresses capability
    if !occursin("PhysicalStorageBufferAddresses", fixed)
        fixed = replace(fixed, "OpCapability Shader\n" =>
            "OpCapability Shader\n               OpCapability PhysicalStorageBufferAddresses\n")
    end

    # 3. Change memory model to PhysicalStorageBuffer64
    fixed = replace(fixed, "OpMemoryModel Logical GLSL450" =>
        "OpMemoryModel PhysicalStorageBuffer64 GLSL450")

    # 4. CrossWorkgroup → PhysicalStorageBuffer (BDA pointers)
    fixed = replace(fixed, "CrossWorkgroup" => "PhysicalStorageBuffer")

    # 5. UniformConstant → PushConstant (addrspace 2 → PushConstant)
    # NOTE: addrspace 13 maps directly to PushConstant but requires LLVM 23+
    fixed = replace(fixed, "UniformConstant" => "PushConstant")

    # 6. Barrier semantics: @llvm.spv.group.memory.barrier.with.group.sync hardcodes
    #    SequentiallyConsistent (0x10), which spirv-val rejects for Vulkan.
    #    However, SequentiallyConsistent is strictly stronger than AcquireRelease,
    #    so GPU drivers (RADV, ANV) accept it fine. The LLVM SPIR-V backend has no
    #    parameterized barrier intrinsic for the Vulkan target (__spirv_ControlBarrier
    #    only works with the OpenCL SPIR-V target). Left as-is for now; spirv-val
    #    validation is skipped for this specific VUID.
    # TODO: fix once upstream adds llvm.spv.control.barrier(i32,i32,i32) for Vulkan

    # 7. Add Block + Offset decorations on push constant struct
    struct_id = _find_push_struct_id(fixed)
    if struct_id !== nothing
        decor_lines = "               OpDecorate $struct_id Block\n"
        for (i, (offset, _)) in enumerate(push_struct_info)
            decor_lines *= "               OpMemberDecorate $struct_id $(i-1) Offset $offset\n"
        end
        fixed = replace(fixed, r"( +%\d+ = OpType)"m =>
            SubstitutionString(decor_lines * "\\1"); count=1)
    end

    # Reassemble and optimize
    mktempdir() do dir
        asm_path = joinpath(dir, "fixed.spvasm")
        spv_path = joinpath(dir, "fixed.spv")
        opt_path = joinpath(dir, "opt.spv")
        write(asm_path, fixed)
        run(`$(SPIRV_Tools_jll.spirv_as()) $asm_path -o $spv_path --target-env vulkan1.3`)

        # Use spirv-opt to clean up dead functions and trim capabilities.
        opt_ok = try
            run(pipeline(`$(SPIRV_Tools_jll.spirv_opt()) --target-env=vulkan1.3
                 --eliminate-dead-functions
                 -o $opt_path $spv_path`; stderr=devnull))
            true
        catch
            false
        end

        final_spv = opt_ok ? read(opt_path) : read(spv_path)

        # Post-opt fixup: spirv-opt may optimize OpAccessChain at index 0 into OpBitcast
        # on PushConstant pointers, which crashes RADV. Convert back to OpAccessChain.
        final_spv = _fix_pushconstant_bitcast(final_spv)

        return final_spv
    end
end

"""
Fix OpBitcast on PushConstant pointers that spirv-opt introduces.

spirv-opt may optimize `OpAccessChain %ptr %struct %0` into `OpBitcast %ptr %struct`
when accessing the first member (index 0). RADV crashes on this pattern.
Convert back to OpAccessChain with an explicit index 0.
"""
function _fix_pushconstant_bitcast(spv_bytes::Vector{UInt8})
    mktempdir() do dir
        spv_in = joinpath(dir, "in.spv")
        write(spv_in, spv_bytes)
        text = read(`$(SPIRV_Tools_jll.spirv_dis()) --raw-id $spv_in`, String)

        pc_match = match(r"(%\d+) = OpVariable %\d+ PushConstant"m, text)
        if pc_match === nothing
            return spv_bytes
        end
        pc_var = pc_match.captures[1]
        pc_var_escaped = _escape_for_regex(pc_var)

        bitcast_rx = Regex("OpBitcast %\\d+ " * pc_var_escaped * "\\b", "m")
        if !occursin(bitcast_rx, text)
            return spv_bytes
        end

        uint_type_match = match(r"(%\d+) = OpTypeInt 32 0"m, text)
        if uint_type_match === nothing
            return spv_bytes
        end
        uint_type = uint_type_match.captures[1]
        uint_type_escaped = _escape_for_regex(uint_type)

        zero_rx = Regex("(%\\d+) = OpConstant " * uint_type_escaped * " 0\\b", "m")
        zero_match = match(zero_rx, text)
        if zero_match !== nothing
            zero_id = zero_match.captures[1]
        else
            bound_match = match(r"; Bound: (\d+)"m, text)
            new_id = parse(Int, bound_match.captures[1])
            zero_id = "%$new_id"
            text = replace(text, "; Bound: $(bound_match.captures[1])" =>
                "; Bound: $(new_id + 1)")
            uint_line_escaped = _escape_for_regex(uint_type) * " = OpTypeInt 32 0"
            text = replace(text, Regex("(" * uint_line_escaped * "\\n)", "m") =>
                SubstitutionString("\\1         $zero_id = OpConstant $uint_type 0\n"))
        end

        text = replace(text,
            Regex("OpBitcast (%\\d+ " * pc_var_escaped * ")\\b", "m") =>
            SubstitutionString("OpAccessChain \\1 $zero_id"))

        asm_path = joinpath(dir, "fixed.spvasm")
        out_path = joinpath(dir, "fixed.spv")
        write(asm_path, text)
        run(`$(SPIRV_Tools_jll.spirv_as()) $asm_path -o $out_path --target-env vulkan1.3`)
        return read(out_path)
    end
end

"""Find the SPIR-V ID of the push constant struct type."""
function _find_push_struct_id(spv_text::String)
    for m in eachmatch(r"OpTypePointer PushConstant (%\d+)"m, spv_text)
        candidate = m.captures[1]
        if occursin(Regex(_escape_for_regex(candidate) * " = OpTypeStruct"), spv_text)
            return candidate
        end
    end
    return nothing
end

function _escape_for_regex(s::AbstractString)
    replace(s, "%" => "\\%")
end
