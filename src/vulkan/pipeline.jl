# Vulkan compute pipeline creation and caching

using SPIRV_Tools_jll

struct VkComputePipeline
    shader_module::Vulkan.ShaderModule
    pipeline_layout::Vulkan.PipelineLayout
    pipeline::Vulkan.Pipeline
    push_constant_size::UInt32
end

# Cache: (spirv_bytes_hash, push_constant_size) → VkComputePipeline
const _pipeline_cache = Dict{UInt, VkComputePipeline}()

"""
    _validate_spirv(spirv_bytes) -> nothing

Validate SPIR-V binary with spirv-val for Vulkan 1.3.
Throws an error if validation fails, preventing driver segfaults from invalid SPIR-V.
"""
# Known spirv-val errors that are acceptable (drivers handle them correctly).
# See VULKAN_FINDINGS.md for rationale on each.
const _SPIRV_VAL_IGNORED_VUIDS = [
    # Barrier SequentiallyConsistent semantics: strictly stronger than required
    # AcquireRelease. RADV/ANV accept it. No LLVM-level fix exists (verified via MWE).
    "VUID-StandaloneSpirv-MemorySemantics-10866",
]

# Error substrings to ignore in spirv-val output (for patterns without VUIDs)
const _SPIRV_VAL_IGNORED_MESSAGES = String[]

function _validate_spirv(spirv_bytes::Vector{UInt8})
    mktempdir() do dir
        spv_path = joinpath(dir, "kernel.spv")
        write(spv_path, spirv_bytes)

        # Run spirv-val via JLL do-block, capturing stderr
        SPIRV_Tools_jll.spirv_val() do val_exe
            err_buf = IOBuffer()
            proc = run(pipeline(ignorestatus(`$val_exe --target-env vulkan1.3 $spv_path`);
                                stderr=err_buf))
            if proc.exitcode != 0
                err_text = String(take!(err_buf))
                # Filter out known-acceptable errors
                real_errors = filter(split(err_text, '\n')) do line
                    startswith(line, "error:") &&
                        !any(vuid -> contains(line, vuid), _SPIRV_VAL_IGNORED_VUIDS) &&
                        !any(msg -> contains(line, msg), _SPIRV_VAL_IGNORED_MESSAGES)
                end
                isempty(real_errors) && return  # all errors are known-acceptable
                # Get disassembly for debugging
                dis = SPIRV_Tools_jll.spirv_dis() do dis_exe
                    try read(`$dis_exe --raw-id $spv_path`, String) catch; "" end
                end
                dis_lines = split(dis, '\n')
                # Extract error line numbers and show context around them
                err_context = String[]
                for err_line in real_errors
                    m = match(r"line (\d+)", err_line)
                    if m !== nothing
                        ln = parse(Int, m.captures[1])
                        start_ln = max(1, ln - 10)
                        end_ln = min(length(dis_lines), ln + 5)
                        push!(err_context, "--- Around line $ln ---")
                        for k in start_ln:end_ln
                            prefix = k == ln ? ">>>" : "   "
                            push!(err_context, "$prefix $k: $(dis_lines[k])")
                        end
                    end
                end
                context_str = isempty(err_context) ? join(last(dis_lines, 40), '\n') : join(err_context, '\n')
                error("SPIR-V validation failed — refusing to create pipeline (would segfault driver).\n",
                      "spirv-val errors:\n", join(real_errors, '\n'), "\n\n",
                      "Disassembly context:\n$context_str")
            end
        end
    end
end

"""
    get_pipeline(spirv_bytes, entry_name, push_constant_size) -> VkComputePipeline

Create or retrieve a cached compute pipeline from SPIR-V bytecode.
"""
function get_pipeline(spirv_bytes::Vector{UInt8}, entry_name::String,
                      push_constant_size::Integer)
    key = hash((spirv_bytes, push_constant_size))
    cached = get(_pipeline_cache, key, nothing)
    if cached !== nothing
        return cached
    end

    # Save SPIR-V for debugging if requested
    if get(ENV, "ABACUS_SAVE_IR", "") == "1"
        write("/tmp/abacus_debug_fixed.spv", spirv_bytes)
    end

    # Validate SPIR-V before sending to driver to prevent segfaults
    _validate_spirv(spirv_bytes)

    dev = vk_device()

    # Create shader module
    # SPIR-V is UInt32-aligned; Vulkan.jl expects the code as Vector{UInt32}
    code_u32 = reinterpret(UInt32, spirv_bytes)
    shader_mod = unwrap(Vulkan.create_shader_module(dev, length(spirv_bytes), code_u32))

    # Push constant range (all data in a single range, visible to compute stage)
    push_ranges = if push_constant_size > 0
        [Vulkan.PushConstantRange(Vulkan.SHADER_STAGE_COMPUTE_BIT,
                                   UInt32(0), UInt32(push_constant_size))]
    else
        Vulkan.PushConstantRange[]
    end

    # Pipeline layout (no descriptor sets — we use push constants + BDA)
    layout = unwrap(Vulkan.create_pipeline_layout(dev, Vulkan.DescriptorSetLayout[],
        push_ranges))

    # Compute pipeline
    stage = Vulkan.PipelineShaderStageCreateInfo(
        Vulkan.SHADER_STAGE_COMPUTE_BIT,
        shader_mod,
        entry_name
    )
    ci = Vulkan.ComputePipelineCreateInfo(stage, layout, -1)
    pipelines_result = unwrap(Vulkan.create_compute_pipelines(dev, [ci]))
    # create_compute_pipelines returns (Vector{Pipeline}, Result)
    pipeline = pipelines_result[1][1]

    result = VkComputePipeline(shader_mod, layout, pipeline, UInt32(push_constant_size))
    _pipeline_cache[key] = result
    return result
end
