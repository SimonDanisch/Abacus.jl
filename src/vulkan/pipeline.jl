# Vulkan compute pipeline creation and caching

struct VkComputePipeline
    shader_module::Vulkan.ShaderModule
    pipeline_layout::Vulkan.PipelineLayout
    pipeline::Vulkan.Pipeline
    push_constant_size::UInt32
end

# Cache: (spirv_bytes_hash, push_constant_size) → VkComputePipeline
const _pipeline_cache = Dict{UInt, VkComputePipeline}()

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
