# Command buffer recording and submission for compute dispatch

# Pre-built SubmitInfo pointing to the context's reusable command buffer.
# Allocated once to avoid per-dispatch Vector/struct allocations.
const _submit_info = Ref{Any}(nothing)

function _get_submit_info(ctx::VkContext)
    si = _submit_info[]
    if si !== nothing
        return si::Vulkan.SubmitInfo
    end
    si = Vulkan.SubmitInfo(Vulkan.Semaphore[], UInt32[],
                           [ctx.cmd_buf], Vulkan.Semaphore[])
    _submit_info[] = si
    return si
end

"""
    vk_dispatch!(pipeline::VkComputePipeline, push_data::Vector{UInt8},
                 groups::NTuple{3,Integer})

Record and submit a compute dispatch command using pre-allocated resources:
1. Reset and record the reusable command buffer
2. Bind pipeline, push constants, dispatch
3. Submit with reusable fence and wait (blocking)
"""
function vk_dispatch!(pipeline::VkComputePipeline, push_data::Vector{UInt8},
                      groups::NTuple{3, <:Integer})
    ctx = vk_context()
    dev = ctx.device
    cmd = ctx.cmd_buf
    fence = ctx.fence

    # Record into the reusable command buffer (implicitly resets via ONE_TIME_SUBMIT)
    unwrap(Vulkan.begin_command_buffer(cmd,
        Vulkan.CommandBufferBeginInfo(;
            flags = Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)))

    Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)

    if !isempty(push_data)
        GC.@preserve push_data begin
            Vulkan.cmd_push_constants(cmd, pipeline.pipeline_layout,
                Vulkan.SHADER_STAGE_COMPUTE_BIT, 0, length(push_data),
                Ptr{Nothing}(pointer(push_data)))
        end
    end

    Vulkan.cmd_dispatch(cmd, UInt32(groups[1]), UInt32(groups[2]), UInt32(groups[3]))

    unwrap(Vulkan.end_command_buffer(cmd))

    # Submit with reusable fence
    submit_info = _get_submit_info(ctx)
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence))

    # Wait and reset fence for next use
    unwrap(Vulkan.wait_for_fences(dev, [fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(dev, [fence]))

    return nothing
end
