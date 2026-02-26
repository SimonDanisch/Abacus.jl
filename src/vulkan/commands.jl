# Command buffer recording and submission for compute dispatch
#
# Dispatches are recorded into a single command buffer without submitting.
# vk_flush!() ends recording, submits the batch, and waits for completion.
# Memory barriers between dispatches ensure correct read-after-write ordering.

# Pre-built SubmitInfo pointing to the context's reusable command buffer.
# Allocated once to avoid per-dispatch Vector/struct allocations.
const _submit_info = Ref{Any}(nothing)

# In-flight argument buffers that must stay alive until vk_flush!() completes.
# BDA argument buffers are referenced by the GPU via PhysicalStorageBuffer;
# if GC frees them before the GPU finishes, we get GPUVM faults.
const _inflight_arg_bufs = VkManagedBuffer[]

"""Keep an argument buffer alive until the next vk_flush!() completes."""
function _keep_alive!(buf::VkManagedBuffer)
    push!(_inflight_arg_bufs, buf)
    return
end

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

"""Begin recording into the reusable command buffer if not already recording."""
function _ensure_recording(ctx::VkContext)
    ctx.recording && return
    unwrap(Vulkan.begin_command_buffer(ctx.cmd_buf,
        Vulkan.CommandBufferBeginInfo(;
            flags = Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)))
    ctx.recording = true
    return
end

"""
    vk_dispatch!(pipeline::VkComputePipeline, push_data::Vector{UInt8},
                 groups::NTuple{3,Integer})

Record a compute dispatch into the current command buffer (non-blocking).
A memory barrier is inserted between consecutive dispatches to ensure
write-after-read correctness. Call `vk_flush!()` to submit and wait.
"""
function vk_dispatch!(pipeline::VkComputePipeline, push_data::Vector{UInt8},
                      groups::NTuple{3, <:Integer})
    ctx = vk_context()
    cmd = ctx.cmd_buf

    _ensure_recording(ctx)

    # Barrier between consecutive dispatches (not before the first one)
    if ctx.dispatch_count > 0
        barrier = Vulkan.MemoryBarrier(
            C_NULL,
            Vulkan.ACCESS_SHADER_WRITE_BIT,
            Vulkan.ACCESS_SHADER_READ_BIT | Vulkan.ACCESS_SHADER_WRITE_BIT)
        Vulkan.cmd_pipeline_barrier(cmd,
            [barrier],
            Vulkan.BufferMemoryBarrier[],
            Vulkan.ImageMemoryBarrier[];
            src_stage_mask = Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            dst_stage_mask = Vulkan.PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    end

    Vulkan.cmd_bind_pipeline(cmd, Vulkan.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)

    if !isempty(push_data)
        GC.@preserve push_data begin
            Vulkan.cmd_push_constants(cmd, pipeline.pipeline_layout,
                Vulkan.SHADER_STAGE_COMPUTE_BIT, 0, length(push_data),
                Ptr{Nothing}(pointer(push_data)))
        end
    end

    Vulkan.cmd_dispatch(cmd, UInt32(groups[1]), UInt32(groups[2]), UInt32(groups[3]))

    ctx.dispatch_count += 1

    return nothing
end

"""
    vk_flush!()

End command buffer recording, submit the batch, and wait for completion.
No-op if no dispatches have been recorded.
"""
function vk_flush!()
    ctx = vk_context()
    ctx.recording || return nothing

    dev = ctx.device
    fence = ctx.fence

    unwrap(Vulkan.end_command_buffer(ctx.cmd_buf))

    submit_info = _get_submit_info(ctx)
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence))

    unwrap(Vulkan.wait_for_fences(dev, [fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(dev, [fence]))

    ctx.recording = false
    ctx.dispatch_count = 0

    # Release in-flight argument buffers (GPU is done with them now)
    empty!(_inflight_arg_bufs)

    return nothing
end
