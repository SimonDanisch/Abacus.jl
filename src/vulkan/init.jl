# Vulkan initialization — singleton context for compute operations

using Vulkan

export VkContext, vk_context, vk_device, vk_queue, vk_cmd_pool

mutable struct VkContext
    instance::Vulkan.Instance
    physical_device::Vulkan.PhysicalDevice
    device::Vulkan.Device
    queue::Vulkan.Queue
    queue_family_index::UInt32
    cmd_pool::Vulkan.CommandPool
    device_name::String
    # Pre-allocated dispatch resources (reused every vk_dispatch!)
    cmd_buf::Vulkan.CommandBuffer
    fence::Vulkan.Fence
end

const _vk_context = Ref{Union{Nothing, VkContext}}(nothing)

"""
    vk_context() -> VkContext

Return the global Vulkan context, initializing it on first call.
Picks the first discrete GPU with buffer device address support,
falling back to any GPU with compute queues.
"""
function vk_context()
    ctx = _vk_context[]
    if ctx !== nothing
        return ctx
    end
    return _init_vulkan!()
end

vk_device() = vk_context().device
vk_queue() = vk_context().queue
vk_cmd_pool() = vk_context().cmd_pool

function _init_vulkan!()
    # Create instance
    layers = String[]
    extensions = String[]

    instance = unwrap(Vulkan.create_instance(
        layers, extensions;
        application_info = Vulkan.ApplicationInfo(
            v"0.1.0", v"0.1.0", v"1.3";
            application_name = "Abacus",
            engine_name = "Abacus"
        )
    ))

    # Pick physical device: prefer discrete GPU
    pdevs = unwrap(Vulkan.enumerate_physical_devices(instance))
    isempty(pdevs) && error("No Vulkan physical devices found")

    pd = nothing
    for candidate in pdevs
        props = Vulkan.get_physical_device_properties(candidate)
        if props.device_type == Vulkan.PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            pd = candidate
            break
        end
    end
    if pd === nothing
        # fallback: first device
        pd = pdevs[1]
    end

    props = Vulkan.get_physical_device_properties(pd)
    device_name = props.device_name

    # Find a compute queue family
    qf_props = Vulkan.get_physical_device_queue_family_properties(pd)
    compute_qf = nothing

    # Prefer a dedicated compute queue (no graphics bit)
    for (i, qf) in enumerate(qf_props)
        if (qf.queue_flags & Vulkan.QUEUE_COMPUTE_BIT) != 0 &&
           (qf.queue_flags & Vulkan.QUEUE_GRAPHICS_BIT) == 0
            compute_qf = UInt32(i - 1)  # 0-indexed
            break
        end
    end
    # Fallback: any queue with compute
    if compute_qf === nothing
        for (i, qf) in enumerate(qf_props)
            if (qf.queue_flags & Vulkan.QUEUE_COMPUTE_BIT) != 0
                compute_qf = UInt32(i - 1)
                break
            end
        end
    end
    compute_qf === nothing && error("No compute queue family found on $device_name")

    # Enable buffer device address
    bda_features = Vulkan.PhysicalDeviceBufferDeviceAddressFeatures(
        true,   # bufferDeviceAddress
        false,  # bufferDeviceAddressCaptureReplay
        false   # bufferDeviceAddressMultiDevice
    )

    device_queue_ci = Vulkan.DeviceQueueCreateInfo(compute_qf, [1.0f0])
    device = unwrap(Vulkan.create_device(pd, [device_queue_ci], [], String[];
        next = bda_features))

    queue = Vulkan.get_device_queue(device, compute_qf, 0)

    cmd_pool = unwrap(Vulkan.create_command_pool(device, compute_qf;
        flags = Vulkan.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT))

    # Pre-allocate a reusable command buffer and fence for dispatch
    alloc_info = Vulkan.CommandBufferAllocateInfo(cmd_pool,
        Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    cmd_buf = unwrap(Vulkan.allocate_command_buffers(device, alloc_info))[1]
    fence = unwrap(Vulkan.create_fence(device))

    ctx = VkContext(instance, pd, device, queue, compute_qf, cmd_pool, device_name,
                    cmd_buf, fence)
    _vk_context[] = ctx
    return ctx
end

function Base.show(io::IO, ctx::VkContext)
    print(io, "VkContext($(ctx.device_name), queue_family=$(ctx.queue_family_index))")
end
