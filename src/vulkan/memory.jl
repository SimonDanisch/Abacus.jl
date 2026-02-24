# Vulkan buffer memory management with buffer device addresses (BDA)
#
# Device buffers live in DEVICE_LOCAL VRAM for full GPU bandwidth.
# Host↔device transfers go through a reusable staging buffer (HOST_VISIBLE).
# Mirrors AMDGPU's HIPBuffer pattern: device alloc + upload!/download! API.

export VkManagedBuffer

mutable struct VkManagedBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    address::UInt64
    size::Int

    function VkManagedBuffer(buffer, memory, address, size)
        obj = new(buffer, memory, address, size)
        finalizer(_vk_free!, obj)
        return obj
    end
end

Base.sizeof(buf::VkManagedBuffer) = buf.size

function _vk_free!(buf::VkManagedBuffer)
    buf.size = 0
    return
end

# --- Staging buffer (reusable, host-visible, grows as needed) ---

mutable struct VkStagingBuffer
    buffer::Vulkan.Buffer
    memory::Vulkan.DeviceMemory
    mapped_ptr::Ptr{Nothing}
    size::Int
end

const _staging_buf = Ref{Union{Nothing, VkStagingBuffer}}(nothing)

"""Get or grow a reusable host-visible staging buffer of at least `bytes` size."""
function _get_staging(bytes::Int)
    sb = _staging_buf[]
    if sb !== nothing && sb.size >= bytes
        return sb
    end
    alloc_size = max(nextpow(2, bytes), 1 << 20)  # min 1MB, power-of-2
    sb = _alloc_staging(alloc_size)
    _staging_buf[] = sb
    return sb
end

function _alloc_staging(bytes::Int)
    ctx = vk_context()
    dev = ctx.device
    pd = ctx.physical_device

    buf = unwrap(Vulkan.create_buffer(dev, Vulkan.BufferCreateInfo(
        bytes,
        Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[]
    )))

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)
    mem_type_idx = _find_memory_type(pd, mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        Vulkan.MEMORY_PROPERTY_HOST_COHERENT_BIT)
    mem_type_idx === nothing && error("No host-visible coherent memory type found")

    memory = unwrap(Vulkan.allocate_memory(dev, mem_reqs.size, mem_type_idx))
    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))
    mapped_ptr = unwrap(Vulkan.map_memory(dev, memory, 0, bytes))

    return VkStagingBuffer(buf, memory, mapped_ptr, bytes)
end

# --- Device-local allocation ---

"""
    vk_alloc(bytes::Integer) -> VkManagedBuffer

Allocate a Vulkan buffer in device-local VRAM with storage + BDA usage.
"""
function vk_alloc(bytes::Integer)
    alloc_bytes = max(Int(bytes), 4)

    ctx = vk_context()
    dev = ctx.device
    pd = ctx.physical_device

    buf = unwrap(Vulkan.create_buffer(dev, Vulkan.BufferCreateInfo(
        alloc_bytes,
        Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT |
            Vulkan.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_SRC_BIT |
            Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT,
        Vulkan.SHARING_MODE_EXCLUSIVE,
        UInt32[]
    )))

    mem_reqs = Vulkan.get_buffer_memory_requirements(dev, buf)

    mem_type_idx = _find_memory_type(pd, mem_reqs.memory_type_bits,
        Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    mem_type_idx === nothing && error("No device-local memory type found")

    alloc_flags = Vulkan.MemoryAllocateFlagsInfo(UInt32(0);
        flags = Vulkan.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    memory = unwrap(Vulkan.allocate_memory(dev, mem_reqs.size, mem_type_idx;
        next = alloc_flags))

    unwrap(Vulkan.bind_buffer_memory(dev, buf, memory, 0))

    address = Vulkan.get_buffer_device_address(dev,
        Vulkan.BufferDeviceAddressInfo(buf))

    return VkManagedBuffer(buf, memory, address, Int(alloc_bytes))
end

"""Find a memory type index matching the required property flags."""
function _find_memory_type(pd, type_bits::UInt32, required)
    mem_props = Vulkan.get_physical_device_memory_properties(pd)
    for i in 0:(mem_props.memory_type_count - 1)
        if (type_bits & (1 << i)) != 0
            mt = mem_props.memory_types[i + 1]
            if (mt.property_flags & required) == required
                return UInt32(i)
            end
        end
    end
    return nothing
end

# --- Transfer operations (mirrors AMDGPU's upload!/download!/transfer!) ---

"""
    upload!(dst::VkManagedBuffer, dst_offset::Int, src::Ptr, nbytes::Int)

Host→device transfer via staging buffer. Like hipMemcpyHtoD.
"""
function upload!(dst::VkManagedBuffer, dst_offset::Int, src::Ptr, nbytes::Int)
    nbytes == 0 && return
    staging = _get_staging(nbytes)
    unsafe_copyto!(Ptr{UInt8}(staging.mapped_ptr), Ptr{UInt8}(src), nbytes)
    _submit_copy!(staging.buffer, UInt64(0), dst.buffer, UInt64(dst_offset), UInt64(nbytes))
end

"""
    download!(dst::Ptr, src::VkManagedBuffer, src_offset::Int, nbytes::Int)

Device→host transfer via staging buffer. Like hipMemcpyDtoH.
"""
function download!(dst::Ptr, src::VkManagedBuffer, src_offset::Int, nbytes::Int)
    nbytes == 0 && return
    staging = _get_staging(nbytes)
    _submit_copy!(src.buffer, UInt64(src_offset), staging.buffer, UInt64(0), UInt64(nbytes))
    unsafe_copyto!(Ptr{UInt8}(dst), Ptr{UInt8}(staging.mapped_ptr), nbytes)
end

"""
    transfer!(dst::VkManagedBuffer, dst_offset::Int,
              src::VkManagedBuffer, src_offset::Int, nbytes::Int)

Device→device transfer via vkCmdCopyBuffer. Like hipMemcpyDtoD.
"""
function transfer!(dst::VkManagedBuffer, dst_offset::Int,
                   src::VkManagedBuffer, src_offset::Int, nbytes::Int)
    nbytes == 0 && return
    _submit_copy!(src.buffer, UInt64(src_offset), dst.buffer, UInt64(dst_offset), UInt64(nbytes))
end

"""Submit a vkCmdCopyBuffer and wait (blocking)."""
function _submit_copy!(src::Vulkan.Buffer, src_offset::UInt64,
                       dst::Vulkan.Buffer, dst_offset::UInt64, size::UInt64)
    ctx = vk_context()
    dev = ctx.device
    cmd = ctx.cmd_buf
    fence = ctx.fence

    unwrap(Vulkan.begin_command_buffer(cmd,
        Vulkan.CommandBufferBeginInfo(;
            flags = Vulkan.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)))

    region = Vulkan.BufferCopy(src_offset, dst_offset, size)
    Vulkan.cmd_copy_buffer(cmd, src, dst, [region])

    unwrap(Vulkan.end_command_buffer(cmd))

    submit_info = Vulkan.SubmitInfo(Vulkan.Semaphore[], UInt32[],
                                     [cmd], Vulkan.Semaphore[])
    unwrap(Vulkan.queue_submit(ctx.queue, [submit_info]; fence))
    unwrap(Vulkan.wait_for_fences(dev, [fence], true, typemax(UInt64)))
    unwrap(Vulkan.reset_fences(dev, [fence]))
end
