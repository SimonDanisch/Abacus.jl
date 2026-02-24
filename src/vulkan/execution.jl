# Vulkan kernel execution — @vulkan macro and kernel launch

export @vulkan

"""
    @vulkan f(args...)

Compile (cached) and execute a Julia function as a Vulkan compute shader.
"""
macro vulkan(ex...)
    call = ex[end]
    kwargs = ex[1:end-1]

    Meta.isexpr(call, :call) || throw(ArgumentError("@vulkan requires a function call"))
    f = call.args[1]
    args = call.args[2:end]

    quote
        local kernel_f = $(esc(f))
        local kernel_args = ($(map(esc, args)...),)

        # Adapt args: VkArray → UInt64 BDA, scalars pass through
        local adapted_args = map(kernel_convert, kernel_args)
        local adapted_tt = Tuple{map(Core.Typeof, adapted_args)...}

        # Compile (cached via vkfunction)
        local kernel = vkfunction(kernel_f, adapted_tt)

        # Pack push constants
        local push_data = _pack_push_constants(adapted_args, kernel.push_size)

        # Dispatch
        local ndrange = _infer_ndrange(kernel_args)
        local groups = (cld(ndrange, 64), 1, 1)

        GC.@preserve kernel_args begin
            vk_dispatch!(kernel.pipeline, push_data, groups)
        end

        nothing
    end
end

"""Pack kernel arguments into push constant bytes.
Recursively flattens struct fields so the byte layout matches the
LLVM push constant struct (which decomposes structs into scalars to
avoid composite loads that crash RADV)."""
function _pack_push_constants(args::Tuple, push_size::Int)
    data = zeros(UInt8, push_size)
    offset = Ref(0)
    for arg in args
        _pack_arg!(data, offset, arg)
    end
    return data
end

function _pack_arg!(data::Vector{UInt8}, offset::Ref{Int}, arg)
    T = typeof(arg)
    if isprimitivetype(T)
        sz = sizeof(arg)
        align = sz
        offset[] = (offset[] + align - 1) & ~(align - 1)
        ref = Ref(arg)
        GC.@preserve ref begin
            unsafe_copyto!(Ptr{UInt8}(pointer(data, offset[] + 1)),
                          Ptr{UInt8}(pointer_from_objref(ref)), sz)
        end
        offset[] += sz
    elseif isstructtype(T) && isbitstype(T)
        # Recursively flatten struct fields to match LLVM layout
        for i in 1:fieldcount(T)
            _pack_arg!(data, offset, getfield(arg, i))
        end
    else
        error("Cannot pack non-isbits type $(T) into push constants")
    end
end

"""Infer ndrange from the first array argument."""
function _infer_ndrange(args::Tuple)
    for arg in args
        if arg isa VkArray
            return length(arg)
        end
    end
    return 64  # default
end
