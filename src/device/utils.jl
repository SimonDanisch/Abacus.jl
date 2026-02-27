# Separate method tables for CPU and Vulkan backends.
# Each backend has its own overlay table so device overrides don't conflict
# (e.g. KA.__synchronize() needs different implementations per backend).
# This mirrors how AMDGPU and OpenCL are separate packages with separate tables.

# CPU (Abacus) backend method table
Base.Experimental.@MethodTable(method_table)

# Vulkan backend method table
Base.Experimental.@MethodTable(vk_method_table)

macro device_override(ex)
    ex = macroexpand(__module__, ex)
    if VERSION >= v"1.12.0-DEV.745" || v"1.11-rc1" <= VERSION < v"1.12-"
        esc(quote
            Base.Experimental.@consistent_overlay($method_table, $ex)
        end)
    else
        esc(quote
            Base.Experimental.@overlay($method_table, $ex)
        end)
    end
end

macro vk_device_override(ex)
    ex = macroexpand(__module__, ex)
    if VERSION >= v"1.12.0-DEV.745" || v"1.11-rc1" <= VERSION < v"1.12-"
        esc(quote
            Base.Experimental.@consistent_overlay($vk_method_table, $ex)
        end)
    else
        esc(quote
            Base.Experimental.@overlay($vk_method_table, $ex)
        end)
    end
end

# Register on BOTH method tables — used for shared overrides (quirks, math)
# that both CPU and Vulkan backends need (error stubs, Float32 math, etc.)
macro shared_device_override(ex)
    ex = macroexpand(__module__, ex)
    if VERSION >= v"1.12.0-DEV.745" || v"1.11-rc1" <= VERSION < v"1.12-"
        esc(quote
            Base.Experimental.@consistent_overlay($method_table, $ex)
            Base.Experimental.@consistent_overlay($vk_method_table, $ex)
        end)
    else
        esc(quote
            Base.Experimental.@overlay($method_table, $ex)
            Base.Experimental.@overlay($vk_method_table, $ex)
        end)
    end
end

macro device_function(ex)
    ex = macroexpand(__module__, ex)
    def = splitdef(ex)

    # generate a function that errors
    def[:body] = quote
        error("This function is not intended for use on the CPU")
    end

    esc(quote
        $(combinedef(def))
        @device_override $ex
    end)
end
