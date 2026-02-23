module Abacus

using GPUCompiler
using LLVM
using LLVM.Interop
using Adapt
using ExprTools: splitdef, combinedef
using StaticArrays
using GPUArrays
using GPUArraysCore
using Libdl

import KernelAbstractions

# device utilities (method table, @device_override)
include("device/utils.jl")

# device runtime (KernelState, runtime stubs)
include("device/runtime.jl")

# device intrinsics (thread index functions)
include("device/intrinsics.jl")

# device quirks (@device_override for error-throwing functions)
include("device/quirks.jl")

# device math (Float32-only LLVM intrinsic overrides, avoids Float64 promotion)
include("device/math.jl")

# array type
include("array.jl")

# compiler
include("compiler/compilation.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")

# KernelAbstractions backend
include("AbacusKernels.jl")
import .AbacusKernels: AbacusBackend

export AbacusBackend, AbacusArray, AbacusVector, AbacusMatrix, AbacusDeviceArray, @abacus

function __init__()
    # Allocate a fresh per-thread kernel state buffer. This must run at module-load
    # time (not precompile time) so the pointer is valid in the current process.
    _init_kernel_states!()
end

end # module Abacus
