using Abacus
using Test

# Load GPUArrays TestSuite
const GPUArraysTestSuite = let
    mod = @eval module $(gensym())
        using ..Test
        import GPUArrays
        gpuarrays = pathof(GPUArrays)
        gpuarrays_root = dirname(dirname(gpuarrays))
        include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
    end
    mod.TestSuite
end

# Only support types that work with our Vulkan SPIR-V backend
function GPUArraysTestSuite.supported_eltypes(::Type{<:VkArray})
    return (Float32, Int32, Int64)
end

testf(f, xs...; kwargs...) = GPUArraysTestSuite.compare(f, VkArray, xs...; kwargs...)

# Test suites that are known to work well
const STABLE_SUITES = [
    "constructors",
    "base",
    "indexing scalar",
    "reductions/== isequal",
    "reductions/sum prod",
    "reductions/minimum maximum extrema",
    "reductions/any all count",
    "reductions/mapreduce",
    "reductions/reduce",
    "reductions/reducedim!",
    "reductions/mapreducedim!",
]

# Test suites that may crash the RADV driver with complex SPIR-V
const UNSTABLE_SUITES = [
    "broadcasting",           # some complex broadcasts segfault RADV
]

# Test suites that require features we don't support yet
const SKIPPED_SUITES = [
    "random",                 # needs GPU RNG
    "linalg/core",            # needs matmul on GPU
    "linalg/mul!/matrix-matrix",
    "linalg/mul!/vector-matrix",
    "linalg/NaN_false",
    "linalg/norm",
    "math/intrinsics",        # may need special SPIR-V intrinsics
    "math/power",
    "statistics",
    "uniformscaling",
    "vectors",
    "indexing multidimensional",
    "indexing find",
    "ext/jld2",
    "alloc cache",
    "reductions/mapreducedim!_large",
]

# Run stable test suites
@testset "VkArray GPUArrays TestSuite" begin
    for name in STABLE_SUITES
        if haskey(GPUArraysTestSuite.tests, name)
            @testset "$name" begin
                GPUArraysTestSuite.tests[name](VkArray)
            end
        end
    end
end
