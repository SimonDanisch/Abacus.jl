#!/usr/bin/env julia
#
# Abacus Vulkan Backend — Test Suite
#
# Run all stages:     julia --project=. test/runtests.jl
# Run stage 1 only:   julia --project=. test/runtests.jl 1
# Run stage 2 only:   julia --project=. test/runtests.jl 2
# Run stage 3 only:   julia --project=. test/runtests.jl 3
#
# Test Tiers (ordered by complexity — run in order):
#
#   Stage 1: SPIR-V MWE Regression Tests (spirv_mwe_tests.jl)
#            Minimal working examples for each known-problematic SPIR-V pattern.
#            These isolate specific compilation bugs and must ALL pass first.
#
#   Stage 2: GPUArrays Test Suite (gpuarrays_testsuite.jl)
#            Standard GPUArrays conformance: broadcast, reductions, indexing, linalg.
#            Tests GPUArrays interface compliance (AbstractGPUArray contract).
#
#   Stage 3: Ray Tracing Integration (raytracing_test.jl)
#            End-to-end rendering on Vulkan: scene build, BVH, materials, integrators.
#            Requires the main RayTracing project environment.
#
# Adding new tests:
#   - Found a new SPIR-V compilation bug? Add an MWE to spirv_mwe_tests.jl.
#   - New GPUArrays operation works? Remove it from the SKIP set in gpuarrays_testsuite.jl.
#   - New rendering feature? Add a test to raytracing_test.jl.

stages = if length(ARGS) > 0
    [parse(Int, a) for a in ARGS]
else
    [1, 2]  # Don't include stage 3 by default (needs main project env)
end

for stage in stages
    if stage == 1
        println("\n", "="^70)
        println("  Stage 1: SPIR-V MWE Regression Tests")
        println("="^70, "\n")
        include("spirv_mwe_tests.jl")
    elseif stage == 2
        println("\n", "="^70)
        println("  Stage 2: GPUArrays Test Suite")
        println("="^70, "\n")
        include("gpuarrays_testsuite.jl")
    elseif stage == 3
        println("\n", "="^70)
        println("  Stage 3: Ray Tracing Integration")
        println("="^70, "\n")
        include("raytracing_test.jl")
    else
        @warn "Unknown stage: $stage (valid: 1, 2, 3)"
    end
end
