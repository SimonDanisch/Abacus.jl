#!/usr/bin/env julia
#
# Stage 1: SPIR-V MWE (Minimum Working Example) Regression Tests
#
# Run with: julia --project=. test/spirv_mwe_tests.jl
#
# Each test isolates a specific SPIR-V compilation pattern that has caused
# validation errors, driver crashes, or wrong results in the past.
# These must ALL pass before running the GPUArrays or ray tracing test suites.
#
# When a new compilation bug is found, add a focused MWE test here FIRST,
# then fix the bug, then verify the MWE passes.

using Test
using Abacus
import KernelAbstractions as KA
import AcceleratedKernels as AK

const backend = Abacus.VulkanBackend()

function device_alive()
    try
        x = VkArray(Float32[1.0])
        return Array(x)[1] == 1.0f0
    catch
        return false
    end
end

@testset "Stage 1: SPIR-V MWE Regression Tests" begin

    # ─────────────────────────────────────────────────────────────────────
    # Basic compilation & data roundtrip
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.1 — Array roundtrip" begin
        @test Array(VkArray(Float32[1,2,3,4])) == Float32[1,2,3,4]
        @test Array(VkArray(Int32[10,20,30])) == Int32[10,20,30]
        @test Array(VkArray(UInt32[1,2,3])) == UInt32[1,2,3]
        z = KA.zeros(backend, Float32, 8)
        @test all(Array(z) .== 0f0)
    end

    # ─────────────────────────────────────────────────────────────────────
    # Simple KA kernel — tests basic compilation pipeline
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.2 — Simple KA kernel (double)" begin
        @kernel function double_kernel!(a)
            i = @index(Global)
            @inbounds a[i] *= 2f0
        end
        a = VkArray(Float32[1,2,3,4])
        double_kernel!(backend)(a, ndrange=4)
        KA.synchronize(backend)
        @test Array(a) == Float32[2,4,6,8]
    end

    # ─────────────────────────────────────────────────────────────────────
    # BDA pointer arithmetic — VkDeviceArray GEP on PhysicalStorageBuffer
    # BUG: SPIR-V backend dropped indices from flat GEPs on raw BDA pointers.
    # FIX: _flatten_bda_array_geps! in compilation.jl
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.3 — Broadcast (BDA pointer arithmetic)" begin
        a = VkArray(Float32[1,2,3,4])
        b = VkArray(Float32[10,20,30,40])
        @test Array(a .+ b) == Float32[11,22,33,44]
        @test Array(a .* 2f0) == Float32[2,4,6,8]
        @test Array(@. a * b + 2a) == Float32[12,44,96,168]
    end

    # ─────────────────────────────────────────────────────────────────────
    # Transcendental functions — GLSL.std.450 instruction remapping
    # BUG: llvm-spirv emits OpenCL.std instruction numbers; wrong for Vulkan.
    # FIX: fix_ext_inst_import! remaps OpenCL→GLSL instruction numbers.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.4 — Transcendental functions (sin/cos/exp)" begin
        x = VkArray(Float32[0, 0.5, 1.0])
        @test all(isapprox.(Array(sin.(x)), sin.(Float32[0, 0.5, 1.0]); atol=1f-6))
        @test all(isapprox.(Array(cos.(x)), cos.(Float32[0, 0.5, 1.0]); atol=1f-6))
        @test all(isapprox.(Array(exp.(x)), exp.(Float32[0, 0.5, 1.0]); atol=1f-5))
    end

    # ─────────────────────────────────────────────────────────────────────
    # Tuple/struct output from map — tests OpTypeArray in PhysicalStorageBuffer
    # BUG: Julia tuples become LLVM [N x T] arrays. SPIR-V backend emits
    #   OpTypeArray without ArrayStride → "array with stride 0" validation error.
    # FIX: _flatten_bda_array_geps! replaces array GEPs with ptr arithmetic.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.5 — Tuple return from map (struct output)" begin
        vk = VkArray(Float32[1,2,3,4])
        vk2 = map(x -> (x^2, sin(x), cos(x)), vk)
        result = Array(vk2)
        for (i, r) in enumerate(result)
            x = Float32(i)
            @test all(isapprox.(r, (x^2, sin(x), cos(x)); atol=1f-5))
        end
    end

    # ─────────────────────────────────────────────────────────────────────
    # Shared memory — KA @localmem + barrier
    # BUG: KA's gensym IDs (##static_shmem#284) contain '#' which is an
    #   LLVM IR comment delimiter → global name broken → computation eliminated.
    # FIX: Sanitize ID with replace(string(Id), r"[^a-zA-Z0-9_]" => "_").
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.6 — Shared memory (@localmem + barrier)" begin
        @kernel function shmem_kernel!(output, input)
            tid = @index(Local)
            shared = @localmem Float32 (4,)
            @inbounds shared[tid] = input[tid]
            @synchronize()
            # Read from reversed position
            @inbounds output[tid] = shared[5 - tid]
        end
        a = VkArray(Float32[1,2,3,4])
        b = similar(a)
        shmem_kernel!(backend, 4)(b, a, ndrange=4)
        KA.synchronize(backend)
        @test Array(b) == Float32[4,3,2,1]
    end

    # ─────────────────────────────────────────────────────────────────────
    # Custom reduction — tests compile-time unrolled tree reduction
    # BUG: GPUArrays default reduction uses while-loop with barriers →
    #   "Block is already a merge block" in SPIR-V (two OpSelectionMerge
    #   targeting same block). Also, view-of-view offset bug in derive.
    # FIX: Custom mapreduce.jl with unrolled tree + bounded for-loop.
    #   GPUArrays.derive offset accumulation fix in array.jl.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.7 — Reduction (sum, small + large)" begin
        @test sum(VkArray(Float32[1,2,3,4,5])) ≈ 15f0
        @test prod(VkArray(Float32[1,2,3,4])) ≈ 24f0
        @test minimum(VkArray(Float32[3,1,4,1,5,9])) ≈ 1f0
        @test maximum(VkArray(Float32[3,1,4,1,5,9])) ≈ 9f0
        # Large reduction (tests two-pass path + derive offset)
        h = Float32.(1:100_000)
        @test abs(sum(VkArray(h)) - sum(h)) / sum(h) < 0.01
    end

    # ─────────────────────────────────────────────────────────────────────
    # Signed integer division — unchecked sdiv/srem
    # BUG: Julia's checked_sdiv_int/checked_srem_int generate OpSwitch with
    #   broken merge blocks in SPIR-V.
    # FIX: @device_override div/rem to use unchecked sdiv_int/srem_int.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.8 — Signed integer division (div/rem)" begin
        a = VkArray(Int32[10, -7, 15, -20])
        b = VkArray(Int32[3, 2, -4, 7])
        @test Array(a .÷ b) == Int32[3, -3, -3, -2]
        @test Array(a .% b) == Int32[1, -1, 3, -6]
    end

    # ─────────────────────────────────────────────────────────────────────
    # AK.accumulate! — shared memory GEP patterns A and B
    # BUG (Pattern A): LLVM folds GEP(@shared, 0, i-1) into
    #   GEP(ConstGEP(@shared, -1, N-1), i) → negative indices in OpAccessChain.
    # BUG (Pattern B): Flat GEP chains on addrspace(3) break in SPIR-V backend.
    # FIX: _fix_shared_geps! rewrites both to structured GEPs.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.9 — Prefix sum (accumulate! — shared memory GEP patterns)" begin
        v = VkArray(Float32[1,2,3,4,5])
        AK.accumulate!(+, v; init=0f0)
        @test Array(v) ≈ Float32[1,3,6,10,15]
        # Multi-block Blelloch scan
        h = rand(Float32, 50_000)
        g = VkArray(copy(h))
        AK.accumulate!(+, g; init=0f0)
        @test isapprox(Array(g), cumsum(h); rtol=1e-3)
    end

    # ─────────────────────────────────────────────────────────────────────
    # AK.sort! — freeze instruction + shared memory
    # BUG: LLVM freeze instruction not supported by SPIR-V IRTranslator.
    # FIX: _replace_freeze! replaces freeze with operand.
    # Also depends on shared memory GEP fixes (Pattern A + B).
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.10 — Sort (freeze + shared memory)" begin
        v = VkArray(Float32[3,1,4,1,5,9,2,6])
        AK.sort!(v)
        @test Array(v) == Float32[1,1,2,3,4,5,6,9]
        # UInt32 sort (different structured control flow patterns)
        u = VkArray(UInt32[5,3,1,4,2,8,7,6])
        AK.sort!(u)
        @test Array(u) == UInt32[1,2,3,4,5,6,7,8]
    end

    # ─────────────────────────────────────────────────────────────────────
    # AK.sort! for Float32 with explicit lt/by — structured control flow
    # BUG: Float32 merge_sort_block kernel has "conditional loop exit from
    #   latch" pattern that generates phantom OpPhi predecessors.
    # FIX: Continue targets allowed bare OpBranchConditional even when
    #   conditionally exiting; trampolined_continues dict prevents duplicate blocks.
    # STATUS: Active development — this test may fail until the fix lands.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.11 — Sort Float32 with lt/by (conditional loop exit)" begin
        a = VkArray(Float32[5,3,1,4,2])
        AK.merge_sort!(a, lt=isless, by=identity)
        @test Array(a) == Float32[1,2,3,4,5]
    end

    # ─────────────────────────────────────────────────────────────────────
    # AK.any / AK.all — tiny shared memory (@localmem Int8 (1,))
    # BUG: LLVM optimized [1 x i8] zeroinitializer → i1 false → OpConstantFalse.
    #   Vulkan only allows OpConstantNull for Workgroup initializers.
    # FIX: Changed vk_localmemory template from zeroinitializer to undef.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.12 — any/all (tiny shared memory)" begin
        v = VkArray(Float32[1,2,3,4,5])
        @test AK.any(x -> x > 3f0, v) == true
        @test AK.any(x -> x > 100f0, v) == false
        @test AK.all(x -> x > 0f0, v) == true
        @test AK.all(x -> x > 3f0, v) == false
        @test AK.count(x -> x > 2f0, v) == 3
    end

    # ─────────────────────────────────────────────────────────────────────
    # 2D KA kernel (matmul) — multi-dimensional indexing
    # Tests @index(Global, NTuple) and 2D ndrange dispatch.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.13 — KA matmul (2D kernel)" begin
        @kernel function matmul_kernel!(output, a, b)
            i, j = @index(Global, NTuple)
            tmp = zero(eltype(output))
            for k in 1:size(a)[2]
                tmp += a[i, k] * b[k, j]
            end
            output[i, j] = tmp
        end
        a = VkArray(rand(Float32, 16, 8))
        b = VkArray(rand(Float32, 8, 12))
        output = KA.zeros(backend, Float32, 16, 12)
        matmul_kernel!(backend)(output, a, b, ndrange=(16, 12))
        KA.synchronize(backend)
        @test isapprox(Array(output), Array(a) * Array(b))
    end

    # ─────────────────────────────────────────────────────────────────────
    # AK.foreachindex with closures — Adapt.jl VkArray adaptation
    # Tests that VkArray arguments are properly adapted to VkDeviceArray
    # when captured in closures passed to AcceleratedKernels.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.14 — foreachindex with closure capture" begin
        src = VkArray(Float32[10,20,30,40])
        dst = VkArray(zeros(Float32, 4))
        AK.foreachindex(src) do i
            dst[i] = src[i] + 1f0
        end
        @test Array(dst) == Float32[11,21,31,41]
    end

    # ─────────────────────────────────────────────────────────────────────
    # AK.sortperm — key+value sort (merge_sort_by_key)
    # Tests index array manipulation alongside data sorting.
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.15 — sortperm + merge_sort_by_key" begin
        h = rand(Float32, 1000)
        g = VkArray(copy(h))
        ix = VkArray(collect(1:1000))
        AK.sortperm!(ix, g)
        @test Array(ix) == sortperm(h)
    end

    # ─────────────────────────────────────────────────────────────────────
    # AK.searchsortedfirst — binary search on GPU
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.16 — searchsortedfirst (binary search)" begin
        sorted = VkArray(Float32.(1:100))
        needles = VkArray(Float32[5.5, 10.0, 50.5, 99.0])
        result = AK.searchsortedfirst(sorted, needles)
        expected = [searchsortedfirst(Float32.(1:100), n) for n in Float32[5.5, 10.0, 50.5, 99.0]]
        @test Array(result) == expected
    end

    # ─────────────────────────────────────────────────────────────────────
    # Large sort (multi-block, 100k elements) — stress test
    # Tests async batch dispatch (156 kernel dispatches in one submit).
    # ─────────────────────────────────────────────────────────────────────
    @testset "1.17 — Large sort (100k, multi-block dispatch)" begin
        h = rand(Float32, 100_000)
        g = VkArray(copy(h))
        AK.sort!(g)
        @test Array(g) == sort(h)
    end

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    if !device_alive()
        @warn "Vulkan device lost during MWE tests — some results may be unreliable"
    end
end
