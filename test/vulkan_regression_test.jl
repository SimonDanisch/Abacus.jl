#!/usr/bin/env julia
#
# Vulkan Backend Regression Test Suite
#
# Run with: julia --project=dev/Abacus dev/Abacus/test/vulkan_regression_test.jl
#
# Tests all operations that are known to work on the Vulkan backend.
# Any failure here is a regression that must be fixed before merging.
#
# Each testset documents the Vulkan-specific bug it guards against.

using Test
using Abacus
import AcceleratedKernels

@testset "Vulkan Backend Regression Tests" begin

    # =====================================================================
    @testset "Construction & Roundtrip" begin
        @test Array(VkArray(Float32[1,2,3,4])) == Float32[1,2,3,4]
        @test Array(VkArray(Float32[1 2; 3 4])) == Float32[1 2; 3 4]
        @test Array(VkArray(Int32[10,20,30])) == Int32[10,20,30]

        # zeros / ones via KA
        z = Abacus.VulkanKernels.KA.zeros(Abacus.VulkanKernels.VulkanBackend(), Float32, 8)
        @test all(Array(z) .== 0f0)
    end

    # =====================================================================
    @testset "Broadcast" begin
        a = VkArray(Float32[1,2,3,4])
        b = VkArray(Float32[10,20,30,40])

        @test Array(a .+ b) == Float32[11,22,33,44]
        @test Array(a .- b) == Float32[-9,-18,-27,-36]
        @test Array(a .* 2f0) == Float32[2,4,6,8]
        @test Array(@. a * b + 2a) == Float32[12,44,96,168]

        # Transcendental functions
        x = VkArray(Float32[0, 0.5, 1.0])
        @test all(isapprox.(Array(sin.(x)), sin.(Float32[0, 0.5, 1.0]); atol=1f-6))
        @test all(isapprox.(Array(cos.(x)), cos.(Float32[0, 0.5, 1.0]); atol=1f-6))
    end

    # =====================================================================
    # BUG: GPUArrays reductions used a while-loop with barriers in
    # _reduce_workgroup, causing "Block is already a merge block" in SPIR-V.
    # Vulkan requires structured control flow — while loops with barriers
    # inside produce two OpSelectionMerge targeting the same block.
    # FIX: Compile-time unrolled tree reduction (no loop) + bounded
    # for-loop grid stride in _vk_reduce_kernel (mapreduce.jl).
    @testset "GPUArrays Reductions" begin
        a = VkArray(Float32[1,2,3,4,5])
        @test sum(a) ≈ 15f0
        @test prod(VkArray(Float32[1,2,3,4])) ≈ 24f0

        b = VkArray(Float32[3,1,4,1,5,9])
        @test minimum(b) ≈ 1f0
        @test maximum(b) ≈ 9f0

        # Large reduction (tests two-pass path)
        h = Float32.(1:100_000)
        @test abs(sum(VkArray(h)) - sum(h)) / sum(h) < 0.01
    end

    # =====================================================================
    @testset "GPUArrays map" begin
        a = VkArray(Float32[1,2,3,4])
        @test Array(map(x -> x^2, a)) == Float32[1,4,9,16]
        @test Array(map(x -> x + 1f0, a)) == Float32[2,3,4,5]
    end

    # =====================================================================
    # BUG: Tuple/struct output (e.g. map returning NTuple{3,Float32})
    # caused "array with stride 0" SPIR-V validation error.
    # Julia represents tuples as LLVM array types ([3 x float]). The SPIR-V
    # backend emits OpTypeArray without ArrayStride decoration, which Vulkan
    # requires for PhysicalStorageBuffer access. Additionally, the backend
    # drops indices from flat GEPs on raw BDA pointers.
    # FIX: _flatten_bda_array_geps! in compilation.jl replaces array-typed
    # GEPs with explicit ptrtoint+add+inttoptr pointer arithmetic,
    # completely avoiding OpTypeArray in PhysicalStorageBuffer.
    @testset "Tuple/Struct map" begin
        vk = VkArray(Float32[1,2,3,4])
        vk2 = map(x -> (x^2, sin(x), cos(x)), vk)
        result = Array(vk2)
        for (i, r) in enumerate(result)
            x = Float32(i)
            @test all(isapprox.(r, (x^2, sin(x), cos(x)); atol=1f-5))
        end
    end

    # =====================================================================
    @testset "AcceleratedKernels" begin
        @testset "foreachindex" begin
            v = VkArray(Float32[1,2,3,4])
            AcceleratedKernels.foreachindex(v) do i
                v[i] *= 2f0
            end
            @test Array(v) == Float32[2,4,6,8]
        end

        @testset "foreachindex with capture" begin
            src = VkArray(Float32[10,20,30,40])
            dst = VkArray(zeros(Float32, 4))
            AcceleratedKernels.foreachindex(src) do i
                dst[i] = src[i] + 1f0
            end
            @test Array(dst) == Float32[11,21,31,41]
        end

        @testset "map!" begin
            a = VkArray(Float32[1,2,3,4])
            b = similar(a)
            AcceleratedKernels.map!(x -> x^2, b, a)
            @test Array(b) == Float32[1,4,9,16]
        end

        @testset "reduce" begin
            a = VkArray(Float32[1,2,3,4,5])
            @test AcceleratedKernels.reduce(+, a; init=0f0) ≈ 15f0

            # Large reduce
            h = Float32.(1:10_000)
            r = AcceleratedKernels.reduce(+, VkArray(h); init=0f0)
            @test abs(r - sum(h)) / sum(h) < 0.01
        end

        @testset "sum / prod / extrema" begin
            a = VkArray(Float32[1,2,3,4,5])
            @test AcceleratedKernels.sum(a) ≈ 15f0
            @test AcceleratedKernels.prod(VkArray(Float32[1,2,3,4])) ≈ 24f0
            @test AcceleratedKernels.maximum(VkArray(Float32[3,1,4,1,5])) ≈ 5f0
            @test AcceleratedKernels.minimum(VkArray(Float32[3,1,4,1,5])) ≈ 1f0
        end

        @testset "mapreduce" begin
            a = VkArray(Float32[1,2,3,4])
            @test AcceleratedKernels.mapreduce(x -> x^2, +, a; init=0f0) ≈ 30f0
        end

        # BUG: AK.any/all uses @localmem Int8 (1,) — a tiny shared memory
        # allocation. LLVM optimized [1 x i8] zeroinitializer down to i1 false,
        # which the SPIR-V backend emits as OpConstantFalse. Vulkan only
        # allows OpConstantNull for Workgroup variable initializers.
        # FIX: Changed vk_localmemory IR template from zeroinitializer to
        # undef — prevents LLVM from reducing the type (intrinsics.jl).
        @testset "any / all / count" begin
            v = VkArray(Float32[1,2,3,4,5])
            @test AcceleratedKernels.any(x -> x > 3f0, v) == true
            @test AcceleratedKernels.any(x -> x > 100f0, v) == false
            @test AcceleratedKernels.all(x -> x > 0f0, v) == true
            @test AcceleratedKernels.all(x -> x > 3f0, v) == false
            @test AcceleratedKernels.count(x -> x > 2f0, v) == 3
        end
    end

    # =====================================================================
    # BUG: AK.accumulate! produced NaN due to two shared memory GEP issues:
    #   Pattern A: ConstantExpr GEP bases with negative indices — LLVM folds
    #     GEP(@shared, 0, i-1) into GEP(ConstGEP(@shared, -1, N-1), i).
    #     SPIR-V OpAccessChain cannot represent negative array indices.
    #   Pattern B: Flat GEP chains on addrspace(3) — GEP(T, %structured_gep, offset)
    #     silently breaks in the SPIR-V backend.
    # FIX: _fix_shared_geps! in compilation.jl rewrites both patterns into
    # structured GEPs with correct non-negative indices.
    @testset "AcceleratedKernels accumulate!" begin
        v = VkArray(Float32[1,2,3,4,5])
        AcceleratedKernels.accumulate!(+, v; init=0f0)
        @test Array(v) ≈ Float32[1,3,6,10,15]

        # Large accumulate (tests multi-block Blelloch scan)
        h = rand(Float32, 50_000)
        g = VkArray(copy(h))
        AcceleratedKernels.accumulate!(+, g; init=0f0)
        @test isapprox(Array(g), cumsum(h); rtol=1e-3)
    end

    # =====================================================================
    # BUG: AK.sort! crashed with LLVM freeze instruction not supported by
    #   SPIR-V backend (IRTranslator::translateFreeze). Also produced wrong
    #   results due to the shared memory GEP issues described above.
    # FIX: _replace_freeze! in compilation.jl replaces freeze instructions
    # with their operands (safe — SPIR-V has no poison/undef semantics).
    # Shared memory fixes (Pattern A + B) fixed correctness.
    @testset "AcceleratedKernels sort!" begin
        v = VkArray(Float32[3,1,4,1,5,9,2,6])
        AcceleratedKernels.sort!(v)
        @test Array(v) == Float32[1,1,2,3,4,5,6,9]

        # Medium sort
        h = rand(Float32, 10_000)
        g = VkArray(copy(h))
        AcceleratedKernels.sort!(g)
        @test Array(g) == sort(h)

        # Large sort
        h2 = rand(Float32, 100_000)
        g2 = VkArray(copy(h2))
        AcceleratedKernels.sort!(g2)
        @test Array(g2) ≈ sort(h2)
    end

    # =====================================================================
    @testset "AcceleratedKernels sortperm!" begin
        h = rand(Float32, 1000)
        g = VkArray(copy(h))
        ix = VkArray(collect(1:1000))
        AcceleratedKernels.sortperm!(ix, g)
        @test Array(ix) == sortperm(h)
    end

    # =====================================================================
    @testset "AcceleratedKernels merge_sort_by_key!" begin
        keys_h = rand(Float32, 5000)
        vals_h = collect(Int32, 1:5000)
        keys_g = VkArray(copy(keys_h))
        vals_g = VkArray(copy(vals_h))
        AcceleratedKernels.merge_sort_by_key!(keys_g, vals_g)
        perm = sortperm(keys_h)
        @test isapprox(Array(keys_g), keys_h[perm])
        @test Array(vals_g) == Int32.(vals_h[perm])
    end

    # =====================================================================
    @testset "AcceleratedKernels searchsortedfirst" begin
        sorted = VkArray(Float32.(1:100))
        needles = VkArray(Float32[5.5, 10.0, 50.5, 99.0])
        result = AcceleratedKernels.searchsortedfirst(sorted, needles)
        expected = [searchsortedfirst(Float32.(1:100), n) for n in Float32[5.5, 10.0, 50.5, 99.0]]
        @test Array(result) == expected
    end

    # =====================================================================
    @testset "KernelAbstractions matmul" begin
        import KernelAbstractions as KA
        @kernel function matmul_kernel!(output, a, b)
            i, j = @index(Global, NTuple)
            tmp_sum = zero(eltype(output))
            for k in 1:size(a)[2]
                tmp_sum += a[i, k] * b[k, j]
            end
            output[i, j] = tmp_sum
        end
        backend = Abacus.VulkanKernels.VulkanBackend()
        a = VkArray(rand(Float32, 64, 32))
        b = VkArray(rand(Float32, 32, 16))
        output = KA.zeros(backend, Float32, 64, 16)
        matmul_kernel!(backend)(output, a, b, ndrange=(64, 16))
        KA.synchronize(backend)
        @test isapprox(Array(output), Array(a) * Array(b))
    end
end
