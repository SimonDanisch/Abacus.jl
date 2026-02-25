# AcceleratedKernels: Vulkan vs AMDGPU Benchmark
#
# Compares AcceleratedKernels performance on the same GPU via two backends:
#   - Abacus Vulkan (SPIR-V → RADV NIR → ACO)
#   - AMDGPU/ROCm   (native GCN ISA via ROCm)
#
# Run with: julia --project=dev dev/Abacus/test/vulkan_ak_benchmark.jl

using Abacus, AMDGPU, Printf
import AcceleratedKernels as AK
import KernelAbstractions as KA

# ── Helpers ─────────────────────────────────────────────────────────────

function median_time(f; nruns=20, warmup=3)
    for _ in 1:warmup; f(); end
    times = [(@elapsed f()) for _ in 1:nruns]
    sort!(times)
    return times[nruns ÷ 2]
end

function fmt_time(t)
    if t < 1e-3
        return @sprintf("%7.1f μs", t * 1e6)
    else
        return @sprintf("%7.2f ms", t * 1e3)
    end
end

function print_result(name, vk_t, roc_t; bytes=0)
    bw_vk  = bytes > 0 ? @sprintf("  %5.1f GB/s", bytes / vk_t / 1e9) : ""
    bw_roc = bytes > 0 ? @sprintf("  %5.1f GB/s", bytes / roc_t / 1e9) : ""
    ratio = @sprintf("%.2fx", vk_t / roc_t)
    println("  $(rpad(name, 30)) Vulkan $(fmt_time(vk_t))$bw_vk | AMDGPU $(fmt_time(roc_t))$bw_roc | $ratio")
end

# ── Benchmark functions ─────────────────────────────────────────────────

function bench_reduce(N; nruns=20)
    data = rand(Float32, N)
    vk = Abacus.VkArray(data)
    roc = ROCArray(data)

    vk_t = median_time(; nruns) do
        AK.reduce(+, vk; init=0f0)
    end
    roc_t = median_time(; nruns) do
        AK.reduce(+, roc; init=0f0)
        AMDGPU.synchronize()
    end
    print_result("reduce N=$N", vk_t, roc_t; bytes=N*4)
end

function bench_mapreduce(N; nruns=20)
    data = rand(Float32, N)
    vk = Abacus.VkArray(data)
    roc = ROCArray(data)

    vk_t = median_time(; nruns) do
        AK.mapreduce(x -> x^2, +, vk; init=0f0)
    end
    roc_t = median_time(; nruns) do
        AK.mapreduce(x -> x^2, +, roc; init=0f0)
        AMDGPU.synchronize()
    end
    print_result("mapreduce N=$N", vk_t, roc_t; bytes=N*4)
end

function bench_accumulate(N; nruns=20)
    data = rand(Float32, N)

    vk_t = median_time(; nruns) do
        vk = Abacus.VkArray(copy(data))
        AK.accumulate!(+, vk; init=0f0)
        KA.synchronize(Abacus.VulkanKernels.VulkanBackend())
    end
    roc_t = median_time(; nruns) do
        roc = ROCArray(copy(data))
        AK.accumulate!(+, roc; init=0f0)
        AMDGPU.synchronize()
    end
    print_result("accumulate N=$N", vk_t, roc_t; bytes=N*4*2)  # read+write
end

function bench_sort(N; nruns=20)
    data = rand(Float32, N)

    vk_t = median_time(; nruns) do
        vk = Abacus.VkArray(copy(data))
        AK.sort!(vk)
        KA.synchronize(Abacus.VulkanKernels.VulkanBackend())
    end
    roc_t = median_time(; nruns) do
        roc = ROCArray(copy(data))
        AK.sort!(roc)
        AMDGPU.synchronize()
    end
    print_result("sort N=$N", vk_t, roc_t)
end

function bench_sortperm(N; nruns=20)
    data = rand(Float32, N)

    vk_t = median_time(; nruns) do
        vk = Abacus.VkArray(copy(data))
        ix = Abacus.VkArray(collect(1:N))
        AK.sortperm!(ix, vk)
        KA.synchronize(Abacus.VulkanKernels.VulkanBackend())
    end
    roc_t = median_time(; nruns) do
        roc = ROCArray(copy(data))
        ix = ROCArray(collect(1:N))
        AK.sortperm!(ix, roc)
        AMDGPU.synchronize()
    end
    print_result("sortperm N=$N", vk_t, roc_t)
end

function bench_searchsortedfirst(N, M; nruns=20)
    sorted = Float32.(1:N)
    needles = rand(Float32, M) .* N
    vk_s = Abacus.VkArray(sorted)
    vk_n = Abacus.VkArray(needles)
    roc_s = ROCArray(sorted)
    roc_n = ROCArray(needles)

    vk_t = median_time(; nruns) do
        AK.searchsortedfirst(vk_s, vk_n)
        KA.synchronize(Abacus.VulkanKernels.VulkanBackend())
    end
    roc_t = median_time(; nruns) do
        AK.searchsortedfirst(roc_s, roc_n)
        AMDGPU.synchronize()
    end
    print_result("searchsortedfirst N=$N M=$M", vk_t, roc_t)
end

function bench_foreachindex(N; nruns=20)
    data = rand(Float32, N)

    vk_t = median_time(; nruns) do
        vk = Abacus.VkArray(copy(data))
        AK.foreachindex(vk) do i
            @inbounds vk[i] *= 2f0
        end
        KA.synchronize(Abacus.VulkanKernels.VulkanBackend())
    end
    roc_t = median_time(; nruns) do
        roc = ROCArray(copy(data))
        AK.foreachindex(roc) do i
            @inbounds roc[i] *= 2f0
        end
        AMDGPU.synchronize()
    end
    print_result("foreachindex N=$N", vk_t, roc_t; bytes=N*4*2)
end

function bench_any_all(N; nruns=20)
    data = rand(Float32, N)
    vk = Abacus.VkArray(data)
    roc = ROCArray(data)

    vk_t = median_time(; nruns) do
        AK.any(x -> x > 0.5f0, vk)
    end
    roc_t = median_time(; nruns) do
        AK.any(x -> x > 0.5f0, roc)
        AMDGPU.synchronize()
    end
    print_result("any N=$N", vk_t, roc_t; bytes=N*4)
end

function bench_matmul(M, K, N; nruns=20)
    a_data = rand(Float32, M, K)
    b_data = rand(Float32, K, N)

    @kernel function matmul_k!(out, a, b)
        i, j = @index(Global, NTuple)
        s = zero(eltype(out))
        for k in 1:size(a)[2]
            s += a[i, k] * b[k, j]
        end
        out[i, j] = s
    end

    vk_backend = Abacus.VulkanKernels.VulkanBackend()
    roc_backend = ROCBackend()

    a_vk = Abacus.VkArray(a_data); b_vk = Abacus.VkArray(b_data)
    o_vk = KA.zeros(vk_backend, Float32, M, N)
    a_roc = ROCArray(a_data); b_roc = ROCArray(b_data)
    o_roc = KA.zeros(roc_backend, Float32, M, N)

    vk_t = median_time(; nruns) do
        matmul_k!(vk_backend)(o_vk, a_vk, b_vk, ndrange=(M, N))
        KA.synchronize(vk_backend)
    end
    roc_t = median_time(; nruns) do
        matmul_k!(roc_backend)(o_roc, a_roc, b_roc, ndrange=(M, N))
        KA.synchronize(roc_backend)
    end
    gflops_vk  = 2.0 * M * K * N / vk_t / 1e9
    gflops_roc = 2.0 * M * K * N / roc_t / 1e9
    ratio = @sprintf("%.2fx", vk_t / roc_t)
    println("  $(rpad("matmul $(M)x$(K) * $(K)x$(N)", 30)) Vulkan $(fmt_time(vk_t))  $(round(gflops_vk, digits=1)) GFLOP/s | AMDGPU $(fmt_time(roc_t))  $(round(gflops_roc, digits=1)) GFLOP/s | $ratio")
end

# ── Main ────────────────────────────────────────────────────────────────

ctx = Abacus.vk_context()
println("Vulkan: ", ctx.device_name)
println("AMDGPU: ", AMDGPU.device())
println()

println("═"^90)
println("  AcceleratedKernels: Vulkan (RADV) vs AMDGPU (ROCm)")
println("═"^90)

println("\n── Reductions ──")
for N in [10_000, 100_000, 1_000_000, 10_000_000]
    bench_reduce(N)
end

println("\n── MapReduce ──")
for N in [10_000, 100_000, 1_000_000, 10_000_000]
    bench_mapreduce(N)
end

println("\n── Accumulate (prefix sum) ──")
for N in [10_000, 100_000, 1_000_000]
    bench_accumulate(N)
end

println("\n── Sort ──")
for N in [1_000, 10_000, 100_000, 1_000_000]
    bench_sort(N)
end

println("\n── Sortperm ──")
for N in [1_000, 10_000, 100_000]
    bench_sortperm(N)
end

println("\n── SearchSortedFirst ──")
bench_searchsortedfirst(100_000, 10_000)
bench_searchsortedfirst(1_000_000, 100_000)

println("\n── Foreachindex ──")
for N in [10_000, 100_000, 1_000_000, 10_000_000]
    bench_foreachindex(N)
end

println("\n── Any ──")
for N in [10_000, 100_000, 1_000_000]
    bench_any_all(N)
end

println("\n── KA Matmul ──")
bench_matmul(128, 128, 128)
bench_matmul(256, 256, 256)
bench_matmul(512, 512, 512)

println("\n", "═"^90)
println("  Done.")
println("═"^90)
