# Vulkan vs AMDGPU sum() benchmark
#
# Compares reduction performance on the same GPU via two backends:
#   - Abacus Vulkan (SPIR-V → RADV NIR → ACO)
#   - AMDGPU/ROCm   (native GCN ISA via ROCm)

using Abacus, AMDGPU

function bench_sum(N; nruns=20)
    data = rand(Float32, N)
    a_vk  = Abacus.VkArray(data)
    a_roc = ROCArray(data)

    # Warm up
    sum(a_vk); sum(a_roc); AMDGPU.synchronize()

    # Vulkan
    vk_times = [(@elapsed sum(a_vk)) for _ in 1:nruns]

    # AMDGPU
    roc_times = Float64[]
    for _ in 1:nruns
        AMDGPU.synchronize()
        t = @elapsed begin
            sum(a_roc)
            AMDGPU.synchronize()
        end
        push!(roc_times, t)
    end

    sort!(vk_times); sort!(roc_times)
    vk_med  = vk_times[nruns ÷ 2]
    roc_med = roc_times[nruns ÷ 2]

    bw_vk  = N * 4 / vk_med  / 1e9
    bw_roc = N * 4 / roc_med / 1e9

    println(
        "N=$(lpad(N ÷ 1_000_000, 3))M: " *
        "Vulkan $(lpad(round(vk_med*1e6, digits=1), 8))μs ($(lpad(round(bw_vk, digits=1), 5)) GB/s) | " *
        "AMDGPU $(lpad(round(roc_med*1e6, digits=1), 8))μs ($(lpad(round(bw_roc, digits=1), 5)) GB/s) | " *
        "ratio $(round(vk_med/roc_med, digits=2))x"
    )
end

# --- main ---

ctx = Abacus.vk_context()
println("Vulkan: ", ctx.device_name)
println("AMDGPU: ", AMDGPU.device())
println()

for N in [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    bench_sum(N)
end
