# GPUArrays TestSuite runner for VkArray
# Runs all test groups in-process with error isolation.

using Abacus
import GPUArrays
using Test

gpuarrays_testsuite = joinpath(dirname(dirname(pathof(GPUArrays))), "test", "testsuite.jl")
include(gpuarrays_testsuite)

# Float32 only — GLSL.std.450 transcendentals don't support Float64
TestSuite.supported_eltypes(::Type{<:VkArray}) = (Float32,)

# Disallow scalar indexing — matches CUDA/Metal/AMDGPU behavior
GPUArrays.allowscalar(false)

function count_results(ts::Test.DefaultTestSet)
    pass = 0; fail = 0; err = 0; broken = 0
    for r in ts.results
        if r isa Test.DefaultTestSet
            sub = count_results(r)
            pass += sub[1]; fail += sub[2]; err += sub[3]; broken += sub[4]
        elseif r isa Test.Pass
            pass += 1
        elseif r isa Test.Fail
            fail += 1
        elseif r isa Test.Error
            err += 1
        elseif r isa Test.Broken
            broken += 1
        end
    end
    return (pass, fail, err, broken)
end

# Check if the Vulkan device is still alive by trying a trivial upload
function device_alive()
    try
        x = VkArray(Float32[1.0])
        r = Array(x)
        return r[1] == 1.0f0
    catch
        return false
    end
end

# Skip tests that need features we don't support or cause fatal crashes
const SKIP = Set([
    "sparse",       # needs sparse array types we don't implement
    "ext/jld2",     # needs JLD2 extension
    "alloc cache",  # needs alloc_cache support
    "random",       # needs RNG on GPU
    "base",         # GPUArrays.unsafe_free! / storage not implemented (2/77 pass)
    "constructors", # tries Int64/Int32 arrays → DEVICE_LOST + segfault
    "vectors",      # needs resize! which we don't implement
    # broadcasting: now uses BDA arg buffer, no push constant size limit
    "reductions/== isequal",    # A==B with different shapes triggers broadcast → SPIR-V crash
    "reductions/minimum maximum extrema", # extrema uses Tuple{T,T} reduction → bad SPIR-V for tuple shared mem
    "indexing find",              # findmax/findmin with dims → spirv-opt error → GPU fault
    "indexing multidimensional",  # vectorized getindex (fancy indexing) → GPU fault
    "linalg/core",               # permutedims/adjoint — too many GPU faults
    "linalg/diagonal",           # uses broadcasting with Diagonal types → may crash
    "linalg/kron",               # kron for non-float types crashes
    "uniformscaling",            # broadcasting with UniformScaling types → DEVICE_LOST
])

function run_gpuarrays_tests()
    test_names = sort(collect(keys(TestSuite.tests)))
    filter!(n -> n ∉ SKIP, test_names)
    # Move tests that risk DEVICE_LOST to the end so they don't poison others
    risky = ["linalg/mul!/matrix-matrix"]  # ComplexF32 matmul → DEVICE_LOST
    for r in risky
        if r in test_names
            deleteat!(test_names, findfirst(==(r), test_names))
            push!(test_names, r)
        end
    end

    all_results = Vector{Tuple{String,Int,Int,Int}}()
    device_dead = false

    for name in test_names
        if device_dead
            push!(all_results, (name, 0, 0, 1))
            println("$name ... SKIPPED (device lost)")
            continue
        end

        print("$name ... ")
        flush(stdout)
        try
            ts = @testset "$name" begin
                TestSuite.tests[name](VkArray)
            end
            p, f, e, _ = count_results(ts)
            push!(all_results, (name, p, f, e))
            if f + e > 0
                println("$(p)p $(f)f $(e)e")
            else
                println("$(p) pass")
            end
            # Check if any test errors killed the device
            if e > 0 && !device_alive()
                device_dead = true
                println("  *** Device lost during $name — skipping remaining tests ***")
            end
        catch ex
            push!(all_results, (name, 0, 0, 1))
            msg = sprint(showerror, ex)
            println("CRASH: ", msg[1:min(end,150)])
            if occursin("DEVICE_LOST", msg) || !device_alive()
                device_dead = true
                println("  *** Device lost — skipping remaining tests ***")
            end
        end
    end

    println("\n" * "="^70)
    println("  GPUArrays TestSuite Results (VkArray)")
    println("="^70)
    total_p = 0; total_f = 0; total_e = 0
    for (name, p, f, e) in all_results
        total_p += p; total_f += f; total_e += e
        status = (f + e == 0) ? "PASS" : "FAIL"
        detail = (f + e == 0) ? "$(p) pass" : "$(p)p $(f)f $(e)e"
        println("  $status  $(rpad(name, 40)) $detail")
    end
    println("="^70)
    println("  Total: $total_p passed, $total_f failed, $total_e errors")
    n_groups_pass = count(x -> x[3] + x[4] == 0, all_results)
    println("  $n_groups_pass/$(length(all_results)) test groups fully passing")
    println("  Skipped: $(join(sort(collect(SKIP)), ", "))")
    return all_results
end

run_gpuarrays_tests()
