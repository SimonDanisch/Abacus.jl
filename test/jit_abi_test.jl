using Abacus
using Abacus: AbacusDeviceArray, Adaptor, abacusconvert
using GPUCompiler
using Test
import Adapt

@testset "Abacus compilation" begin

    @testset "simple kernel compiles and executes" begin
        arr = AbacusArray{Float32}(undef, 4)
        dev = Adapt.adapt(Adaptor(), arr)
        @abacus (A -> (A[1] = 42.0f0; nothing))(dev)
        @test Array(arr)[1] == 42.0f0
    end

    @testset "multi-threaded kernel with indices" begin
        function fill_indexed(A::AbacusDeviceArray{Float32,1}, val::Float32)
            i = Abacus.thread_position_in_threadgroup_x()
            A[i] = val * Float32(i)
            return nothing
        end
        arr = AbacusArray{Float32}(undef, 4)
        @abacus threads=4 fill_indexed(Adapt.adapt(Adaptor(), arr), 10.0f0)
        @test Array(arr) == Float32[10, 20, 30, 40]
    end

    @testset "Float64 rejected at array construction" begin
        @test_throws ErrorException AbacusArray{Float64}(undef, 4)
    end

    @testset "Float64 rejected in kernel computation" begin
        # Runtime Float64 arithmetic that LLVM can't constant-fold
        function bad_f64_runtime(A::AbacusDeviceArray{Float32,1})
            x = Float64(A[1])
            A[1] = Float32(sin(x))
            return nothing
        end
        arr = AbacusArray{Float32}(undef, 4)
        @test_throws GPUCompiler.InvalidIRError @abacus bad_f64_runtime(Adapt.adapt(Adaptor(), arr))
    end

    @testset "dynamic dispatch rejected" begin
        # Type-unstable global forces dynamic dispatch in compiled code
        global _unstable_ref = Ref{Any}(1.0f0)
        function bad_dispatch_global(A::AbacusDeviceArray{Float32,1})
            A[1] = _unstable_ref[]  # type is Any → dynamic dispatch
            return nothing
        end
        arr = AbacusArray{Float32}(undef, 4)
        @test_throws GPUCompiler.InvalidIRError @abacus bad_dispatch_global(Adapt.adapt(Adaptor(), arr))
    end

    @testset "KernelAbstractions backend" begin
        using KernelAbstractions
        @kernel function ka_fill(A, val)
            i = @index(Global, Linear)
            A[i] = val
        end
        arr = AbacusArray{Float32}(undef, 8)
        ka_fill(AbacusBackend())(arr, 7.0f0; ndrange=8)
        @test Array(arr) == fill(7.0f0, 8)
    end
end
