## gpucompiler interface implementation

struct AbacusCompilerParams <: AbstractCompilerParams end
const AbacusCompilerConfig = CompilerConfig{NativeCompilerTarget, AbacusCompilerParams}
const AbacusCompilerJob = CompilerJob{NativeCompilerTarget, AbacusCompilerParams}

GPUCompiler.runtime_module(::AbacusCompilerJob) = Abacus

GPUCompiler.method_table(::AbacusCompilerJob) = method_table

GPUCompiler.kernel_state_type(job::AbacusCompilerJob) = KernelState

# Allow gpu_malloc/gpu_free, libc malloc/free, and Julia runtime functions
# that appear in error paths but won't actually be reached at runtime.
# We execute natively on the host CPU so these are all safe.
const _allowed_intrinsics = Set([
    "gpu_malloc", "gpu_free", "malloc", "free",
    # Julia runtime functions from error/print paths
    "ijl_module_parent", "ijl_module_name",
    "jl_module_parent", "jl_module_name",
    "ijl_object_id__cold", "jl_object_id__cold",
    "ijl_string_to_string", "jl_string_to_string",
    "ijl_symbol_name", "jl_symbol_name",
    "memcpy", "memset", "memmove",
    # Thread ID for per-thread kernel state access
    "jl_threadid", "ijl_threadid",
])
GPUCompiler.isintrinsic(::AbacusCompilerJob, fn::String) = fn in _allowed_intrinsics

# GPU-strict IR validation — copied from Metal.jl's validate_ir.
# The default check_ir → check_ir! pipeline already catches dynamic dispatch,
# runtime calls, GC allocations, and error paths. We add Float64 and i128 rejection.
function GPUCompiler.validate_ir(job::AbacusCompilerJob, mod::LLVM.Module)
    errors = GPUCompiler.IRError[]

    # reject Float64 values (same as Metal)
    function is_illegal_double(val)
        T_bad = LLVM.DoubleType()
        if LLVM.value_type(val) != T_bad
            return false
        end
        return true
    end
    append!(errors, GPUCompiler.check_ir_values(mod, is_illegal_double, "use of double value"))

    # Note: i128 is NOT checked in IR — Metal doesn't do this either.
    # i128 rejection happens at array construction (check_eltype).
    # Some valid patterns (e.g. _mul_high via llvmcall) use i128 as an intermediate.

    return errors
end


## compiler implementation (cache, configure, compile, and link)

# single global compiler cache
const _compiler_cache = Dict{Any, Any}()
function compiler_cache()
    return _compiler_cache
end

# cache of compiler configurations
const _compiler_configs = Dict{UInt, AbacusCompilerConfig}()
function compiler_config(; kwargs...)
    h = hash(kwargs)
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@noinline function _compiler_config(; kernel=false, name=nothing, always_inline=true, kwargs...)
    # NativeCompilerTarget with jlruntime=false enforces GPU constraints:
    # no GC, no dynamic dispatch, no runtime calls
    target = NativeCompilerTarget(; jlruntime=false)
    params = AbacusCompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# temp directory for compiled shared libraries (persists for session lifetime)
const _sodir = Ref{String}("")
function sodir()
    if isempty(_sodir[])
        _sodir[] = mktempdir(; cleanup=true)
    end
    return _sodir[]
end

# compile to LLVM IR → validate → emit PIC object code
function compile(@nospecialize(job::CompilerJob))
    JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(:llvm, job)
        entry = LLVM.name(meta.entry)

        # strip debug info
        if job.config.strip
            LLVM.strip_debuginfo!(mod)
        end
        GPUCompiler.prepare_execution!(job, mod)

        # emit object code with PIC relocation (required for shared libraries)
        triple = LLVM.triple(mod)
        t = LLVM.Target(triple=triple)
        tm = LLVM.TargetMachine(t, triple;
            reloc=LLVM.API.LLVMRelocPIC)
        obj = String(LLVM.emit(tm, mod, LLVM.API.LLVMObjectFile))

        (; obj, entry)
    end
end

# link: write .o → cc -shared → .so → dlopen → dlsym → Ptr{Cvoid}
function link(@nospecialize(job::CompilerJob), compiled)
    dir = sodir()
    id = string(hash(compiled.entry); base=16)
    obj_path = joinpath(dir, "$(id).o")
    lib_path = joinpath(dir, "$(id).$(Libdl.dlext)")

    open(obj_path, "w") do io
        write(io, compiled.obj)
    end
    run(`$(LLD_jll.lld_path) -flavor gnu -shared -o $lib_path $obj_path`)

    handle = Libdl.dlopen(lib_path)
    fptr = Libdl.dlsym(handle, compiled.entry)
    @assert fptr != C_NULL "Failed to find symbol $(compiled.entry) in $lib_path"
    return fptr
end
