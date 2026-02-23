## gpucompiler interface implementation

struct AbacusCompilerParams <: AbstractCompilerParams end
const AbacusCompilerConfig = CompilerConfig{NativeCompilerTarget, AbacusCompilerParams}
const AbacusCompilerJob = CompilerJob{NativeCompilerTarget, AbacusCompilerParams}

GPUCompiler.runtime_module(::AbacusCompilerJob) = Abacus

GPUCompiler.method_table(::AbacusCompilerJob) = method_table

GPUCompiler.kernel_state_type(job::AbacusCompilerJob) = KernelState

# Don't add byval attributes to bittype args — keep Julia's native calling convention.
# add_kernel_state! still adds KernelState as first arg by value; kernel_state_to_reference!
# then converts it to a pointer so ccall can pass Ptr{KernelState} directly.
GPUCompiler.pass_by_value(::AbacusCompilerJob) = false

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
])
GPUCompiler.isintrinsic(::AbacusCompilerJob, fn::String) = fn in _allowed_intrinsics

# Strict Metal-compatible IR validation.
#
# This is an exact copy of Metal's validate_ir (GPUCompiler/src/metal.jl) so that Abacus
# rejects the exact same IR constructs that Metal would reject. The design goal of Abacus
# is to serve as a drop-in CPU backend that catches Metal-incompatible code on non-macOS
# machines, enabling local testing of GPU kernels before running them on Apple hardware.
#
# Checked: Float64 (double), except values used exclusively in metal_os_log calls.
#          Int128 (i128), unconditionally.
function GPUCompiler.validate_ir(job::AbacusCompilerJob, mod::LLVM.Module)
    errors = GPUCompiler.IRError[]

    # Metal does not support double precision, except for logging
    function is_illegal_double(val)
        T_bad = LLVM.DoubleType()
        if LLVM.value_type(val) != T_bad
            return false
        end
        function used_for_logging(use::LLVM.Use)
            usr = LLVM.user(use)
            if usr isa LLVM.CallInst
                callee = LLVM.called_operand(usr)
                if callee isa LLVM.Function && startswith(LLVM.name(callee), "metal_os_log")
                    return true
                end
            end
            return false
        end
        if all(used_for_logging, LLVM.uses(val))
            return false
        end
        return true
    end
    append!(errors, GPUCompiler.check_ir_values(mod, is_illegal_double, "use of double value"))

    # Metal never supports 128-bit integers
    append!(errors, GPUCompiler.check_ir_values(mod, LLVM.IntType(128)))

    # Run LLVM's built-in verifier on the post-optimization IR.
    # This catches broken IR that optimization passes can produce — e.g. a `select` on a
    # struct type that an LLVM scalarization pass incorrectly transforms into a
    # type-mismatched scalar `select` (the verifier diagnoses "Select values must have
    # same type as select instruction"). Without this, such broken IR silently passes
    # Abacus's LLJIT (which is more permissive) but crashes Metal's `air` or SPIR-V's
    # `llc` with an opaque "Broken function found, compilation aborted" message.
    try
        LLVM.verify(mod)
    catch e
        push!(errors, ("broken LLVM IR after optimization ($(e.info))", [], nothing))
    end

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
@noinline function _compiler_config(; kernel=true, name=nothing, always_inline=true, kwargs...)
    # NativeCompilerTarget with jlruntime=false enforces GPU constraints:
    # no GC, no dynamic dispatch, no runtime calls.
    # kernel=true enables the kernel state mechanism (KernelState as first arg).
    target = NativeCompilerTarget(; jlruntime=false)
    params = AbacusCompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# Global LLJIT instance (session lifetime) for in-process kernel JIT execution.
const _lljit = Ref{Any}(nothing)
const _lljit_lock = ReentrantLock()

function _get_or_create_lljit()
    Base.@lock _lljit_lock begin
        if _lljit[] === nothing
            jit = LLVM.LLJIT(; tm=LLVM.JITTargetMachine())
            jd = LLVM.JITDylib(jit)
            # Resolve symbols from the host process (malloc, free, etc.)
            prefix = Sys.isapple() ? '_' : '\0'
            gen = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
            LLVM.add!(jd, gen)

            # On Windows/MinGW, GPUCompiler strips probe-stack="inline-asm" so LLJIT's
            # codegen emits calls to ___chkstk_ms (from static libgcc.a) for large frames.
            # That symbol isn't in any DLL, but ntdll __chkstk has identical semantics:
            # both take the frame size in %rax, probe stack pages, and leave %rsp unchanged.
            # Register ntdll's __chkstk under the name ___chkstk_ms so LLJIT can link it.
            if Sys.iswindows()
                ntdll = Libdl.dlopen("ntdll.dll"; throw_error=false)
                if ntdll !== nothing
                    ptr = Libdl.dlsym(ntdll, "__chkstk"; throw_error=false)
                    if ptr !== nothing
                        sym = LLVM.mangle(jit, "___chkstk_ms")
                        pair = LLVM.API.LLVMOrcCSymbolMapPair(
                            sym.ref,
                            LLVM.API.LLVMJITEvaluatedSymbol(
                                reinterpret(UInt64, ptr),
                                LLVM.API.LLVMJITSymbolFlags(
                                    UInt8(LLVM.API.LLVMJITSymbolGenericFlagsExported |
                                          LLVM.API.LLVMJITSymbolGenericFlagsCallable),
                                    UInt8(0)
                                )
                            )
                        )
                        mu = LLVM.absolute_symbols([pair])
                        LLVM.define(jd, mu)
                    end
                end
            end

            _lljit[] = jit
        end
        return _lljit[]::LLVM.LLJIT
    end
end

# compile: LLVM IR → optimize → convert state arg to pointer → serialize to bitcode.
# kernel_state_to_reference! wraps the entry function so the KernelState is received
# as a ptr (loaded on entry) instead of by value, matching ccall's Ptr{KernelState}.
function compile(@nospecialize(job::CompilerJob))
    JuliaContext() do ctx
        mod, meta = GPUCompiler.compile(:llvm, job)
        entry_fn = LLVM.name(meta.entry)

        if job.config.strip
            LLVM.strip_debuginfo!(mod)
        end
        GPUCompiler.prepare_execution!(job, mod)

        # Convert kernel state arg from pass-by-value to pass-by-reference.
        # add_kernel_state! adds KernelState as { i32×6 } by value; on x64 this would
        # require special ABI handling. Converting to a pointer lets ccall pass
        # Ptr{KernelState} transparently on all platforms.
        if job.config.kernel && GPUCompiler.kernel_state_type(job) !== Nothing
            if haskey(LLVM.functions(mod), entry_fn)
                entry_f = LLVM.functions(mod)[entry_fn]
                GPUCompiler.kernel_state_to_reference!(job, mod, entry_f)
                # kernel_state_to_reference! preserves the function name
            end
        end

        # Final sanity check: verify IR correctness after all Abacus-specific
        # transformations (prepare_execution! + kernel_state_to_reference!).
        LLVM.verify(mod)

        bitcode = convert(Vector{UInt8}, mod)
        (; bitcode, entry=entry_fn)
    end
end

# link: parse bitcode → align triple/datalayout to LLJIT → add to LLJIT → return fptr
function link(@nospecialize(job::CompilerJob), compiled)
    jit = _get_or_create_lljit()
    jd = LLVM.JITDylib(jit)

    # Check if the symbol is already loaded (e.g. same kernel compiled again after a
    # REPL method redefinition creates a new MethodInstance cache miss). Compilation is
    # deterministic for the same (specTypes, config), so reuse the existing definition.
    try
        addr = LLVM.lookup(jit, compiled.entry)
        fptr = LLVM.pointer(addr)
        fptr != C_NULL && return fptr
    catch
    end

    JuliaContext() do ctx
        mod = parse(LLVM.Module, compiled.bitcode)
        # Align module triple and data layout with the LLJIT target.
        LLVM.triple!(mod, LLVM.triple(jit))
        LLVM.datalayout!(mod, LLVM.datalayout(jit))
        LLVM.add!(jit, jd, LLVM.ThreadSafeModule(mod))
    end

    addr = LLVM.lookup(jit, compiled.entry)
    fptr = LLVM.pointer(addr)
    @assert fptr != C_NULL "Failed to find symbol $(compiled.entry)"
    return fptr
end


# _compile_cached is the compiler function passed to GPUCompiler.cached_compilation.
# GPUCompiler's built-in disk cache does not work for runtime (non-precompiled) kernels
# because Base.object_build_id returns nothing for them. We simply delegate to compile,
# and rely on GPUCompiler's in-memory cache (keyed by CodeInstance) for the session.
_compile_cached(@nospecialize(job::CompilerJob)) = compile(job)
