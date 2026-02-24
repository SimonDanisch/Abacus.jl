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
    # C runtime functions (struct copies, math lowered from LLVM intrinsics)
    "memcpy", "memset", "memmove",
    "powf", "sqrtf", "sinf", "cosf", "tanf",
    "expf", "exp2f", "logf", "log2f", "log10f",
    "fabsf", "floorf", "ceilf", "roundf", "truncf",
    "fmodf", "fmaf", "fminf", "fmaxf",
    "atan2f", "asinf", "acosf", "atanf",
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
    # validation but crashes Metal's `air` or SPIR-V's `llc` with an opaque
    # "Broken function found, compilation aborted" message.
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

# Persistent directory for compiled shared libraries.
# Includes Julia version to auto-invalidate cache when Julia is upgraded.
const _sodir = Ref{String}("")
function sodir()
    if isempty(_sodir[])
        dir = joinpath(first(Base.DEPOT_PATH), "compiled", "abacus_kernels",
                       "jl$(VERSION.major).$(VERSION.minor)")
        mkpath(dir)
        _sodir[] = dir
    end
    return _sodir[]
end

# compile: LLVM IR → optimize → convert state arg to pointer → emit PIC object code.
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
            end
        end

        # Final sanity check: verify IR correctness after all Abacus-specific
        # transformations (prepare_execution! + kernel_state_to_reference!).
        LLVM.verify(mod)

        if Sys.iswindows()
            # Julia's LLVM on MinGW produces ELF object files (the x86_64-w64-mingw32
            # target has ELF mangling m:e). Switch to the MSVC triple so LLVM's codegen
            # emits proper COFF, which lld-link can turn into a DLL.
            msvc_triple = "x86_64-pc-windows-msvc"
            t = LLVM.Target(triple=msvc_triple)
            tm = LLVM.TargetMachine(t, msvc_triple; reloc=LLVM.API.LLVMRelocPIC)
            LLVM.triple!(mod, msvc_triple)
            LLVM.datalayout!(mod, LLVM.DataLayout(tm))

            # Disable stack probing entirely. GPUCompiler strips probe-stack (GPU code
            # doesn't need it), and on MSVC targets LLVM would otherwise insert calls to
            # __chkstk which we'd need to import from ntdll.dll. Our kernels run on the
            # host CPU with adequate stack, so probing is unnecessary.
            for fn in LLVM.functions(mod)
                if !LLVM.isdeclaration(fn)
                    push!(LLVM.function_attributes(fn),
                          LLVM.StringAttribute("no-stack-arg-probe", ""))
                end
            end

            # MSVC CRT requires _fltused to be defined when floating-point code is present.
            if !haskey(LLVM.globals(mod), "_fltused")
                i32 = LLVM.Int32Type()
                gv = LLVM.GlobalVariable(mod, i32, "_fltused")
                LLVM.initializer!(gv, LLVM.ConstantInt(i32, 1))
                LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)
            end
        else
            # Linux / macOS: emit with the native triple.
            triple = LLVM.triple(mod)
            t = LLVM.Target(triple=triple)
            tm = LLVM.TargetMachine(t, triple; reloc=LLVM.API.LLVMRelocPIC)
        end

        obj = LLVM.emit(tm, mod, LLVM.API.LLVMObjectFile)
        (; obj, entry=entry_fn)
    end
end

# Windows CRT import library for lld-link.
# lld-link /noentry creates DLLs without the C runtime. When kernels reference
# C runtime functions (memmove from struct copies, powf from math), the linker
# needs an import library that maps these symbols to ucrtbase.dll.
const _ucrt_symbols = [
    "malloc", "free",
    "memmove", "memcpy", "memset",
    "powf", "sqrtf", "sinf", "cosf", "tanf",
    "expf", "exp2f", "logf", "log2f", "log10f",
    "fabsf", "floorf", "ceilf", "roundf", "truncf",
    "fmodf", "fmaf", "fminf", "fmaxf",
    "atan2f", "asinf", "acosf", "atanf",
    "sinhf", "coshf", "tanhf", "copysignf", "cbrtf",
    "pow", "sqrt", "sin", "cos", "tan",
    "exp", "exp2", "log", "log2", "log10",
    "fabs", "floor", "ceil", "round", "trunc",
    "fmod", "fma", "fmin", "fmax",
    "atan2", "asin", "acos", "atan",
]
const _ucrt_importlib = Ref{String}("")
function _ensure_ucrt_importlib()
    lib = _ucrt_importlib[]
    if !isempty(lib) && isfile(lib)
        return lib
    end
    dir = sodir()
    def_path = joinpath(dir, "ucrt.def")
    lib_path = joinpath(dir, "ucrt.lib")
    open(def_path, "w") do io
        println(io, "LIBRARY ucrtbase.dll")
        println(io, "EXPORTS")
        for sym in _ucrt_symbols
            println(io, "    ", sym)
        end
    end
    LLD_jll.lld() do exe
        run(`$exe -flavor link /lib /def:$def_path /machine:x64 /out:$lib_path`)
    end
    rm(def_path; force=true)
    _ucrt_importlib[] = lib_path
    return lib_path
end

# Link object code into a platform-specific shared library.
function _link_shlib(obj_path, lib_path, entry_name)
    if Sys.isapple()
        # Apple's cc (clang) handles Mach-O shared libraries natively.
        # -undefined dynamic_lookup allows unresolved symbols (resolved at dlopen time).
        run(`cc -shared -undefined dynamic_lookup -o $lib_path $obj_path`)
    elseif Sys.iswindows()
        # lld-link creates a PE DLL from the COFF object file.
        # /noentry: no DllMain needed. /export: makes the kernel symbol visible to dlsym.
        # Must use the LLD_jll.lld() wrapper so lld.exe finds its LLVM shared libraries.
        # Link against ucrtbase.dll import lib to resolve CRT symbols (memmove, powf, etc.).
        ucrt_lib = _ensure_ucrt_importlib()
        LLD_jll.lld() do exe
            run(`$exe -flavor link /dll /noentry /export:$entry_name /out:$lib_path $obj_path $ucrt_lib`)
        end
    else
        # Linux/other: standard ELF shared library via LLD's GNU-compatible driver.
        LLD_jll.lld() do exe
            run(`$exe -flavor gnu -shared -o $lib_path $obj_path`)
        end
    end
end

# Hash-based short filename to avoid Windows MAX_PATH (260 char) limit.
# Long C++ mangled names like _Z30gpu_broadcast_kernel_cartesian16CompilerMetadata...
# easily exceed this. The symbol name is still used for dlsym lookup.
_lib_basename(entry::String) = string(hash(entry); base=16)

# link: write .o → link shared library → dlopen → dlsym → Ptr{Cvoid}
function link(@nospecialize(job::CompilerJob), compiled)
    dir = sodir()
    base = _lib_basename(compiled.entry)
    lib_path = joinpath(dir, "$(base).$(Libdl.dlext)")

    # Fast path: reuse previously linked shared library from persistent cache.
    if isfile(lib_path)
        handle = Libdl.dlopen(lib_path; throw_error=false)
        if handle !== nothing
            fptr = Libdl.dlsym(handle, compiled.entry; throw_error=false)
            if fptr !== C_NULL && fptr !== nothing
                return fptr
            end
        end
    end

    # Write object code and link to shared library.
    obj_path = joinpath(dir, "$(base).o")
    open(obj_path, "w") do io
        write(io, compiled.obj)
    end
    _link_shlib(obj_path, lib_path, compiled.entry)
    rm(obj_path; force=true)  # cleanup intermediate object file

    handle = Libdl.dlopen(lib_path)
    fptr = Libdl.dlsym(handle, compiled.entry)
    @assert fptr != C_NULL "Failed to find symbol $(compiled.entry) in $lib_path"
    return fptr
end


# _compile_cached is the compiler function passed to GPUCompiler.cached_compilation.
# GPUCompiler's built-in disk cache does not work for runtime (non-precompiled) kernels
# because Base.object_build_id returns nothing for them. We simply delegate to compile,
# and rely on GPUCompiler's in-memory cache (keyed by CodeInstance) for the session.
_compile_cached(@nospecialize(job::CompilerJob)) = compile(job)
