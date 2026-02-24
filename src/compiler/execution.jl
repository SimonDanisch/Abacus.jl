export @abacus


## high-level @abacus interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline]
const LAUNCH_KWARGS = [:groups, :threads]

macro abacus(ex...)
    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @abacus should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword arguments
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, COMPILER_KWARGS, LAUNCH_KWARGS)
    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key,val = kwarg.args
        if key === :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @abacus should be a Bool"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@abacus with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    @gensym f_var kernel_f kernel_args kernel_tt kernel

    push!(code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $abacusconvert($f_var)
                $kernel_args = map($abacusconvert, ($(var_exprs...),))
                $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                $kernel = $abacusfunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
         end)

    return esc(quote
        let
            $code
        end
    end)
end


## argument conversion

abacusconvert(arg) = adapt(Adaptor(), arg)

# Base.RefValue isn't GPU compatible
struct AbacusRefValue{T} <: Ref{T}
    x::T
end
Base.getindex(r::AbacusRefValue) = r.x
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = AbacusRefValue(adapt(to, r[]))

struct AbacusRefType{T} <: Ref{DataType} end
Base.getindex(::AbacusRefType{T}) where {T} = T
Adapt.adapt_structure(::Adaptor, r::Base.RefValue{<:Union{DataType, Type}}) =
    AbacusRefType{r[]}()

Adapt.adapt_structure(to::Adaptor,
                      bc::Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)


## ABI call — matches Julia's specfunc calling convention
# Ghost types are skipped, aggregates passed by pointer (Ref), scalars directly.
# Follows emit_call_specfun_other from Julia's codegen.

@generated function abi_call(f::Ptr{Cvoid}, rt::Type{RT}, tt::Type{T},
                              args::Vararg{Any, N}) where {T, RT, N}
    argtt    = tt.parameters[1]
    rettype  = rt.parameters[1]
    argtypes = DataType[argtt.parameters...]

    argexprs = Union{Expr, Symbol}[]
    ccall_types = DataType[]

    before = :()
    after = :(ret)

    JuliaContext() do ctx
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

        for (source_i, source_typ) in enumerate(argtypes)
            if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
                continue
            end

            argexpr = :(args[$source_i])

            isboxed = GPUCompiler.deserves_argbox(source_typ)
            et = isboxed ? T_prjlvalue : convert(LLVMType, source_typ)

            if isboxed
                push!(ccall_types, Any)
            elseif isa(et, LLVM.StructType) || isa(et, LLVM.ArrayType)
                push!(ccall_types, Ptr{source_typ})
                argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
            else
                push!(ccall_types, source_typ)
            end
            push!(argexprs, argexpr)
        end

        if GPUCompiler.isghosttype(rettype) || Core.Compiler.isconstType(rettype)
            # ghost return — ccall returns Nothing
        elseif !GPUCompiler.deserves_retbox(rettype)
            rt_llvm = convert(LLVMType, rettype)
            if !isa(rt_llvm, LLVM.VoidType) && GPUCompiler.deserves_sret(rettype, rt_llvm)
                before = :(sret = Ref{$rettype}())
                pushfirst!(argexprs, :(sret))
                pushfirst!(ccall_types, Ptr{rettype})
                rettype = Nothing
                after = :(sret[])
            end
        end
    end

    quote
        $before
        ret = ccall(f, $rettype, ($(ccall_types...),), $(argexprs...))
        $after
    end
end


## host-side kernel API

struct HostKernel{F,TT}
    f::F
    fptr::Ptr{Cvoid}
end

const abacusfunction_lock = ReentrantLock()

function abacusfunction(f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    Base.@lock abacusfunction_lock begin
        cache = compiler_cache()
        source = methodinstance(F, tt)
        config = compiler_config(; name, kwargs...)::AbacusCompilerConfig
        fptr = GPUCompiler.cached_compilation(cache, source, config, _compile_cached, link)

        h = hash(fptr, hash(f, hash(tt)))
        kernel = get(_kernel_instances, h, nothing)
        if kernel === nothing
            kernel = HostKernel{F,tt}(f, fptr)
            _kernel_instances[h] = kernel
        end
        return kernel::HostKernel{F,tt}
    end
end

const _kernel_instances = Dict{UInt, Any}()


## kernel execution — calls compiled native code via ccall
# Workgroups are launched in parallel across Julia threads.
# Per-thread kernel state is stored in a fixed raw buffer indexed by thread ID.

function (kernel::HostKernel{F,TT})(args...; groups=1, threads=1) where {F,TT}
    groups = groups isa Integer ? (groups, 1, 1) :
             length(groups) == 1 ? (groups[1], 1, 1) :
             length(groups) == 2 ? (groups[1], groups[2], 1) :
             (groups[1], groups[2], groups[3])

    threads = threads isa Integer ? (threads, 1, 1) :
              length(threads) == 1 ? (threads[1], 1, 1) :
              length(threads) == 2 ? (threads[1], threads[2], 1) :
              (threads[1], threads[2], threads[3])

    converted_args = map(abacusconvert, args)

    # KernelState is the first argument of every compiled kernel (added by add_kernel_state!
    # and converted to a pointer by kernel_state_to_reference!). abi_call maps it to
    # Ptr{KernelState} + Ref(ks), matching the ptr parameter in the compiled function.
    full_tt = Tuple{KernelState, F, TT.parameters...}

    total_groups = groups[1] * groups[2] * groups[3]

    Threads.@threads :static for group_linear in 1:total_groups
        gx = ((group_linear - 1) % groups[1]) + 1
        gy = (((group_linear - 1) ÷ groups[1]) % groups[2]) + 1
        gz = (((group_linear - 1) ÷ (groups[1] * groups[2])) % groups[3]) + 1
        for tz in 1:threads[3], ty in 1:threads[2], tx in 1:threads[1]
            ks = KernelState(Int32(gx), Int32(gy), Int32(gz),
                             Int32(tx), Int32(ty), Int32(tz))
            _set_kernel_state!(ks)  # also update buffer for non-compiled path
            abi_call(kernel.fptr, Nothing, full_tt, ks, kernel.f, converted_args...)
        end
    end
    return nothing
end
