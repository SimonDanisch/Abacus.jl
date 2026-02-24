# device quirks — @shared_device_override for error-throwing functions
# Copied from Metal.jl's device/quirks.jl
# These replace error paths (which cause dynamic dispatch/runtime calls)
# with simple throw(nothing) so they compile cleanly on GPU.

macro print_and_throw(args...)
    quote
        throw(nothing)
    end
end

# math.jl
@shared_device_override @noinline Base.Math.throw_complex_domainerror(f::Symbol, x) =
    @print_and_throw "This operation requires a complex input to return a complex result"
@shared_device_override @noinline Base.Math.throw_exp_domainerror(x) =
    @print_and_throw "Exponentiation yielding a complex result requires a complex argument"
@shared_device_override function Base.Math.exponent(x::T) where T<:Base.IEEEFloat
    xs = reinterpret(Unsigned, x) & ~Base.sign_mask(T)
    xs >= Base.exponent_mask(T) && @print_and_throw "Cannot be NaN or Inf."
    k = Int(xs >> Base.significand_bits(T))
    if k == 0 # x is subnormal
        xs == 0 && @print_and_throw "Cannot be ±0.0."
        m = leading_zeros(xs) - Base.exponent_bits(T)
        k = 1 - m
    end
    return k - Base.exponent_bias(T)
end

# intfuncs.jl
@shared_device_override @noinline Base.throw_domerr_powbysq(::Any, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@shared_device_override @noinline Base.throw_domerr_powbysq(::Integer, p) =
    @print_and_throw "Cannot raise an integer to a negative power"
@shared_device_override @noinline Base.throw_domerr_powbysq(::AbstractMatrix, p) =
    @print_and_throw "Cannot raise an integer to a negative power"

# checked.jl
@shared_device_override @noinline Base.Checked.throw_overflowerr_binaryop(op, x, y) =
    @print_and_throw "Binary operation overflowed"
@shared_device_override @noinline Base.Checked.throw_overflowerr_negation(op, x, y) =
    @print_and_throw "Negation overflowed"

# boot.jl
@shared_device_override @noinline Core.throw_inexacterror(f::Symbol, ::Type{T}, val) where {T} =
    @print_and_throw "Inexact conversion"

# Integer conversion — GPU hardware does unchecked truncation.
# On our native x86 backend, Julia's checked_trunc_uint/sint actually run
# and can hit `unreachable` (SIGILL). Override with unchecked truncation
# to match GPU behavior.
@shared_device_override @inline Core.checked_trunc_uint(::Type{To}, x) where {To} =
    unsafe_trunc(To, x)
@shared_device_override @inline Core.checked_trunc_sint(::Type{To}, x) where {To} =
    unsafe_trunc(To, x)

# Signed integer division — Julia's checked_sdiv_int/checked_srem_int generate
# branches for div-by-zero and INT_MIN/-1 overflow checks. These branches cause
# the LLVM SPIR-V backend to emit invalid structured control flow (OpSwitch with
# broken merge blocks). Override with unchecked intrinsics for GPU.
@shared_device_override @inline Base.div(x::T, y::T) where {T<:Base.BitSigned} =
    Base.sdiv_int(x, y)
@shared_device_override @inline Base.rem(x::T, y::T) where {T<:Base.BitSigned} =
    Base.srem_int(x, y)

# abstractarray.jl
@shared_device_override @noinline Base.throw_boundserror(A, I) =
    @print_and_throw "Out-of-bounds array access"

# trig.jl
@shared_device_override @noinline Base.Math.sincos_domain_error(x) =
    @print_and_throw "sincos(x) is only defined for finite x."

# diagonal.jl
import LinearAlgebra
@shared_device_override function Base.setindex!(D::LinearAlgebra.Diagonal, v, i::Int, j::Int)
    @boundscheck checkbounds(D, i, j)
    if i == j
        @inbounds D.diag[i] = v
    elseif !iszero(v)
        @print_and_throw "cannot set off-diagonal entry to a nonzero value"
    end
    return v
end

# number.jl
@shared_device_override @inline function Base.getindex(x::Number, I::Integer...)
    @boundscheck all(isone, I) ||
        @print_and_throw "Out-of-bounds access of scalar value"
    x
end

# complex.jl
@shared_device_override function Base.ssqs(x::T, y::T) where T<:Real
    k::Int = 0
    ρ = x*x + y*y
    if !isfinite(ρ) && (isinf(x) || isinf(y))
        ρ = convert(T, Inf)
    elseif isinf(ρ) || (ρ==0 && (x!=0 || y!=0)) || ρ<nextfloat(zero(T))/(2*eps(T)^2)
        m::T = max(abs(x), abs(y))
        k = m==0 ? 0 : exponent(m)
        xk, yk = ldexp(x,-k), ldexp(y,-k)
        ρ = xk*xk + yk*yk
    end
    ρ, k
end

@static if VERSION >= v"1.12.0-DEV.1736"
    let BitInteger64 = Union{Int64,UInt64}
        @shared_device_override function Base.checkbounds(::Type{Bool}, v::StepRange{<:BitInteger64, <:BitInteger64}, i::BitInteger64)
            @inline
            return checkindex(Bool, eachindex(IndexLinear(), v), i)
        end
    end
end
