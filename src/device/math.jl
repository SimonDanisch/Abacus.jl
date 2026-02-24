# Math function overrides — Float32-only implementations for GPU validation
# Mirrors Metal.jl's device/intrinsics/math.jl
# Uses LLVM intrinsics (native x86) instead of Metal AIR intrinsics.

using Base: FastMath

### Floating Point Math via LLVM intrinsics
# Julia's stdlib promotes Float32 → Float64 for accuracy. We override to stay in Float32.

for (jlfun, llvmfun) in [
    (:(Base.sqrt),  "llvm.sqrt"),
    (:(Base.sin),   "llvm.sin"),
    (:(Base.cos),   "llvm.cos"),
    (:(Base.exp),   "llvm.exp"),
    (:(Base.exp2),  "llvm.exp2"),
    (:(Base.log),   "llvm.log"),
    (:(Base.log2),  "llvm.log2"),
    (:(Base.log10), "llvm.log10"),
    (:(Base.abs),   "llvm.fabs"),
    (:(Base.floor), "llvm.floor"),
    (:(Base.ceil),  "llvm.ceil"),
    (:(Base.trunc), "llvm.trunc"),
    (:(Base.round), "llvm.round"),
]
    # LLVM IR type names: Float32 → "float", Float16 → "half"
    # Intrinsic name suffix: Float32 → "f32", Float16 → "f16"
    for (T, llvm_type, suffix) in [(Float32, "float", "f32"), (Float16, "half", "f16")]
        llvm_name = "$(llvmfun).$(suffix)"
        @eval @shared_device_override @inline $jlfun(x::$T) = Base.llvmcall(
            ($("declare $llvm_type @$llvm_name($llvm_type)\ndefine $llvm_type @entry($llvm_type %0) {\n  %r = call $llvm_type @$llvm_name($llvm_type %0)\n  ret $llvm_type %r\n}"), "entry"),
            $T, Tuple{$T}, x)
    end
end

# pow(Float32, Float32) via llvm.pow
for (T, llvm_type, suffix) in [(Float32, "float", "f32"), (Float16, "half", "f16")]
    @eval @shared_device_override @inline Base.:(^)(x::$T, y::$T) = Base.llvmcall(
        ($("declare $llvm_type @llvm.pow.$suffix($llvm_type, $llvm_type)\ndefine $llvm_type @entry($llvm_type %0, $llvm_type %1) {\n  %r = call $llvm_type @llvm.pow.$suffix($llvm_type %0, $llvm_type %1)\n  ret $llvm_type %r\n}"), "entry"),
        $T, Tuple{$T, $T}, x, y)
end

# fma via llvm.fma
for (T, llvm_type, suffix) in [(Float32, "float", "f32"), (Float16, "half", "f16")]
    @eval @shared_device_override @inline Base.fma(a::$T, b::$T, c::$T) = Base.llvmcall(
        ($("declare $llvm_type @llvm.fma.$suffix($llvm_type, $llvm_type, $llvm_type)\ndefine $llvm_type @entry($llvm_type %0, $llvm_type %1, $llvm_type %2) {\n  %r = call $llvm_type @llvm.fma.$suffix($llvm_type %0, $llvm_type %1, $llvm_type %2)\n  ret $llvm_type %r\n}"), "entry"),
        $T, Tuple{$T, $T, $T}, a, b, c)
end

# copysign via llvm.copysign — CPU backend only.
# llc crashes on G_FCOPYSIGN for Vulkan SPIR-V, so Vulkan gets a pure-Julia version below.
for (T, llvm_type, suffix) in [(Float32, "float", "f32"), (Float16, "half", "f16")]
    @eval @device_override @inline Base.copysign(x::$T, y::$T) = Base.llvmcall(
        ($("declare $llvm_type @llvm.copysign.$suffix($llvm_type, $llvm_type)\ndefine $llvm_type @entry($llvm_type %0, $llvm_type %1) {\n  %r = call $llvm_type @llvm.copysign.$suffix($llvm_type %0, $llvm_type %1)\n  ret $llvm_type %r\n}"), "entry"),
        $T, Tuple{$T, $T}, x, y)
end

# copysign — Vulkan backend: pure-Julia (no llvm.copysign, llc can't select G_FCOPYSIGN)
@vk_device_override @inline function Base.copysign(x::Float32, y::Float32)
    ax = abs(x)
    return y < 0f0 ? -ax : ax
end
@vk_device_override @inline function Base.copysign(x::Float16, y::Float16)
    ax = abs(x)
    return y < Float16(0) ? -ax : ax
end

### Pure-Julia Float32 implementations (copied from Metal.jl)

# Avoid Float64 in pow(Float32, Integer)
@shared_device_override @inline function Base.:(^)(x::Float32, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x * x
    y == 3 && return x * x * x
    x^Float32(y)
end
@shared_device_override @inline function Base.:(^)(x::Float16, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x * x
    y == 3 && return x * x * x
    x^Float16(y)
end

# hypot without Float64 (from Metal.jl, originally from Cosmopolitan Libc)
@inline function _hypot(a::T, b::T) where T <: AbstractFloat
    if isinf(a) || isinf(b)
        return T(Inf)
    end
    a = abs(a)
    b = abs(b)
    if a < b
        b, a = a, b
    end
    if iszero(a)
        return b
    end
    r = b / a
    return a * sqrt(one(T) + r * r)
end
@shared_device_override Base.hypot(x::Float32, y::Float32) = _hypot(x, y)
@shared_device_override Base.hypot(x::Float16, y::Float16) = _hypot(x, y)

# log1p(Float32) — from Metal.jl (openlibm's log1pf)
const _ln2_hi = 0.6931381f0
const _ln2_lo = 9.058001f-6
const _Lp1 = 0.6666667f0
const _Lp2 = 0.4f0
const _Lp3 = 0.2857143f0
const _Lp4 = 0.22222199f0
const _Lp5 = 0.18183573f0
const _Lp6 = 0.15313838f0
const _Lp7 = 0.14798199f0

@shared_device_override function Base.Math.log1p(x::Float32)
    hx = reinterpret(Int32, x)
    ax = hx & 0x7fffffff

    k = 1
    if hx < 0x3ed413d0
        if ax >= 0x3f800000
            if x == -1
                return -Inf32
            elseif isnan(x)
                return NaN32
            else
                Base.Math.throw_complex_domainerror(:log1p, x)
            end
        end
        if ax < 0x38000000
            if ax < 0x33800000
                return x
            else
                return x - x * x * 0.5f0
            end
        end
        if hx > 0 || hx <= reinterpret(Int32, 0xbe95f619)
            k = 0
            f = x
            hu = 1f0
        end
    end

    if hx >= 0x7f800000
        return x + x
    end

    if k ≠ 0
        if hx < 0x5a000000
            u = 1f0 + x
            hu = reinterpret(Int32, u)
            k = (hu >> 23) - 127
            c = k > 0 ? 1f0 - (u - x) : x - (u - 1f0)
            c /= u
        else
            u = x
            hu = reinterpret(Int32, u)
            k = (hu >> 23) - 127
            c = 0f0
        end
        hu &= 0x007fffff
        if hu < 0x3504f4
            u = reinterpret(Float32, hu | 0x3f800000)
        else
            k += 1
            u = reinterpret(Float32, hu | 0x3f000000)
            hu = (0x00800000 - hu) >> 2
        end
        f = u - 1f0
    end

    hfsq = 0.5f0 * f * f
    if hu == 0
        if f == 0
            if k == 0
                return 0f0
            else
                c += k * _ln2_lo
                return k * _ln2_hi + c
            end
        end
        R = hfsq * (1f0 - _Lp1 * f)
        if k == 0
            return f - R
        else
            return k * _ln2_hi - ((R - (k * _ln2_lo + c)) - f)
        end
    end

    s = f / (2f0 + f)
    z = s * s
    R = z * (_Lp1 + z * (_Lp2 + z * (_Lp3 + z * (_Lp4 + z * (_Lp5 + z * (_Lp6 + z * _Lp7))))))
    if k == 0
        return f - (hfsq - s * (hfsq + R))
    else
        return k * _ln2_hi - ((hfsq - (s * (hfsq + R) + (k * _ln2_lo + c))) - f)
    end
end

# expm1(Float32) — from Metal.jl (Norbert Juffa)
@shared_device_override function Base.expm1(a::Float32)
    j = fma(1.442695f0, a, 12582912.0f0)
    j = j - 12582912.0f0
    i = unsafe_trunc(Int32, j)
    f = fma(j, -6.93145752f-1, a)
    f = fma(j, -1.42860677f-6, f)

    s = f * f
    if a == 0.0f0
        s = a
    end

    r = fma(1.98423862f-4, f, 1.39347673f-3)
    t = fma(8.33342969f-3, f, 4.16667424f-2)
    r = fma(r, s, t)
    r = fma(r, f, 1.66666701f-1)
    r = fma(r, f, 4.99999970f-1)

    u = (j == 1) ? (f + 0.5f0) : f
    v = fma(r, s, u)
    s = 0.5f0
    t = ldexp(s, i)
    y = t - s
    x = (t - y) - s
    r = fma(v, t, x) + y
    r = r + r

    if j == 0
        r = v
    end
    if j == 1
        r = v + v
    end
    if abs(a - 1.0f0) > 88.0f0
        r = 2^a
        r = fma(r, r, -1.0f0)
    end

    return r
end

### Integer: _mul_high override to avoid i128
# Julia's _mul_high uses i128 for the multiplication. Metal provides hardware mul_hi.
# On x86, we implement _mul_high without i128 using inline assembly or widening multiply.
# SPIR-V/Vulkan doesn't support i128 at all (llc crashes), so the Vulkan backend gets
# a pure-Julia implementation using 32-bit multiply-add decomposition.

@static if isdefined(Base, :mul_hi) # Julia >= 1.13
    # Use Base.mul_hi which should be available
else
    # CPU backend: i128 widening multiply via llvmcall (works on x86)
    @device_override function Base.MultiplicativeInverses._mul_high(a::Int64, b::Int64)
        Base.llvmcall(
            ("""define i64 @entry(i64 %a, i64 %b) {
                %a_wide = sext i64 %a to i128
                %b_wide = sext i64 %b to i128
                %prod = mul i128 %a_wide, %b_wide
                %hi = lshr i128 %prod, 64
                %result = trunc i128 %hi to i64
                ret i64 %result
            }""", "entry"),
            Int64, Tuple{Int64, Int64}, a, b)
    end
    @device_override function Base.MultiplicativeInverses._mul_high(a::UInt64, b::UInt64)
        Base.llvmcall(
            ("""define i64 @entry(i64 %a, i64 %b) {
                %a_wide = zext i64 %a to i128
                %b_wide = zext i64 %b to i128
                %prod = mul i128 %a_wide, %b_wide
                %hi = lshr i128 %prod, 64
                %result = trunc i128 %hi to i64
                ret i64 %result
            }""", "entry"),
            UInt64, Tuple{UInt64, UInt64}, a, b)
    end

    # Vulkan backend: pure-Julia _mul_high (no i128 — SPIR-V doesn't support it).
    # Decomposes 64-bit multiply into four 32-bit multiplies.
    @vk_device_override function Base.MultiplicativeInverses._mul_high(a::UInt64, b::UInt64)
        a_lo = a % UInt32; a_hi = (a >> 32) % UInt32
        b_lo = b % UInt32; b_hi = (b >> 32) % UInt32
        lo_lo = UInt64(a_lo) * UInt64(b_lo)
        lo_hi = UInt64(a_lo) * UInt64(b_hi)
        hi_lo = UInt64(a_hi) * UInt64(b_lo)
        hi_hi = UInt64(a_hi) * UInt64(b_hi)
        mid = (lo_lo >> 32) + (lo_hi % UInt32(0xffffffff)) + (hi_lo % UInt32(0xffffffff))
        return hi_hi + (lo_hi >> 32) + (hi_lo >> 32) + (mid >> 32)
    end
    @vk_device_override function Base.MultiplicativeInverses._mul_high(a::Int64, b::Int64)
        # Signed: compute unsigned high word, then adjust for signs
        hi = Base.MultiplicativeInverses._mul_high(reinterpret(UInt64, a), reinterpret(UInt64, b))
        # Correction: if a < 0, subtract b from high; if b < 0, subtract a from high
        result = reinterpret(Int64, hi)
        if a < 0; result -= b; end
        if b < 0; result -= a; end
        return result
    end
end
