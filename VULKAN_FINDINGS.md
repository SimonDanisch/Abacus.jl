# Vulkan Backend Implementation Findings

## Critical Environment Rule
- `env_path` must ALWAYS be `/sim/Programmieren/VulkanDev` for every `julia_eval` call.

---

## Completed Work

### Phase 1: Core Vulkan Compute (DONE)
- Vulkan init, memory (BDA), VkArray, compilation, pipeline, dispatch, KA backend all working.
- 1D and 2D KA kernels working (double, matrix_add, matmul).
- GPUArrays integration: broadcast, sum, mapreduce all run on GPU via KA → VulkanBackend.

### @localmem / SharedMemory Support (DONE)
- Added `@vk_device_override` for `KA.SharedMemory` in `VulkanKernels.jl:119-122`.
- Uses `Abacus.vk_localmemory(Val(Id), T, Val(len))` in `intrinsics.jl`.
- End-to-end tested: localmem + barrier + reverse read pattern works correctly.

### Performance vs AMDGPU (GPUArrays broadcast / KA kernels)
```
Operation                          Vulkan    AMDGPU    Ratio
Dispatch overhead (16 elems)        46us      36us     1.3x slower
c .= a .+ b (N=1M)                 64us      51us     1.25x slower
c .= @. a*b+2a (1024x1024)        145us      70us     2.08x slower
sum (N=1024)                       172us      76us     2.26x slower
sum (N=1M)                         178us     135us     1.32x slower
KA matmul 256x256                  155us     655us     0.24x FASTER
KA matmul 512x512                  357us    2564us     0.14x FASTER
```

---

## Key Bugs Found & Fixed

### 1. LLVM SPIR-V Backend ConstantExpr GEP Bug (compilation.jl:781-843)
**Problem**: The LLVM SPIR-V backend correctly translates ConstantExpr GEPs on push
constants in the entry basic block, but mishandles them in non-entry blocks (producing
byte-offset arithmetic with OpAccessChain on uchar type instead of struct member access).
**Fix**: `_hoist_push_constant_loads!()` — hoists all loads from push constant globals
(addrspace 2) into the entry block. Push constants are uniform (constant for entire
shader invocation), so this is semantically valid.
**Failed approaches**:
- `_lower_constexpr_geps!`: LLVM.gep! with constant indices constant-folds back to ConstantExpr.
- `freeze` on indices: prevents folding but crashes llc (`StructType::getTypeAtIndex`).

### 2. Flat GEPs on Shared Memory (compilation.jl: `_fix_shared_geps!`)
**Problem**: After inlining + InstCombine, two-level GEPs on addrspace(3) array globals
get flattened to single-index form. The LLVM SPIR-V backend silently drops the index,
producing OpAccessChain without indices.
**Fix**: `_fix_shared_geps!()` rewrites flat GEPs back to structured [array_type, gv, 0, idx] form.

### 3. KA.SharedMemory gensym ID breaks LLVM IR (intrinsics.jl)
**Problem**: KA's `@localmem` generates gensym IDs like `##static_shmem#284`. The `#`
characters are comment delimiters in LLVM IR, so when used in global variable names
via `llvmcall`, they break the IR parsing. The kernel silently compiles to just an
exception handler (all computation eliminated).
**Fix**: Sanitize ID with `replace(string(Id), r"[^a-zA-Z0-9_]" => "_")`.

### 4. Alignment Decorations on Workgroup Variables (spirv_fixup.jl)
**Problem**: The LLVM SPIR-V backend emits `OpDecorate ... Alignment 4` on Workgroup
(shared memory) variables. The `Alignment` decoration requires the `Kernel` capability,
which is not present in Vulkan SPIR-V (`Shader` capability only).
**Fix**: Strip with `replace(fixed, r"^\s*OpDecorate .* Alignment \d+\n"m => "")`.

### 5. Barrier Semantics (NOT fixed — acceptable)
**Problem**: `@llvm.spv.group.memory.barrier.with.group.sync` hardcodes
SequentiallyConsistent (0x10=16). spirv-val rejects this for Vulkan.
**Decision**: Left as-is. SequentiallyConsistent is strictly stronger than the required
AcquireRelease, so correctness is preserved. RADV and ANV accept it.
**Why no fix**: The only LLVM SPIR-V backend barrier intrinsic for Vulkan is the
non-parameterized `@llvm.spv.group.memory.barrier.with.group.sync` (no way to specify
semantics). The parameterized `@llvm.spv.control.barrier(i32,i32,i32)` is NOT lowered
by llc for the Vulkan target — it's left as an unresolved function declaration requiring
Linkage capability. Verified via MWE.

### 6. GPUArrays.derive View-of-View Offset Bug (array.jl)
**Problem**: `GPUArrays.derive` didn't accumulate the parent VkArray's element offset.
When creating a view-of-view (e.g. `@view (@view dst[3:4])[1:1]`), the inner view's
offset was computed from 0 instead of inheriting the outer view's offset.
**Symptom**: `AK.reduce` returned wrong results for N>512 (multi-block reduction).
The second reduction pass wrote to the wrong location.
**Fix**: `offset += (a.offset * sizeof(S)) ÷ sizeof(T)` in `GPUArrays.derive`.
Matches AMDGPU's `ROCArray` pattern.

### 7. OpOrdered Requires Kernel Capability (VulkanKernels.jl)
**Problem**: Julia's `isless(::Float32, ::Float32)` generates `fcmp ord` → `OpOrdered`
in SPIR-V, which requires the `Kernel` capability not available in Vulkan's `Shader` model.
**Symptom**: spirv-val rejects sort kernels with "OpOrdered requires Kernel capability".
**Fix**: `@vk_device_override` on `Base.isless(::Float32, ::Float32)` to use `x < y`.
Drops NaN handling, matching GPU convention.

### 8. spirv-val Validation Guard (pipeline.jl)
**Problem**: Invalid SPIR-V passed to `vkCreateComputePipelines` crashes the RADV driver
(segfault in `libvulkan_radeon.so`).
**Fix**: Added `_validate_spirv()` that runs `spirv-val --target-env vulkan1.3` before
pipeline creation. Filters known-acceptable errors via `_SPIRV_VAL_IGNORED_VUIDS` list
(currently: barrier SequentiallyConsistent VUID). Uses JLL do-block syntax.

---

## SPIR-V Text Fixups (spirv_fixup.jl) — Why Each Is Needed

All text fixups are blocked on upstream LLVM SPIR-V backend limitations:

| # | Fixup | Why text-based |
|---|-------|---------------|
| 1 | Remove Linkage capability + decorations | llc emits Linkage for any external declaration; no way to suppress |
| 2 | Add PhysicalStorageBufferAddresses cap | No LLVM address space maps to this yet |
| 3 | PhysicalStorageBuffer64 memory model | Same — no LLVM mapping |
| 4 | CrossWorkgroup → PhysicalStorageBuffer | addrspace(1) maps to CrossWorkgroup; no Vulkan BDA mapping |
| 5 | UniformConstant → PushConstant | addrspace(2) maps to UniformConstant; addrspace(13)→PushConstant needs LLVM 23+ |
| 6 | Remove Alignment decorations | llc emits Alignment on Workgroup vars; requires Kernel capability |
| 7 | Barrier semantics | Accepted as-is (SequentiallyConsistent); drivers handle it |
| 8 | Block/Offset decorations | llc doesn't emit these for push constant structs |

## LLVM IR Level Fixups (_prepare_module_for_vulkan!)

| Fix | Why at LLVM level |
|-----|-------------------|
| Strip lifetime intrinsics | Must happen before llc (llc crashes on them) |
| Internal linkage on non-entry fns | Prevents llc from emitting Linkage decorations |
| Fix load/store alignment | llc requires explicit alignment on all memory ops |
| Fix shared memory GEPs | InstCombine flattens structured GEPs; llc drops flat GEP indices |
| Hoist push constant loads | llc mishandles ConstantExpr GEPs in non-entry blocks |
| Replace unreachable | llc can't structure dead-end blocks into valid Vulkan SPIR-V |
| Strip noreturn/assume | These attributes cause llc issues |

---

## Current Status & Next Steps

### Working AK Operations
- `AK.foreachindex` — closures with captured VkArrays properly adapted via Adapt.jl
- `AK.map` / `AK.map!`
- `AK.reduce` — all sizes (after derive offset fix #6)
- `AK.sum` / `AK.prod` / `AK.maximum` / `AK.minimum` / `AK.count`
- `AK.mapreduce`
- `AK.sort!` / `AK.sortperm` / `AK.sortperm!`
- `AK.accumulate` / `AK.cumsum`
- `AK.searchsortedfirst`
- `AK.any` / `AK.all`

### AcceleratedKernels Benchmarks (Vulkan vs AMDGPU, with async batch dispatch)

GPU: AMD Radeon RX 7900 XTX — Vulkan (RADV NAVI31) vs AMDGPU (ROCm)

```
Operation                              Vulkan       AMDGPU       Ratio
── Reductions ──
reduce N=10,000                         175 μs       65 μs       2.70x
reduce N=100,000                        174 μs       64 μs       2.71x
reduce N=1,000,000                      240 μs       78 μs       3.10x
reduce N=10,000,000                     272 μs      165 μs       1.65x

── MapReduce ──
mapreduce N=10,000                      175 μs       65 μs       2.70x
mapreduce N=100,000                     173 μs       64 μs       2.70x
mapreduce N=1,000,000                   240 μs       78 μs       3.07x
mapreduce N=10,000,000                  285 μs      176 μs       1.62x

── Accumulate (prefix sum) ──
accumulate N=10,000                     294 μs       67 μs       4.38x
accumulate N=100,000                    369 μs      113 μs       3.26x
accumulate N=1,000,000                 1230 μs      468 μs       2.64x

── Sort ──
sort N=1,000                            300 μs      102 μs       2.93x
sort N=10,000                           576 μs      163 μs       3.53x
sort N=100,000                          859 μs      486 μs       1.77x
sort N=1,000,000                       3430 μs     3820 μs       0.90x  ← Vulkan FASTER

── Sortperm ──
sortperm N=1,000                        417 μs      158 μs       2.64x
sortperm N=10,000                       723 μs      341 μs       2.12x
sortperm N=100,000                     1220 μs      625 μs       1.96x

── SearchSortedFirst ──
searchsortedfirst N=100K, M=10K          75 μs       42 μs       1.77x
searchsortedfirst N=1M, M=100K          107 μs       78 μs       1.37x

── Foreachindex ──
foreachindex N=10,000                   112 μs       45 μs       2.47x
foreachindex N=100,000                  180 μs       90 μs       2.00x
foreachindex N=1,000,000               1020 μs      562 μs       1.81x
foreachindex N=10,000,000             38850 μs    19930 μs       1.95x

── Any ──
any N=10,000                            156 μs       61 μs       2.55x
any N=100,000                           168 μs       64 μs       2.64x
any N=1,000,000                         175 μs       85 μs       2.05x

── KA Matmul ──
matmul 128×128                          111 μs       90 μs       1.24x
matmul 256×256                          182 μs      290 μs       0.63x  ← Vulkan FASTER
matmul 512×512                          433 μs     1590 μs       0.27x  ← Vulkan FASTER
```

**Analysis**: Async batch dispatch (commands.jl) records all dispatches into a single
command buffer with compute→compute memory barriers, submitting only at sync points
(flush/copy). This eliminated ~46μs per-dispatch fence overhead. Key wins:
- **Sort 1M: Vulkan beats AMDGPU** (3.4ms vs 3.8ms) — 156 dispatches batched into one submit
- Matmul: Vulkan 2-4x faster at 256+ (single dispatch, compute-bound)
- Reduce/mapreduce: still ~2-3x slower at small N (dominated by fixed dispatch cost)

### Known Issues
- Raw `@vulkan`-style kernels (bypassing KA) have a pointer indexing issue:
  `unsafe_store!(ptr::LLVMPtr, val, idx)` — the index is dropped in SPIR-V.
  VkDeviceArray works around this by computing address arithmetic in Julia.
  Not blocking anything since KA path works fine.

### Pending
- **AcceleratedKernels-Benchmark** repo benchmarks (RBF kernel, LJG potential)
- Investigate reduce dispatch overhead (2.7x ratio vs 1.5x for simple ops)

---

## File Locations

| File | Purpose |
|------|---------|
| `src/vulkan/VulkanKernels.jl` | KA VulkanBackend — indexing, SharedMemory, launch |
| `src/vulkan/intrinsics.jl` | Device intrinsics — built-in IDs, barrier, shared memory |
| `src/vulkan/compilation.jl` | GPUCompiler → LLVM IR → SPIR-V pipeline |
| `src/vulkan/spirv_fixup.jl` | Text-based SPIR-V post-processor |
| `src/vulkan/pipeline.jl` | VkComputePipeline creation + caching |
| `src/vulkan/array.jl` | VkArray (GPUArrays), VkDeviceArray (device-side) |
| `src/vulkan/init.jl` | Vulkan instance/device/queue singleton |
| `src/vulkan/memory.jl` | VkManagedBuffer + BDA |
| `src/device/quirks.jl` | Shared device overrides (div/rem, bounds, math) |
