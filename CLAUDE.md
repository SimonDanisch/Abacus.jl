# Abacus — Vulkan GPU Backend for Julia

Abacus provides a Vulkan compute backend for Julia, compiling Julia code to SPIR-V
shaders via GPUCompiler → LLVM IR → llvm-spirv → SPIR-V binary fixup.

## Architecture

```
Julia source
  → GPUCompiler (LLVM IR)
  → vkcompile (LLVM optimizations + SPIR-V-specific transforms)
  → llvm-spirv (SPIRV-LLVM-Translator, OpenCL-style SPIR-V)
  → spirv_fixup.jl (binary post-processor: OpenCL → Vulkan)
  → spirv-val (validation)
  → vkCreateComputePipelines
```

### Key Source Files

| File | Purpose |
|------|---------|
| `src/vulkan/compilation.jl` | GPUCompiler → LLVM IR → SPIR-V pipeline (`vkcompile`) |
| `src/vulkan/spirv_fixup.jl` | Binary SPIR-V post-processor (OpenCL → Vulkan conversion) |
| `src/vulkan/intrinsics.jl` | Device intrinsics: barrier, shared memory, built-in IDs |
| `src/vulkan/mapreduce.jl` | Custom GPU reductions (compile-time unrolled tree reduction) |
| `src/vulkan/pipeline.jl` | VkComputePipeline creation, caching, spirv-val guard |
| `src/vulkan/array.jl` | VkArray (host, GPUArrays), VkDeviceArray (device), VkPtr |
| `src/vulkan/memory.jl` | VkManagedBuffer + BDA (buffer device address) |
| `src/vulkan/commands.jl` | Async batch dispatch (command buffer recording) |
| `src/vulkan/VulkanKernels.jl` | KernelAbstractions VulkanBackend |
| `src/device/utils.jl` | Method table macros (@device_override, @vk_device_override) |
| `src/device/quirks.jl` | GPU device overrides (div/rem, bounds, math, error paths) |

## Test Suite

Run: `julia --project=. test/runtests.jl`

Three test tiers (ordered by complexity):

1. **Stage 1: SPIR-V MWE Regression Tests** (`test/spirv_mwe_tests.jl`)
   - Minimal working examples for specific SPIR-V compilation challenges
   - Each test isolates ONE known-problematic pattern
   - Must all pass before proceeding to Stage 2

2. **Stage 2: GPUArrays Test Suite** (`test/gpuarrays_testsuite.jl`)
   - Standard GPUArrays conformance tests
   - Tests broadcast, reductions, indexing, linear algebra
   - Known skips documented with reasons

3. **Stage 3: Ray Tracing Integration** (`test/raytracing_test.jl`)
   - End-to-end ray tracing on Vulkan backend
   - Tests the full rendering pipeline (BVH, materials, integrators)

## SPIR-V Compilation Pipeline — Known Challenges & Solutions

### LLVM-Level Transforms (compilation.jl, `vkcompile`)

| Transform | Problem | Solution |
|-----------|---------|----------|
| `_hoist_push_constant_loads!` | ConstantExpr GEP on push constants in non-entry blocks produces byte-offset arithmetic | Hoist all addrspace(2) loads to entry block |
| `_fix_shared_geps!` | InstCombine flattens two-level GEPs on shared memory; SPIR-V backend drops flat GEP indices | Rewrite flat GEPs back to structured `[array, gv, 0, idx]` form |
| `_replace_freeze!` | LLVM `freeze` instruction not supported by SPIR-V backend | Replace with operand (safe — SPIR-V has no poison/undef) |
| `_replace_unreachable!` | `throw(nothing)` → unreachable creates dead-end blocks that can't be structured | Replace with `gpu_report_exception()` + return |
| `_lift_byte_geps_on_allocas!` | SROA rewrites struct access as `gep i8, ptr %alloca, <offset>` → crashes SPIR-V legalize_bitcast | Convert to typed struct GEPs with member path |
| `_fixup_structured_cfg!` | Blocks targeted by multiple branches break StructurizeCFG loop structuring | Insert trampoline blocks pre-StructurizeCFG |
| `_fix_shared_geps! Part 0.5` | Heterogeneous shared memory types need two-level GEPs merged | Merge array+struct GEPs into single multi-index GEP |
| AlwaysInliner + SROA + InstCombine | Alloca+store from byval struct handling → store_deref in NIR that RADV ACO can't handle | Inline + scalar optimize eliminates allocas |
| SimplifyCFG | Nested conditionals without SimplifyCFG → two OpSelectionMerge targeting same block | Run before StructurizeCFG |

### SPIR-V Binary Fixup Passes (spirv_fixup.jl)

llvm-spirv outputs OpenCL-style SPIR-V. These passes convert to Vulkan-compliant:

| Pass | What it does |
|------|-------------|
| `fix_ext_inst_import!` | OpenCL.std → GLSL.std.450 instruction remapping |
| `fix_capabilities!` | Strip Addresses/Linkage/Kernel; add Shader/PhysicalStorageBuffer/VariablePointers |
| `fix_memory_model!` | OpenCL(2) → GLSL450(1) |
| `fix_entry_points!` | Synthesize OpEntryPoint (llvm-spirv uses Linkage model, no entry points) |
| `fix_execution_modes!` | Create LocalSize execution mode |
| `fix_decorations!` | Add Block/Offset decorations for push constant structs; strip Alignment on Workgroup |
| `fix_storage_classes!` | CrossWorkgroup → PhysicalStorageBuffer, UniformConstant → PushConstant |
| `fix_structured_control_flow!` | Add OpSelectionMerge/OpLoopMerge (llvm-spirv emits NONE) |
| `fix_merge_placement!` | Reorder merge instructions to immediately precede branch |
| `fix_dead_blocks!` | Remove orphan blocks (unreachable OpLabels from trampoline insertion) |
| `fix_duplicate_merge_targets!` | Insert trampoline when multiple merges target same block |
| `fix_merge_single_pred_blocks!` | Merge single-predecessor blocks (like spirv-opt --merge-blocks) |
| `fix_dead_functions!` | Remove Julia runtime stubs with invalid SPIR-V |
| `fix_copy_memory_sized!` | OpCopyMemorySized → OpCopyMemory (type-safe, pre-opt) |
| `fix_pushconstant_bitcast!` | Create uint32 type/zero const for OpAccessChain index |

### Structured Control Flow (the hardest part)

SPIR-V requires **structured control flow**: every `OpBranchConditional` must be preceded
by `OpSelectionMerge` or be a loop continue target. Key concepts:

- **Forward dominators** detect back-edges (edge A→B where B dominates A = loop)
- **Natural loop bodies** computed from back-edges via reverse reachability
- **ipdom** (immediate post-dominator) determines selection merge targets
- **Continue targets** are allowed bare `OpBranchConditional` (no selection merge needed)
  - Even if the continue target conditionally exits the loop (break from continue is valid)
- **Trampoline blocks** resolve when:
  - Selection ipdom = loop continue target (must split continue target)
  - Selection ipdom = loop merge target (duplicate merge — redirect)
  - Multiple blocks share the same trampoline → track with `trampolined_continues` dict

#### Active Issue: Conditional Loop Exit from Latch

Pattern: latch block conditionally exits the loop (`OpBranchConditional %cond %exit %header`).
When the latch IS the continue target:
- The latch CAN have a bare `OpBranchConditional` (continue target exemption)
- It CAN branch to the loop merge (break from continue is valid per SPIR-V spec 2.11.2)
- The trampoline approach creates unreachable merge targets that become phantom predecessors

Current status: Working on ensuring continue targets that conditionally exit are properly
handled without generating unreachable blocks.

## Device Overrides

- **@consistent_overlay** (not @overlay) required on Julia >= 1.12 for kwargs constant folding
- Overlay registration happens at package load time — requires Julia restart, Revise can't reload
- Key overrides in `quirks.jl`:
  - `div`/`rem` → unchecked `sdiv_int`/`srem_int` (checked versions generate invalid OpSwitch)
  - `isless(::Float32)` → simple `<` (avoids OpOrdered requiring Kernel capability)
  - Error-throwing functions → `throw(nothing)` to avoid dynamic dispatch

## Mapreduce Strategy

Custom implementation in `mapreduce.jl` (NOT GPUArrays default):
- **Compile-time unrolled** tree reduction with barriers (no while loop)
- While loops with barriers inside produce "Block is already a merge block" in SPIR-V
- Bounded `for` loop for grid stride (MAX_ITERS computed host-side)
- Two-pass reduction for large arrays

## Async Batch Dispatch

`commands.jl` records all GPU dispatches into a single Vulkan command buffer with
compute→compute memory barriers, submitting only at sync points (flush/copy).
This eliminates ~46μs per-dispatch fence overhead. Sort with 156 dispatches runs in
one submit, making Vulkan competitive with (sometimes faster than) AMDGPU.

## Debugging & Validation

- **spirv-val**: `~/.julia/artifacts/8769e627f0911e0e6ab9b58f5961768a06de3e61/bin/spirv-val`
- **spirv-dis**: same directory
- **Debug files**: spirv_fixup.jl writes to `/tmp/debug_pre_fixup.spv` and `/tmp/debug_fixup_output.spv`
- **Validate**: `spirv-val --target-env vulkan1.3 /tmp/debug_fixup_output.spv`
- **Disassemble**: `spirv-dis /tmp/debug_fixup_output.spv -o /tmp/debug.dis`
- **Acceptable validation error**: barrier SequentiallyConsistent semantics (`VUID-StandaloneSpirv-MemorySemantics-10866`)

### Revise Caveats

- `intrinsics.jl:356` has persistent Revise TypeError — restart if intrinsics changes are needed
- After @eval-ing spirv_fixup/compilation functions, clear: `_pipeline_cache`, `_vk_compiler_cache`, `_vk_kernel_cache`
- NEVER @eval @generated functions (like `vk_localmemory`) — corrupts generated cache

## Known Limitations

- **Shared memory heterogeneous types** (e.g. `Tuple{Bool,Int64}`): ptrtoint on addrspace(3) pointer not allowed in Vulkan; workaround via MArray quirks override
- **Int64 width**: Some SPIR-V backends reject Int64 in complex if-else chains; use Int32 explicitly
- **Float64 transcendentals**: GLSL.std.450 only supports Float32; `supported_eltypes = (Float32,)`
- **No GPU RNG**: GPUArrays random tests skipped
- **No resize!**: GPUArrays vectors tests skipped
