# Thread index intrinsics
#
# Per-thread kernel state stored in a fixed raw buffer (never moved by GC).
# The launch loop writes state before each compiled kernel call (for non-compiled
# Julia paths). Compiled kernels receive KernelState as their first LLVM argument
# via GPUCompiler's kernel_state_type mechanism.
#
# @device_override versions use @generated to splice the llvmcall expression from
# GPUCompiler.kernel_state_value(KernelState) directly into each function body.
# That expression calls julia.gpu.state_getter which, after add_kernel_state!,
# lowers to parameters(fun)[1] — the KernelState passed by the execution loop.

# Fixed-size raw buffer for per-thread KernelState (one slot per Julia thread).
# Non-const so __init__ can allocate a fresh buffer each time the module is loaded.
const _KERNEL_STATES_MAX_THREADS = 256
_kernel_states_ptr::Ptr{KernelState} = Ptr{KernelState}(C_NULL)

# Called from Abacus.__init__() — allocates a fresh, process-local buffer.
function _init_kernel_states!()
    global _kernel_states_ptr
    _kernel_states_ptr = Ptr{KernelState}(Libc.calloc(_KERNEL_STATES_MAX_THREADS, sizeof(KernelState)))
end

# Write kernel state for the current Julia thread (used by the non-compiled path).
@inline function _set_kernel_state!(state::KernelState)
    tid = Threads.threadid()  # 1-based
    unsafe_store!(_kernel_states_ptr, state, tid)
end

# Regular Julia versions (non-compiled code, e.g. AbacusContext dispatch)
@inline thread_position_in_threadgroup_x() = unsafe_load(_kernel_states_ptr, Threads.threadid()).local_x
@inline thread_position_in_threadgroup_y() = unsafe_load(_kernel_states_ptr, Threads.threadid()).local_y
@inline thread_position_in_threadgroup_z() = unsafe_load(_kernel_states_ptr, Threads.threadid()).local_z

@inline threadgroup_position_in_grid_x() = unsafe_load(_kernel_states_ptr, Threads.threadid()).group_x
@inline threadgroup_position_in_grid_y() = unsafe_load(_kernel_states_ptr, Threads.threadid()).group_y
@inline threadgroup_position_in_grid_z() = unsafe_load(_kernel_states_ptr, Threads.threadid()).group_z

# @device_override versions (compiled into kernels via GPUCompiler)
# GPUCompiler.kernel_state_value(KernelState) returns an Expr (not a value) that,
# when eval'd as a function body, emits a llvmcall to julia.gpu.state_getter.
# After the add_kernel_state! pass, that intrinsic is replaced by parameters(f)[1].
# We use @generated to splice that Expr into the function body at compile time.
@device_override @inline @generated thread_position_in_threadgroup_x() =
    :($(GPUCompiler.kernel_state_value(KernelState)).local_x)
@device_override @inline @generated thread_position_in_threadgroup_y() =
    :($(GPUCompiler.kernel_state_value(KernelState)).local_y)
@device_override @inline @generated thread_position_in_threadgroup_z() =
    :($(GPUCompiler.kernel_state_value(KernelState)).local_z)

@device_override @inline @generated threadgroup_position_in_grid_x() =
    :($(GPUCompiler.kernel_state_value(KernelState)).group_x)
@device_override @inline @generated threadgroup_position_in_grid_y() =
    :($(GPUCompiler.kernel_state_value(KernelState)).group_y)
@device_override @inline @generated threadgroup_position_in_grid_z() =
    :($(GPUCompiler.kernel_state_value(KernelState)).group_z)
