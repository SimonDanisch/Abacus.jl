# Thread index intrinsics
#
# Per-thread kernel state stored in a fixed raw buffer (never moved by GC).
# Thread ID obtained via jl_threadid (safe in compiled code via llvmcall).
# The launch loop writes state before each compiled kernel call.

# Fixed-size raw buffer for per-thread KernelState (one slot per Julia thread).
# Libc.malloc ensures the pointer is stable (not moved by GC).
const _KERNEL_STATES_MAX_THREADS = 256
const _kernel_states_ptr = Ptr{KernelState}(Libc.calloc(_KERNEL_STATES_MAX_THREADS, sizeof(KernelState)))

# Get 0-based Julia thread ID via llvmcall.
# This compiles cleanly through GPUCompiler (jl_threadid is whitelisted).
@inline _jl_threadid() = Base.llvmcall(
    ("""
     declare i16 @jl_threadid()
     define i32 @entry() {
         %tid = call i16 @jl_threadid()
         %ext = zext i16 %tid to i32
         ret i32 %ext
     }
     """, "entry"),
    Int32, Tuple{})

# Write kernel state for the current Julia thread
@inline function _set_kernel_state!(state::KernelState)
    tid = Threads.threadid()  # 1-based
    unsafe_store!(_kernel_states_ptr, state, tid)
end

# Read kernel state for the current Julia thread (compiled code path)
@inline function _read_kernel_state()
    tid = _jl_threadid() + Int32(1)  # 0-based → 1-based
    unsafe_load(_kernel_states_ptr, tid)
end

# Regular Julia versions (non-compiled code, e.g. AbacusContext dispatch)
@inline thread_position_in_threadgroup_x() = unsafe_load(_kernel_states_ptr, Threads.threadid()).local_x
@inline thread_position_in_threadgroup_y() = unsafe_load(_kernel_states_ptr, Threads.threadid()).local_y
@inline thread_position_in_threadgroup_z() = unsafe_load(_kernel_states_ptr, Threads.threadid()).local_z

@inline threadgroup_position_in_grid_x() = unsafe_load(_kernel_states_ptr, Threads.threadid()).group_x
@inline threadgroup_position_in_grid_y() = unsafe_load(_kernel_states_ptr, Threads.threadid()).group_y
@inline threadgroup_position_in_grid_z() = unsafe_load(_kernel_states_ptr, Threads.threadid()).group_z

# @device_override versions (compiled into .so by GPUCompiler)
# Use _jl_threadid via llvmcall to get thread ID in compiled code.
@device_override @inline thread_position_in_threadgroup_x() = _read_kernel_state().local_x
@device_override @inline thread_position_in_threadgroup_y() = _read_kernel_state().local_y
@device_override @inline thread_position_in_threadgroup_z() = _read_kernel_state().local_z

@device_override @inline threadgroup_position_in_grid_x() = _read_kernel_state().group_x
@device_override @inline threadgroup_position_in_grid_y() = _read_kernel_state().group_y
@device_override @inline threadgroup_position_in_grid_z() = _read_kernel_state().group_z
