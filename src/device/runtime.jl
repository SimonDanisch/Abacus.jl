# device runtime libraries

## Julia library

# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

signal_exception() = return
report_exception(ex) = return
report_oom(sz) = return
report_exception_name(ex) = return
report_exception_frame(idx, func, file, line) = return

## kernel state

# Thread indices stored as a plain struct. The launch loop writes to
# _current_kernel_state (a global Ref) before each compiled kernel call.
# Both regular Julia and @device_override intrinsics read from the same Ref.
struct KernelState
    group_x::Int32
    group_y::Int32
    group_z::Int32
    local_x::Int32
    local_y::Int32
    local_z::Int32
end

## memory allocation stubs
# GPUCompiler's runtime expects malloc/free for dynamic allocation (MArray etc.).
# We use llvmcall to avoid ccall PLT stubs contaminating every compiled module.
malloc(sz) = reinterpret(Ptr{Nothing}, Base.llvmcall(
    ("""
     declare ptr @malloc(i64)
     define i64 @entry(i64 %0) {
         %ptr = call ptr @malloc(i64 %0)
         %int = ptrtoint ptr %ptr to i64
         ret i64 %int
     }
     """, "entry"),
    UInt, Tuple{UInt}, UInt(sz)))

free(ptr) = Base.llvmcall(
    ("""
     declare void @free(ptr)
     define void @entry(i64 %0) {
         %ptr = inttoptr i64 %0 to ptr
         call void @free(ptr %ptr)
         ret void
     }
     """, "entry"),
    Nothing, Tuple{UInt}, reinterpret(UInt, ptr))
