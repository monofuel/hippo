# Hippo macros and pragmas

This document describes the core Hippo pragmas/macros that control how
functions and variables are emitted for GPU backends.

See also: `docs/vector-sum.md` for a basic tutorial.

## Function pragmas

### hippoGlobal
- Marks a kernel entry point that is launched from the host.
- Runs on the device and is executed by many threads in parallel.
- Only callable from host code via `hippoLaunchKernel`.

Example:
```nim
proc addKernel(a, b, c: ptr[int32]) {.hippoGlobal.} =
  let tid = threadIdx.x + blockIdx.x * blockDim.x
  let aArray = cast[ptr UncheckedArray[int32]](a)
  let bArray = cast[ptr UncheckedArray[int32]](b)
  let cArray = cast[ptr UncheckedArray[int32]](c)
  cArray[tid] = aArray[tid] + bArray[tid]
```

### hippoDevice
- Marks a helper function that runs on the device only.
- Callable from `hippoGlobal` or other `hippoDevice`/`hippoHostDevice` functions.

Example:
```nim
proc square(x: int32): int32 {.hippoDevice.} = x * x
```

### hippoHost
- Marks a host-only function (explicit for clarity when mixing host/device code).
- Useful for API symmetry and for avoiding accidental device compilation.

### hippoHostDevice
- Compiles the function for both host and device.
- Useful for helpers shared by kernels and host logic.

Example:
```nim
proc clamp01(x: float32): float32 {.hippoHostDevice.} =
  if x < 0'f32: 0'f32
  elif x > 1'f32: 1'f32
  else: x
```

## Memory pragmas

### hippoShared
- Declares static shared memory in a kernel.
- Shared memory is per-block, low latency, and limited in size.

Example:
```nim
var cache {.hippoShared.}: array[256, float32]
```

### hippoConstant
- Declares device constant memory (read-only, cached).
- Use a device symbol (var/let) for GPU backends. A Nim `const` becomes
  a host-only compile-time constant and cannot be accessed from a kernel
  on HIP/CUDA backends.
- For arrays or larger data, initialize the symbol explicitly with
  `hippoMemcpyToSymbol` before launching kernels.

Example:
```nim
const HostCoeff = 3.5'f32
var DeviceCoeff {.hippoConstant.}: float32 = HostCoeff

proc useCoeff(output: ptr[float32]) {.hippoGlobal.} =
  let tid = threadIdx.x
  let out = cast[ptr UncheckedArray[float32]](output)
  out[tid] = DeviceCoeff * float32(tid)
```

## Launch helpers

### hippoArgs
- Packs kernel arguments in a backend-appropriate way.
- Always use `hippoArgs(...)` when launching kernels with `hippoLaunchKernel`.

Example:
```nim
hippoLaunchKernel(
  addKernel,
  gridDim = newDim3(1'u32),
  blockDim = newDim3(256'u32),
  args = hippoArgs(dev_a.p, dev_b.p, dev_c.p)
)
```
