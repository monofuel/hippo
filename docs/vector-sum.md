# Basic Tutorial on GPU Computing and Using Hippo in Nim

- [vector sum code](../examples/vector_sum_hippo.nim)
- [vector sum compiler args](../examples/vector_sum_hippo.nims)

Hippo is a Nim wrapper around HIP that enables portable GPU code for both AMD (ROCm) and NVIDIA (CUDA) GPUs. This tutorial covers the basics using a vector addition example.
Hippo also has experimental backends for CPU.

See the [Hippo documentation](https://monofuel.github.io/hippo/) and [examples](https://github.com/monofuel/hippo) for more details.

## Overview of GPU Computing

GPUs accelerate data-parallel tasks by executing the same instruction across thousands of cores simultaneously (SIMD / SIMT).

Key concepts:
- **Host**: CPU and main system memory.
- **Device**: GPU and its dedicated memory (VRAM), typically separate from host memory.
  - Note: some systems have unified memory (eg: AMD Ryzen AI Max, Steamdeck, or Apple Silicon). sometimes the memory is shared, and sometimes a portion is dedicated to the device and acts like a traditional VRAM.
- **Kernel**: A function that runs on the GPU, executed by many threads in parallel.

## Defining GPU Functions

Hippo uses pragmas to mark procedures:

- **Host Functions**: Default Nim procedures (no pragma). Handle setup, data transfer, and launching kernels.
- **Device Functions**: `{.hippoDevice.}` - Helper functions callable only from device/global functions.
- **Global Kernels**: `{.hippoGlobal.}` - Entry points launched from host, run in parallel across many threads. can call device functions.
- **Host Device Functions**: `{.hippoHostDevice.}` - Functions that can be called from both host and device. these functions are compiled twice, once for the host and once for the device, and can be ran on either. This is useful for helper functions that can be used anywhere, eg: complex imaginary number operations.

Example kernel:
```nim
proc addKernel(a, b, c: ptr[cint]) {.hippoGlobal.} =
  let tid = blockIdx.x  # Thread index
  if tid < N.uint:      # Bounds check
    let aArray = cast[ptr UncheckedArray[cint]](a)
    let bArray = cast[ptr UncheckedArray[cint]](b)
    let cArray = cast[ptr UncheckedArray[cint]](c)
    cArray[tid] = aArray[tid] + bArray[tid]
```
Inside kernels, use `threadIdx`, `blockIdx`, and `blockDim` to determine what data each thread processes.

## GPU Memory Management

GPU memory is separate from host memory. Workflow:
1. Allocate memory on the device.
2. Copy data from host to device.
3. Perform computations on the device.
4. Copy results back to the host.

- **hippoMalloc(size: int)**: Allocates GPU memory, returns `GpuRef`.
  - Example: `var dev_a = hippoMalloc(sizeof(int32) * N)`

- **hippoMemcpy(dest: pointer, src: pointer, size: int, kind: HippoMemcpyKind)**: Copies data between host and device.
  - Directions: `hipMemcpyHostToDevice`, `hipMemcpyDeviceToHost`, etc.
  - Example: `hippoMemcpy(dev_a, addr a[0], sizeof(int32) * N, hipMemcpyHostToDevice)`
  - Use `addr` for host arrays to get raw pointers.

Hippo handles memory cleanup automatically (see Garbage Collection below).

## Kernel Launching

Use `hippoLaunchKernel(kernel: proc, gridDim: Dim3 = newDim3(1,1,1), blockDim: Dim3 = newDim3(1,1,1), sharedMemBytes: uint32 = 0, stream: HippoStream = nil, args: untyped)`.

Threads are grouped into blocks, blocks into a grid. Use `newDim3(x, y, z)` for 1D/2D/3D layouts. For a 1D vector of size N: `gridDim = newDim3(N.uint32)`. Block size defaults to `newDim3(1,1,1)`; typically 128-1024 for performance.

Pass arguments via `hippoArgs(arg1, arg2, ...)`. Launches are asynchronous; use `hippoSynchronize()` to wait.

Example:
```nim
hippoLaunchKernel(
  addKernel,
  gridDim = newDim3(N.uint32),
  args = hippoArgs(dev_a.p, dev_b.p, dev_c.p)  # .p gets the raw pointer
)
```

## Garbage Collection

Hippo uses Nim's GC to manage GPU resources automatically. `GpuRef` types (returned by `hippoMalloc`) are GC-managed. When a `GpuRef` goes out of scope, Hippo calls `hippoFree` internally to release GPU memory.

For long-running programs, manage scopes carefully to avoid holding memory longer than needed.


## CUDA / HIP platform specific magic

- hippo has a set of `hippo` templates and macros to be platform agnostic.
- however you can still reach in and use the underlying platform specific functions.
- eg: `cudaMalloc`, `hipMalloc`, `cudaMemcpy`, `hipMemcpy`, `cudaLaunchKernel`, `hipLaunchKernel`, etc.
- this is useful for operations not supported by hippo templates yet.
