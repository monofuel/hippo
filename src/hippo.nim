import
  std/[strformat, os]

## Nim Library to enable writing CUDA and HIP kernels in Nim
## All cuda and hip structures and functions are re-exported and can be used
##
## - There are 3 sets of function prefixes.
## - `hippo*` prefixed functions are friendly nim interfaces for either HIP or CUDA
##   - This is the recommended way to use this library, as it is the most nim-like
##   - These functions check for errors and raise them as exceptions
## - `hip*` prefixed functions are the raw HIP C++ functions
## - `cuda*` prefixed functions are the raw CUDA C functions


# -------------------
# Compiler Specifics
# this section is for special cases to add libraries or change includes depending on what compiler and settings are being used.

proc getHipPlatform(): string =
  ## getHipPlatform is a compile time specific function, and gets the target platform for hipcc.
  ## NVCC and HIPCC (when building for nvidia) require that we pass compiler args in -Xcompiler="".
  ## hipcc uses HIP_PLATFORM to determine amd / nvidia.
  ## hipcc defaults to amd if amdclang++ or hip clang is found.
  ## https://github.com/ROCm/llvm-project/blob/00fdfae9aeef14c905550601c2218a6b5962f48c/amd/hipcc/bin/hipvars.pm#L131
  let 
    clangPath = getEnv("HIP_CLANG_PATH", "") / "clang++"
    amdClangPath = "/opt/rocm/bin/amdclang++"
    defaultPlatform = if (fileExists(clangPath) or fileExists(amdClangPath)): "amd" else: "nvidia"
    hipPlatform = getEnv("HIP_PLATFORM", defaultPlatform)
  result = hipPlatform

const HipPlatform = getHipPlatform()

# HippoRuntime can be set to "HIP", "HIP_CPU", or "CUDA"
# HIP hipcc will auto detect the runtime of the build system
# HIP_CPU will use the HIP-CPU runtime header
# CUDA will use nvcc

const HippoRuntime* {.strdefine.} = "HIP"

echo &"DEBUG: Using Hippo Runtime: {HippoRuntime}"
if HippoRuntime == "HIP":
  echo &"DEBUG: Using HIP Platform: {HipPlatform}"

when HippoRuntime == "HIP_CPU":
  # Intel TBB is required for HIP-CPU
  {.passL: "-ltbb".}
  # I forgot when I had to use stdc++? maybe it's needed?
  # {.passL: "-lstdc++".}

  # hip.nim expects hip/hip_runtime.h to be in the include path
  # for HIP-CPU, we want to dynamically set the include path
  # the include path is in this library, but it needs to be imported into the user's project
  const
    hipIncludePath = staticExec("pwd") & "/../HIP-CPU/include"
  {.passC: "-I" & hipIncludePath.}
  echo "DEBUG: Using HIP CPU runtime"
  include hip
elif HippoRuntime == "CUDA":
  # nvcc loads the CUDA runtime header automatically
  echo "DEBUG: Using CUDA runtime"
  include cuda
else:
  echo "DEBUG: Using HIP runtime"
  include hip

# -------------------
# Hippo Templates
# nim wrappers around hip and cuda functions.
# these hippo* functions need to be nim-friendly and warp around the raw hip and cuda functions.

type
  GpuMemory = object ## Wrapper around gpu memory for automatic cleanup
    p*: pointer

template hippoMalloc*(size: int): GpuMemory =
  ## Allocate memory on the GPU and return a pointer to it.
  var g = GpuMemory()
  when HippoRuntime == "CUDA":
    handleError(cudaMalloc(addr g.p, size.cint))
  else:
    handleError(hipMalloc(addr g.p, size.cint))
  g

template hippoMemcpy*(dst: pointer, src: pointer, size: int, kind: HippoMemcpyKind) =
  ## host -> host memory copy
  ## hippoMemcpy is broken out as 4 separate templates to make it easier to work with GpuMemory objects
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst, src, size.cint, kind))
  else:
    handleError(hipMemcpy(dst, src, size.cint, kind))

template hippoMemcpy*(dst: pointer, src: GpuMemory, size: int, kind: HippoMemcpyKind) =
  ## host -> device memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst, src.p, size.cint, kind))
  else:
    handleError(hipMemcpy(dst, src.p, size.cint, kind))

template hippoMemcpy*(dst: GpuMemory, src: pointer, size: int, kind: HippoMemcpyKind) =
  ## device -> host memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst.p, src, size.cint, kind))
  else:
    handleError(hipMemcpy(dst.p, src, size.cint, kind))

template hippoMemcpy*(dst: GpuMemory, src: GpuMemory, size: int, kind: HippoMemcpyKind) =
  ## device -> device memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst.p, src.p, size.cint, kind))
  else:
    handleError(hipMemcpy(dst.p, src.p, size.cint, kind))

template hippoFree*(p: pointer) =
  ## Free memory on the GPU
  when HippoRuntime == "CUDA":
    handleError(cudaFree(p))
  else:
    handleError(hipFree(p))

proc `=destroy`(mem: var GpuMemory) =
  ## Automatically free device memory when the object goes out of scope
  echo "AUTO FREE"
  if mem.p != nil:
    hippoFree(mem.p)
    mem.p = nil

# -------------------
# Kernel Execution

proc launchKernel*(
  kernel: proc,
  gridDim: Dim3 = newDim3(1,1,1), # default to a grid of 1 block
  blockDim: Dim3 = newDim3(1,1,1),  # default to 1 thread per block
  sharedMemBytes: uint32 = 0,
  stream: HippoStream = nil,
  args: tuple
): HippoError =
  # launchKernel is designed to be similar to `kernel`<<<blockDim, gridDim>>>(args)

  # this function is horrible but it works
  # needs to be refactored to handle all the different runtimes and arguments better

  # having some issues between hip and hip-cpu, so defining different versions of launchKernel
  when HippoRuntime == "HIP" and HipPlatform == "amd":
    # This branch works for all args
    echo "executing HIP"
    var kernelArgs: seq[pointer]
    for key, arg in args.fieldPairs:
      kernelArgs.add(cast[pointer](addr arg))
    result = hipLaunchKernel(
      cast[pointer](kernel),
      gridDim,
      blockDim,
      cast[ptr pointer](addr kernelArgs[0]),
    )
  elif HippoRuntime == "HIP" and HipPlatform == "nvidia":
    # TODO fix args on this branch
    hipLaunchKernelGGL(
      kernel,
      gridDim,
      blockDim,
      0, # TODO
      nil, # TODO
      # TODO handle args properly
      cast[ptr[cint]](args[0]),
      cast[ptr[cint]](args[1]),
      cast[ptr[cint]](args[2])
      )
    result = hipGetLastError()
  elif HippoRuntime == "HIP_CPU":
    # TODO fix args on this branch
    echo "executing kernel on CPU"
    hipLaunchKernelGGL(
      kernel,
      gridDim,
      blockDim,
      0, # TODO
      nil, # TODO
      # TODO handle args properly
      args[0],
      args[1],
      args[2]
    )
    result = hipGetLastError()
  elif HippoRuntime == "CUDA":
    # This branch works for all args
    echo "executing CUDA"
    var kernelArgs: seq[pointer]
    for key, arg in args.fieldPairs:
      kernelArgs.add(cast[pointer](addr arg))
    result = cudaLaunchKernel(
      kernel,
      gridDim,
      blockDim,
      cast[ptr pointer](addr kernelArgs[0])
      #sharedMemBytes,
      #stream
    )
  else:
    raise newException(Exception, &"Unknown runtime: {HippoRuntime}")

template hippoLaunchKernel*(
  kernel: proc,                     ## The GPU kernel procedure to launch
  gridDim: Dim3 = newDim3(1,1,1),   ## default to a grid of 1 block
  blockDim: Dim3 = newDim3(1,1,1),  ## default to 1 thread per block
  sharedMemBytes: uint32 = 0,       ## dynamic shared memory amount to allocate
  stream: HippoStream = nil,        ## Which device stream to run under (defaults to null)
  args: tuple,                      ## Arguments to pass to the GPU kernel
) =
  ## Launch a kernel on the GPU and check for errors
  handleError(launchKernel(kernel, gridDim, blockDim, sharedMemBytes, stream, args))


# -------------------
# Macros
# these Nim macros wrap around the attributes required by cuda and hip (which are identical, and as such are here and not in hip.nim or cuda.nim).

macro hippoGlobal*(fn: untyped): untyped =
  ## Declare a function as `__global__`. global functions are called from the host and run on the device.
  let globalPragma: NimNode = quote:
    {. exportc, codegenDecl: "__global__ $# $#$#".}

  fn.addPragma(globalPragma[0])
  fn.addPragma(globalPragma[1])
  quote do:
    {.push stackTrace: off, checks: off.}
    `fn`
    {.pop.}

macro hippoDevice*(fn: untyped): untyped =
  ## Declare fuctions for use on the `__device__` (the gpu),
  ## to be called by either `device` or `global` functions.
  let globalPragma: NimNode = quote:
    {. exportc, codegenDecl: "__device__ $# $#$#".}

  fn.addPragma(globalPragma[0])
  fn.addPragma(globalPragma[1])
  quote do:
    {.push stackTrace: off, checks: off.}
    `fn`
    {.pop.}


macro hippoHost*(fn: untyped): untyped =
  ## Explicitly declare a function as a `__host__` function (cpu side).
  ## All functions default to `host` functions, so this is not required.
  let globalPragma: NimNode = quote:
    {. exportc, codegenDecl: "__host__ $# $#$#".}

  fn.addPragma(globalPragma[0])
  fn.addPragma(globalPragma[1])
  quote do:
    {.push stackTrace: off, checks: off.}
    `fn`
    {.pop.}

macro hippoShared*(v: untyped): untyped =
  ## Declared a variable as static shared memory `__shared__`.
  ## Shared memory is shared between threads in the same block.
  ## It is faster than global memory, but is limited in size. They are located on-chip.
  ## eg: `var cache {.hippoShared.}: array[256, float]`
  quote do:
    {.push stackTrace: off, checks: off, noinit, exportc, codegenDecl: "__shared__ $# $#".}
    `v`
    {.pop.}

macro hippoConstant*(v: untyped): untyped =
  ## Declared a variable as `__constant__`.
  ## Constants are read-only globals that are cached on-chip.
  ## constants are useful for data that is being read by all threads in a warp at the same time.
  ## if each thread in a warp accesses different addresses in constant memory,
  ## the accesses are serialized and this may cause a 16x slowdown.
  ## eg: `const N {.hippoConstant.} = 1024`
  quote do:
    {.push stackTrace: off, checks: off, noinit, exportc, codegenDecl: "__constant__ $# $#".}
    `v`
    {.pop.}