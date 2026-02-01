import
  std/[strformat, os, macros]

include ./gpu_macros

# supported runtimes: HIP, HIP_CPU, CUDA, and SIMPLE
const HippoRuntime* {.strdefine.} = "HIP"

# supported platforms: amd, nvidia (auto-detected if not specified)
const HippoPlatform* {.strdefine.} = ""

{.warning: "DEBUG: Using Hippo Runtime: " & HippoRuntime.}
echo &"DEBUG: Using Hippo Runtime: {HippoRuntime}"

when not defined(Nimdoc) and ((defined(c) or defined(js)) and HippoRuntime != "SIMPLE"):
  {.error: "The HIP, CUDA and HIP_CPU backends require compiling with cpp".}

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

  # First check if amdclang++ is available on PATH
  try:
    let whichOutput = staticExec("which amdclang++")
    if whichOutput.len > 0:
      result = "amd"
      return
  except:
    discard

  # Fallback to existing logic
  let
    clangPath = getEnv("HIP_CLANG_PATH", "") / "clang++"
    amdClangPath = "/opt/rocm/bin/amdclang++"
    defaultPlatform = if (fileExists(clangPath) or fileExists(amdClangPath)): "amd" else: "nvidia"
    hipPlatform = getEnv("HIP_PLATFORM", defaultPlatform)
  result = hipPlatform

const HipPlatform = if HippoPlatform.len > 0: HippoPlatform else: getHipPlatform()

# HippoRuntime can be set to "HIP", "HIP_CPU", or "CUDA"
# HIP hipcc will auto detect the runtime of the build system
# HIP_CPU will use the HIP-CPU runtime header
# CUDA will use nvcc

if HippoRuntime == "HIP":
  echo &"DEBUG: Using HIP Platform: {HipPlatform}"

when HippoRuntime == "HIP_CPU":
  # Intel TBB is required for HIP-CPU
  when defined(linux):
    {.passL: "-ltbb".}
  elif defined(windows):
    # using msys2 pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-tbb
    {.passL: "-ltbb12".}
  # I forgot when I had to use stdc++? maybe it's needed?
  # {.passL: "-lstdc++".}

  # hip.nim expects hip/hip_runtime.h to be in the include path
  # for HIP-CPU, we want to dynamically set the include path
  # the include path is in this library, but it needs to be imported into the user's project
  const
    hipIncludePath = staticExec("pwd") & "/../HIP-CPU/include"
  {.passC: "-I" & hipIncludePath.}
  {.define: HIP_CPU_RUNTIME.}
  echo "DEBUG: Using HIP CPU runtime"
  include hip
elif HippoRuntime == "CUDA":
  # nvcc loads the CUDA runtime header automatically
  echo "DEBUG: Using CUDA runtime"
  include cuda
elif HippoRuntime == "SIMPLE":
  echo "DEBUG: Using SIMPLE runtime"
  include simple
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
  GpuRef = ref GpuMemory

template hippoMalloc*(size: int): GpuRef =
  ## Allocate memory on the GPU and return a GpuRef object.
  ## GpuMemory is a wrapper around Gpu allocated pointers.
  ## It will automatically free the memory when it goes out of scope.
  var g = GpuRef()
  when HippoRuntime == "CUDA":
    handleError(cudaMalloc(addr g.p, size.cint))
  elif HippoRuntime == "SIMPLE":
    simpleMalloc(addr g.p, size)
  else:
    handleError(hipMalloc(addr g.p, size.cint))
  g

template hippoMemcpy*(dst: pointer, src: pointer, size: int, kind: HippoMemcpyKind) =
  ## host -> host memory copy
  ## hippoMemcpy is broken out as 4 separate templates to make it easier to work with GpuRef objects
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst, src, size.cint, kind))
  elif HippoRuntime == "SIMPLE":
    simpleMemcpy(dst, src, size, kind)
  else:
    handleError(hipMemcpy(dst, src, size.cint, kind))

template hippoMemcpy*(dst: pointer, src: GpuRef, size: int, kind: HippoMemcpyKind) =
  ## host -> device memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst, src.p, size.cint, kind))
  elif HippoRuntime == "SIMPLE":
    simpleMemcpy(dst, src.p, size, kind)
  else:
    handleError(hipMemcpy(dst, src.p, size.cint, kind))

template hippoMemcpy*(dst: GpuRef, src: pointer, size: int, kind: HippoMemcpyKind) =
  ## device -> host memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst.p, src, size.cint, kind))
  elif HippoRuntime == "SIMPLE":
    simpleMemcpy(dst.p, src, size, kind)
  else:
    handleError(hipMemcpy(dst.p, src, size.cint, kind))

template hippoMemcpy*(dst: GpuRef, src: GpuRef, size: int, kind: HippoMemcpyKind) =
  ## device -> device memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst.p, src.p, size.cint, kind))
  elif HippoRuntime == "SIMPLE":
    simpleMemcpy(dst.p, src.p, size, kind)
  else:
    handleError(hipMemcpy(dst.p, src.p, size.cint, kind))

template hippoMemcpyToSymbol*(symbol: untyped, src: pointer, size: int,
                              offset: int = 0,
                              kind: HippoMemcpyKind = HippoMemcpyHostToDevice) =
  ## Copy data from host to a device symbol (eg: __constant__ storage).
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpyToSymbol(symbol, src, size.csize_t, offset.csize_t, kind))
  elif HippoRuntime == "SIMPLE":
    copyMem(addr symbol, src, size)
  else:
    handleError(hipMemcpyToSymbol(hipSymbol(symbol), src, size.csize_t, offset.csize_t, kind))

template hippoFree*(p: pointer) =
  ## Free memory on the GPU
  when HippoRuntime == "CUDA":
    handleError(cudaFree(p))
  elif HippoRuntime == "SIMPLE":
    simpleFree(p)
  else:
    handleError(hipFree(p))

# TODO maybe this should be called hippoDeviceSynchronize to be more consistent?
template hippoSynchronize*() =
  ## Synchronize the device
  when HippoRuntime == "CUDA":
    handleError(cudaDeviceSynchronize())
  elif HippoRuntime == "SIMPLE":
    # in cpu mode, kernels are performed syncronously
    discard
  else:
    handleError(hipDeviceSynchronize())

# Stream Management
template hippoStreamCreate*(): HippoStream =
  ## Create a new stream
  var stream: HippoStream
  when HippoRuntime == "CUDA":
    handleError(cudaStreamCreate(addr stream))
  elif HippoRuntime == "SIMPLE":
    stream = nil
  else:
    handleError(hipStreamCreate(addr stream))
  stream

template hippoStreamDestroy*(stream: HippoStream) =
  ## Destroy a stream
  when HippoRuntime == "CUDA":
    handleError(cudaStreamDestroy(stream))
  elif HippoRuntime == "SIMPLE":
    discard
  else:
    handleError(hipStreamDestroy(stream))

template hippoStreamSynchronize*(stream: HippoStream) =
  ## Synchronize a specific stream
  when HippoRuntime == "CUDA":
    handleError(cudaStreamSynchronize(stream))
  elif HippoRuntime == "SIMPLE":
    discard
  else:
    handleError(hipStreamSynchronize(stream))

# Async Memory Operations
template hippoMemcpyAsync*(dst: pointer, src: pointer, size: int, kind: HippoMemcpyKind, stream: HippoStream) =
  ## Asynchronous memory copy on a stream
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpyAsync(dst, src, size.csize_t, kind, stream))
  elif HippoRuntime == "SIMPLE":
    hippoMemcpy(dst, src, size, kind)
  else:
    handleError(hipMemcpyAsync(dst, src, size.csize_t, kind, stream))

# Page-locked Host Memory
template hippoHostAlloc*(size: int): pointer =
  ## Allocate page-locked host memory
  var hostPtr: pointer
  when HippoRuntime == "CUDA":
    handleError(cudaHostAlloc(addr hostPtr, size.csize_t, 0.uint32))
  elif HippoRuntime == "HIP" or HippoRuntime == "HIP_CPU":
    handleError(hipHostAlloc(addr hostPtr, size.csize_t, 0.uint32))
  else:
    # For SIMPLE, use regular alloc
    hostPtr = alloc(size)
  hostPtr

template hippoHostFree*(p: pointer) =
  ## Free page-locked host memory.
  when HippoRuntime == "CUDA":
    handleError(cudaFreeHost(p))
  elif HippoRuntime == "HIP" or HippoRuntime == "HIP_CPU":
    handleError(hipHostFree(p))
  else:
    dealloc(p)

# Events for Timing
when HippoRuntime == "CUDA":
  type HippoEvent* = cudaEvent_t
elif HippoRuntime == "HIP" or HippoRuntime == "HIP_CPU":
  type HippoEvent* = hipEvent_t
else:
  type HippoEvent* = pointer

template hippoEventCreate*(): HippoEvent =
  ## Create a timing event.
  var event: HippoEvent
  when HippoRuntime == "CUDA":
    handleError(cudaEventCreate(addr event))
  elif HippoRuntime == "SIMPLE":
    event = nil
  else:
    handleError(hipEventCreate(addr event))
  event

template hippoEventDestroy*(event: HippoEvent) =
  ## Destroy a timing event.
  when HippoRuntime == "CUDA":
    handleError(cudaEventDestroy(event))
  elif HippoRuntime == "SIMPLE":
    discard
  else:
    handleError(hipEventDestroy(event))

template hippoEventRecord*(event: HippoEvent, stream: HippoStream = nil) =
  ## Record an event on a stream.
  when HippoRuntime == "CUDA":
    handleError(cudaEventRecord(event, stream))
  elif HippoRuntime == "SIMPLE":
    discard
  else:
    handleError(hipEventRecord(event, stream))

template hippoEventSynchronize*(event: HippoEvent) =
  ## Wait for an event to complete.
  when HippoRuntime == "CUDA":
    handleError(cudaEventSynchronize(event))
  elif HippoRuntime == "SIMPLE":
    discard
  else:
    handleError(hipEventSynchronize(event))

template hippoEventElapsedTime*(start: HippoEvent, stop: HippoEvent): float32 =
  ## Get elapsed time between two events.
  var ms: cfloat
  when HippoRuntime == "CUDA":
    handleError(cudaEventElapsedTime(addr ms, start, stop))
  elif HippoRuntime == "SIMPLE":
    ms = 0.0
  else:
    handleError(hipEventElapsedTime(addr ms, start, stop))
  ms.float32

# Device Properties
when HippoRuntime == "CUDA":
  type HippoDeviceProp* = cudaDeviceProp
elif HippoRuntime == "HIP" or HippoRuntime == "HIP_CPU":
  type HippoDeviceProp* = hipDeviceProp_t
else:
  type HippoDeviceProp* = object
    deviceOverlap*: cint

template hippoGetDevice*(): cint =
  ## Get the current device index.
  var device: cint
  when HippoRuntime == "CUDA":
    handleError(cudaGetDevice(addr device))
  elif HippoRuntime == "SIMPLE":
    device = 0
  else:
    handleError(hipGetDevice(addr device))
  device

template hippoGetDeviceProperties*(prop: var HippoDeviceProp, device: cint) =
  ## Get device properties for the given device.
  when HippoRuntime == "CUDA":
    handleError(cudaGetDeviceProperties(addr prop, device))
  elif HippoRuntime == "SIMPLE":
    prop.deviceOverlap = 0
  else:
    handleError(hipGetDeviceProperties(addr prop, device))

proc `=destroy`*(mem: var GpuMemory) =
  ## Automatically free device memory when the object goes out of scope
  if mem.p != nil:
    hippoFree(mem.p)
    mem.p = nil

proc hippoRefcopy[T](obj: ref T): GpuRef =
  ## Performs a shallow copy of a ref object to the GPU.
  let size = sizeof(T)
  result = hippoMalloc(size)
  hippoMemcpy(result, addr obj[], size, HippoMemcpyHostToDevice)

proc hippoRefcopy[T](gpuref: GpuRef): ref T =
  ## Copies gpu memory to a new ref object on the host
  let size = sizeof(T)
  result = new T
  hippoMemcpy(addr result[], gpuref, size, HippoMemcpyDeviceToHost)

proc hippoRefcopy[T](gpuref: GpuRef, target: ref T) =
  ## Copies gpu memory to a ref object on the host
  let size = sizeof(T)
  hippoMemcpy(addr target[], gpuref, size, HippoMemcpyDeviceToHost)

# -------------------
# Kernel Execution

when HippoRuntime == "HIP" or HippoRuntime == "HIP_CPU":
  macro hipLaunchKernelGGLWithTuple(
    kernel: untyped,
    gridDim: Dim3 = newDim3(1,1,1),
    blockDim: Dim3 = newDim3(1,1,1),
    sharedMemBytes: uint32 = 0,
    stream: HippoStream = nil,
    args: tuple
    ): untyped =

    var callNode = newCall(bindSym"hipLaunchKernelGGL")

    # add the fixed vars
    callNode.add kernel
    callNode.add gridDim
    callNode.add blockDim
    callNode.add sharedMemBytes
    callNode.add stream

    # add every value of the tuple
    for child in args:
      callNode.add child
    result = callNode

template hippoLaunchKernel*(
  kernel: untyped,                  ## The GPU kernel procedure to launch
  gridDim: Dim3 = newDim3(1,1,1),   ## default to a grid of 1 block
  blockDim: Dim3 = newDim3(1,1,1),  ## default to 1 thread per block
  sharedMemBytes: uint32 = 0,       ## dynamic shared memory amount to allocate
  stream: HippoStream = nil,        ## Which device stream to run under (defaults to null)
  args: untyped,     ## tuple of pointers to arguments (pointers to arguments! not arguments!) to pass to the GPU kernel
) =
  var result: HippoError
  ## Launch a kernel on the GPU.
  ## also checks if launchKernel() returns an error.
  ## Important: this only checks if the kernel launch was successful, not the kernel itself.
  ## If you need help debugging, you can call hippoSynchronize() to wait for the kernel to finish and report errors.

  when HippoRuntime == "HIP" and HipPlatform == "amd":
    var kernelArgs: seq[pointer] = cast[seq[pointer]](args)
    result = hipLaunchKernel(
      cast[pointer](kernel),
      gridDim,
      blockDim,
      addr kernelArgs[0],
      sharedMemBytes,
      stream
    )
  elif (HippoRuntime == "HIP" and HipPlatform == "nvidia") or HippoRuntime == "HIP_CPU":
    hipLaunchKernelGGLWithTuple(
      kernel,
      gridDim,
      blockDim,
      sharedMemBytes,
      stream,
      args
    )
    result = hipGetLastError()
  elif HippoRuntime == "CUDA":
    var kernelArgs: seq[pointer] = cast[seq[pointer]](args)
    result = cudaLaunchKernel(
      kernel,
      gridDim,
      blockDim,
      addr kernelArgs[0],
      sharedMemBytes,
      stream
    )
  elif HippoRuntime == "SIMPLE":
    simpleLaunchKernel(kernel, gridDim, blockDim, args)
  else:
    raise newException(Exception, &"Unknown runtime: {HippoRuntime}")

  handleError(result)
  


# -------------------
# Macros
# these Nim macros wrap around the attributes required by cuda and hip (which are identical, and as such are here and not in hip.nim or cuda.nim).

macro hippoGlobal*(fn: untyped): untyped =
  ## Declare a function as `__global__`. global functions are called from the host and run on the device.
  when HippoRuntime == "SIMPLE":
    # SIMPLE needs kernel bodies as iterators so we can yield at hippoSyncthreads.
    expectKind(fn, nnkProcDef)
    let name = fn[0]
    let generics = fn[1]
    let params = fn[3]
    let pragmas = if fn.len > 4 and fn[4].kind == nnkPragma: fn[4] else: newEmptyNode()
    let body = fn[^1]

    let iterBody = newStmtList()
    if body.kind == nnkStmtList:
      for stmt in body:
        iterBody.add(stmt)
    else:
      iterBody.add(body)
    iterBody.add(newTree(nnkYieldStmt, newLit(false)))

    let anonIter = newNimNode(nnkIteratorDef)
    anonIter.add(newEmptyNode())
    anonIter.add(newEmptyNode())
    anonIter.add(newEmptyNode())
    anonIter.add(newNimNode(nnkFormalParams).add(ident("bool")))
    anonIter.add(newEmptyNode())
    anonIter.add(newEmptyNode())
    anonIter.add(iterBody)

    let returnBody = newStmtList()
    returnBody.add(newTree(nnkReturnStmt, anonIter))

    let newParams = params.copyNimTree()
    if newParams.len > 0:
      newParams[0] = ident("auto")
    else:
      newParams.insert(0, ident("auto"))

    result = newNimNode(nnkProcDef)
    result.add(name)
    result.add(generics)
    result.add(newEmptyNode())
    result.add(newParams)
    result.add(pragmas)
    result.add(newEmptyNode())
    result.add(returnBody)
  else:
    let globalPragma: NimNode = quote:
      {. exportc, codegenDecl: "__global__ $# $#$#".}

    fn.addPragma(globalPragma[0])
    fn.addPragma(globalPragma[1])
    result = quote do:
      {.push stackTrace: off, checks: off.}
      `fn`
      {.pop.}

macro hippoDevice*(fn: untyped): untyped =
  ## Declare fuctions for use on the `__device__` (the gpu),
  ## to be called by either `device` or `global` functions.
  when HippoRuntime != "SIMPLE":
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
  when HippoRuntime != "SIMPLE":
    let globalPragma: NimNode = quote:
      {. exportc, codegenDecl: "__host__ $# $#$#".}

    fn.addPragma(globalPragma[0])
    fn.addPragma(globalPragma[1])
  quote do:
    {.push stackTrace: off, checks: off.}
    `fn`
    {.pop.}

macro hippoHostDevice*(fn: untyped): untyped =
  ## Declare a function as both `__host__` and `__device__`.
  ## This is useful for functions that are usable from either the host and the device.
  ## eg: `proc add(a: int, b: int) {.hippoHostDevice.} = a + b`
  when HippoRuntime != "SIMPLE":
    let globalPragma: NimNode = quote:
      {. exportc, codegenDecl: "__device__ __host__ $# $#$#".}

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
  when HippoRuntime != "SIMPLE":
    quote do:
      {.push stackTrace: off, checks: off, noinit, exportc, codegenDecl: "__shared__ $# $#".}
      `v`
      {.pop.}
  else:
    # TODO proper thread vars
    quote do:
      `v`

macro hippoConstant*(v: untyped): untyped =
  ## Declared a variable as `__constant__`.
  ## Constants are read-only globals that are cached on-chip.
  ## constants are useful for data that is being read by all threads in a warp at the same time.
  ## if each thread in a warp accesses different addresses in constant memory,
  ## the accesses are serialized and this may cause a 16x slowdown.
  ## eg: `const N {.hippoConstant.} = 1024`
  when HippoRuntime != "SIMPLE":
    quote do:
      {.push stackTrace: off, checks: off, exportc, codegenDecl: "__constant__ $# $#".}
      `v`
      {.pop.}
  else:
    # TODO proper const vars
    quote do:
      `v`

macro hippoArgs*(args: varargs[untyped]): untyped =
  ## Automatically convert varargs for use with CUDA/HIP.
  ## CUDA/HIP expects an array of arguments or pointers depending on platform.
  when (HippoRuntime == "HIP" and HipPlatform == "nvidia") or HippoRuntime == "HIP_CPU" or HippoRuntime == "SIMPLE":
    # Create a tuple constructor with original arguments
    var tupleNode = newNimNode(nnkTupleConstr)
    for arg in args:
      tupleNode.add(arg)
    result = tupleNode
  else:
    var seqNode = newNimNode(nnkBracket)
    for arg in args:
      seqNode.add(
        quote do:
          cast[ptr pointer](addr `arg`)
      )
    result = quote do:
      @`seqNode`

# -------------------
# Hippo Math Functions
# Runtime-agnostic math functions that work on device

template hippoExp*(x: cfloat): cfloat =
  ## Exponential function (e^x) for single-precision float
  when HippoRuntime == "CUDA":
    expf(x)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cfloat(math.exp(float(x)))
  else: # HIP, HIP_CPU
    expf(x)

template hippoExp*(x: cdouble): cdouble =
  ## Exponential function (e^x) for double-precision float
  when HippoRuntime == "CUDA":
    exp(x)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cdouble(math.exp(float(x)))
  else: # HIP, HIP_CPU
    exp(x)

template hippoLog*(x: cfloat): cfloat =
  ## Natural logarithm for single-precision float
  when HippoRuntime == "CUDA":
    logf(x)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cfloat(math.ln(float(x)))
  else: # HIP, HIP_CPU
    logf(x)

template hippoLog*(x: cdouble): cdouble =
  ## Natural logarithm for double-precision float
  when HippoRuntime == "CUDA":
    log(x)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cdouble(math.ln(float(x)))
  else: # HIP, HIP_CPU
    log(x)

template hippoSin*(x: cfloat): cfloat =
  ## Sine function for single-precision float
  when HippoRuntime == "CUDA":
    sinf(x)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cfloat(math.sin(float(x)))
  else: # HIP, HIP_CPU
    sinf(x)

template hippoCos*(x: cfloat): cfloat =
  ## Cosine function for single-precision float
  when HippoRuntime == "CUDA":
    cosf(x)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cfloat(math.cos(float(x)))
  else: # HIP, HIP_CPU
    cosf(x)

template hippoSqrt*(x: cfloat): cfloat =
  ## Square root function for single-precision float
  when HippoRuntime == "CUDA":
    sqrtf(x)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cfloat(math.sqrt(float(x)))
  else: # HIP, HIP_CPU
    sqrtf(x)

template hippoPow*(base: cfloat, exp: cfloat): cfloat =
  ## Power function (base^exp) for single-precision float
  when HippoRuntime == "CUDA":
    powf(base, exp)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cfloat(math.pow(float(base), float(exp)))
  else: # HIP, HIP_CPU
    powf(base, exp)

template hippoPow*(base: cdouble, exp: cdouble): cdouble =
  ## Power function (base^exp) for double-precision float
  when HippoRuntime == "CUDA":
    pow(base, exp)
  elif HippoRuntime == "SIMPLE":
    # For SIMPLE backend, use Nim's math functions
    cdouble(math.pow(float(base), float(exp)))
  else: # HIP, HIP_CPU
    pow(base, exp)
