## HIP Library for Nim
import
  std/[strformat]

# HippoRuntime can be set to "HIP", "HIP_CPU", or "CUDA"
# HIP hipcc will auto detect the runtime of the build system
# HIP_CPU will use the HIP-CPU runtime header
# CUDA will use nvcc

const HippoRuntime* {.strdefine.} = "HIP"

when HippoRuntime == "HIP_CPU":
  {.passL: "-ltbb".}
  {.passL: "-lstdc++".}
  {.passC: "-I./HIP-CPU/include/".}
  # {.compile: "../HIP-CPU/include/hip/hip_runtime.h".}
  echo "Using HIP CPU runtime"
  include hip
elif HippoRuntime == "CUDA":
  # nvcc loads the CUDA runtime automatically
  # Note: i have not actually setup any CUDA stuff yet
  echo "Using CUDA runtime"
  include cuda
else:
  echo "Using HIP runtime"
  include hip

## Error Helpers
proc handleError*(err: hipError_t) =
  if err != 0:
    var cstr = hipGetErrorString(err).toCString
    raise newException(Exception, &"HIP Error: " & $cstr)

## HIP Attributes
template hippoGlobal*(body: untyped) =
  var
    blockDim {.importc, inject, header: "hip/hip_runtime.h".}: BlockDim
    blockIdx {.importc, inject, header: "hip/hip_runtime.h".}: BlockIdx
    gridDim {.importc, inject, header: "hip/hip_runtime.h".}: GridDim
    threadIdx {.importc, inject, header: "hip/hip_runtime.h".}: ThreadIdx
  {.push stackTrace: off, checks: off, exportc, codegenDecl: "__global__ $# $#$#".}
  body
  {.pop}

template hippoDevice*(body: typed) =
  var
    blockDim {.importc, inject, header: "hip/hip_runtime.h".}: BlockDim
    blockIdx {.importc, inject, header: "hip/hip_runtime.h".}: BlockIdx
    gridDim {.importc, inject, header: "hip/hip_runtime.h".}: GridDim
    threadIdx {.importc, inject, header: "hip/hip_runtime.h".}: ThreadIdx
  {.push stackTrace: off, checks: off, exportc, codegenDecl: "__device__ $# $#$#".}
  body
  {.pop}

template hippoHost*(body: typed) =
  {.push stackTrace: off, checks: off, exportc, codegenDecl: "__host__ $# $#$#".}
  body
  {.pop}

# Kernel Execution

proc launchKernel*(
  kernel: proc,
  gridDim: Dim3 = newDim3(1,1,1), # default to a grid of 1 block
  blockDim: Dim3 = newDim3(1,1,1),  # default to 1 thread per block
  sharedMemBytes: uint32 = 0,
  stream: hipStream_t = nil,
  args: tuple
): hipError_t =
  # launchKernel is designed to be similar to `kernel`<<<blockDim, gridDim>>>(args)

  # having some issues between hip and hip-cpu, so defining different versions of launchKernel
  when HippoRuntime == "HIP":
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
  elif HippoRuntime == "HIP_CPU":
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
    result = hipDeviceSynchronize()
  else:
    raise newException(Exception, &"Unknown runtime: {HippoRuntime}")
  if result != 0:
    return result