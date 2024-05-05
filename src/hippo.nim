## HIP Library for Nim

import
  std/[strformat],
  ./hippoTypes

export hippoTypes

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

## Kernel Execution
proc launchKernel*(
  kernel: proc,
  gridDim: Dim3 = newDim3(1, 1, 1),
  blockDim: Dim3 = newDim3(1, 1, 1),
  args: tuple
): hipError_t =
  # launchKernel is designed to be similar to `kernel`<<<blockDim, gridDim>>>(args)
  # which is syncronous and blocks until the kernel is finished
  var kernelArgs: seq[pointer]
  for key, arg in args.fieldPairs:
    kernelArgs.add(cast[pointer](addr arg))

  result = hipLaunchKernel(
    cast[pointer](kernel),
    gridDim,
    blockDim,
    cast[ptr pointer](addr kernelArgs[0]),
  )
  if result != 0:
    return result
  # Automatically syncronize after running the kernel
  return hipDeviceSynchronize()
