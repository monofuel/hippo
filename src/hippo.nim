## HIP Library for Nim

import
  std/[strformat],
  ./hippoTypes

export hippoTypes

## HIP Attributes
template hippoGlobal*(body: typed) =
  {.push stackTrace: off, checks: off, exportc, codegenDecl: "__global__ $# $#$#".}
  body
  {.pop}

template hippoDevice*(body: typed) =
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
  var kernelArgs: seq[pointer]
  for key, arg in args.fieldPairs:
    kernelArgs.add(cast[pointer](addr arg))

  hipLaunchKernel(
    cast[pointer](kernel),
    gridDim,
    blockDim,
    cast[ptr pointer](addr kernelArgs[0]),
  )

## Error Helpers
proc handleError*(err: hipError_t) =
  if err != 0:
    var cstr = hipGetErrorString(err).toCString
    raise newException(Exception, &"HIP Error: " & $cstr)