## HIP Library for Nim

import
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
  gridDim: dim3 = newDim3(1, 1, 1),
  blockDim: dim3 = newDim3(1, 1, 1),
  args: tuple
): cint =
  var kernelArgs: seq[pointer]
  for key, arg in args.fieldPairs:
    kernelArgs.add(cast[pointer](addr arg))

  hipLaunchKernel(
    cast[pointer](kernel),
    gridDim,
    blockDim,
    cast[ptr pointer](addr kernelArgs[0]),
  )