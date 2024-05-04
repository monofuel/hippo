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
  var args = (addr args[0], addr args[1], addr args[2])
  hipLaunchKernel(
    cast[pointer](kernel),
    gridDim,
    blockDim,
    cast[ptr pointer](addr args)
  )