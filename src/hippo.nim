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

# proc add(a,b: int; c: ptr[int]): {.hippoGlobal.} =
#   c[] = a + b