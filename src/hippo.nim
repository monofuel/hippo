## HIP Library for Nim

import
  std/[macros],
  ./hippoTypes

export hippoTypes


## HIP Attributes
## TODO


# proc add(a,b: int; c: ptr[int]): {.hippoGlobal.} =
#   c[] = a + b