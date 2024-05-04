# nim cpp --cc:clang --clang.cpp.exe=hipcc --clang.cpp.linkerexe=hipcc call_params_emit.nim

{.emit: """
#include "hip/hip_runtime.h"
"""}


{.emit: """
__global__ void add(int a, int b, int *c) {
    *c = a + b;
}
"""}


type
  hipMemcpyKind* {.size: sizeof(cint), importcpp: "hipMemcpyKind".} = enum
    hipMemcpyHostToHost = 0,    ## < Host-to-Host Copy
    hipMemcpyHostToDevice = 1,  ## < Host-to-Device Copy
    hipMemcpyDeviceToHost = 2,  ## < Device-to-Host Copy
    hipMemcpyDeviceToDevice = 3, ## < Device-to-Device Copy
    hipMemcpyDefault = 4        ## < Runtime will automatically determine copy-kind based on virtual addresses.

proc hipMalloc*(`ptr`: ptr pointer; size: cint): cint {.importcpp: "hipMalloc(@)".}
proc hipMemcpy*(dst: pointer; src: pointer; size: cint; kind: hipMemcpyKind): cint {.importcpp: "hipMemcpy(@)".}
proc hipFree*(`ptr`: pointer): cint {.importcpp: "hipFree(@)".}



proc main() =
  echo "DEBUG: init"
  var c: int32
  var dev_c: ptr[int32]
  discard hipMalloc(cast[ptr pointer](addr dev_c), sizeof(int32).cint)
  echo "DEBUG: hipMalloc"
  {.emit: """
  add<<<1,1>>>(2,7,dev_c);
  """.}
  discard hipMemcpy(addr c, dev_c, sizeof(int32).cint, hipMemcpyDeviceToHost)
  echo "DEBUG: hipMemcpy"
  echo "2 + 7 = ", c
  discard hipFree(dev_c)
  echo "DEBUG: hipFree"
  echo "DEBUG: done"



when isMainModule:
  main()