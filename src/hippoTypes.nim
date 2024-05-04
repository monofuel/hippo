
## Import the hip_runtime.h header, required for all HIP code
when defined(hippo_runtime):
  # hippo_runtime can be set to "HIP" or "CUDA"
  when hippo_runtime == "HIP":
    {.emit: """
#include "hip/hip_runtime.h"
"""}
    echo "Using HIP runtime"
  elif hippo_runtime == "CUDA":
    # nvcc loads the CUDA runtime automatically
    echo "Using CUDA runtime"
else:
  # Default to HIP
  {.passC: "-I/opt/rocm/include".}
  # {.header: "hip/hip_runtime.h".}
  {.emit: """
#include "hip/hip_runtime.h"
"""}
  echo "Using HIP runtime"

type
  hipMemcpyKind* {.size: sizeof(cint), header: "hip/hip_runtime.h", importcpp: "hipMemcpyKind".} = enum
    hipMemcpyHostToHost = 0,    ## < Host-to-Host Copy
    hipMemcpyHostToDevice = 1,  ## < Host-to-Device Copy
    hipMemcpyDeviceToHost = 2,  ## < Device-to-Host Copy
    hipMemcpyDeviceToDevice = 3, ## < Device-to-Device Copy
    hipMemcpyDefault = 4        ## < Runtime will automatically determine copy-kind based on virtual addresses.

proc hipMalloc*(`ptr`: ptr pointer; size: cint): cint {.header: "hip/hip_runtime.h",importcpp: "hipMalloc(@)".}
proc hipMemcpy*(dst: pointer; src: pointer; size: cint; kind: hipMemcpyKind): cint {.header: "hip/hip_runtime.h",importcpp: "hipMemcpy(@)".}
proc hipFree*(`ptr`: pointer): cint {.header: "hip/hip_runtime.h",importcpp: "hipFree(@)".}
