
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
    # Note: i have not actually setup any CUDA stuff yet
    echo "Using CUDA runtime"
else:
  # Default to HIP
  {.passC: "-I/opt/rocm/include".}
  # {.header: "hip/hip_runtime.h".}
  {.emit: """
#include "hip/hip_runtime.h"
"""}
  echo "Using HIP runtime"

# TODO cuda runtime types

type
  size_t* = uint64
  uint8_t* = uint8
  uint16_t* = uint16
  uint32_t* = uint32
  uint64_t* = uint64
  hipStream_t* = pointer

type
  hipMemcpyKind* {.size: sizeof(cint), header: "hip/hip_runtime.h", importcpp: "hipMemcpyKind".} = enum
    hipMemcpyHostToHost = 0,    ## < Host-to-Host Copy
    hipMemcpyHostToDevice = 1,  ## < Host-to-Device Copy
    hipMemcpyDeviceToHost = 2,  ## < Device-to-Host Copy
    hipMemcpyDeviceToDevice = 3, ## < Device-to-Device Copy
    hipMemcpyDefault = 4        ## < Runtime will automatically determine copy-kind based on virtual addresses.

type
  dim3* {.importcpp: "dim3", header: "hip/hip_runtime.h", bycopy.} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z

proc newDim3*(x: uint32_t; y: uint32_t; z: uint32_t): dim3 =
  result.x = x
  result.y = y
  result.z = z

proc hipMalloc*(`ptr`: ptr pointer; size: cint): cint {.header: "hip/hip_runtime.h",importcpp: "hipMalloc(@)".}
proc hipMemcpy*(dst: pointer; src: pointer; size: cint; kind: hipMemcpyKind): cint {.header: "hip/hip_runtime.h",importcpp: "hipMemcpy(@)".}
proc hipFree*(`ptr`: pointer): cint {.header: "hip/hip_runtime.h",importcpp: "hipFree(@)".}

proc hipLaunchKernel*(function_address: pointer; numBlocks: dim3; dimBlocks: dim3;
                     args: ptr pointer): cint {.
    importcpp: "hipLaunchKernel(@)", header: "hip/hip_runtime.h".}
# proc hipLaunchKernel*(function_address: pointer; numBlocks: dim3; dimBlocks: dim3;
#                      args: ptr pointer; sharedMemBytes: csize_t; stream: hipStream_t): cint {.
#     importcpp: "hipLaunchKernel(@)", header: "hip/hip_runtime.h".}

proc hipLaunchKernelGGL*(function_address: pointer; numBlocks: dim3; dimBlocks: dim3;): cint {.
    importcpp: "hipLaunchKernelGGL(@)", header: "hip/hip_runtime.h", varargs.}