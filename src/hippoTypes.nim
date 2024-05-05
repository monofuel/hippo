
# HippoRuntime can be set to "HIP", "HIP_CPU", or "CUDA"
const HippoRuntime* {.strdefine.} = "HIP"

## Import the hip_runtime.h header, required for all HIP code
when HippoRuntime == "HIP":
  {.passC: "-I/opt/rocm/include".}
  {.emit: """
#include "hip/hip_runtime.h"
"""}
  echo "Using HIP runtime"

elif HippoRuntime == "HIP_CPU":
  {.passC: "-I./HIP-CPU/include/".}
  {.passL: "-ltbb".}
  {.passL: "-lstdc++".}
  {.emit: """
#include "hip/hip_runtime.h"
"""}
  echo "Using HIP CPU runtime"

elif HippoRuntime == "CUDA":
  # nvcc loads the CUDA runtime automatically
  # Note: i have not actually setup any CUDA stuff yet
  echo "Using CUDA runtime"


# TODO cuda runtime types

type
  size_t* = uint64
  uint8_t* = uint8
  uint16_t* = uint16
  uint32_t* = uint32
  uint64_t* = uint64
  hipStream_t* = pointer
  hipError_t* {.importcpp: "hipError_t", header: "hip/hip_runtime.h".} = cint

type
  hipMemcpyKind* {.size: sizeof(cint), header: "hip/hip_runtime.h", importcpp: "hipMemcpyKind".} = enum
    hipMemcpyHostToHost = 0,    ## < Host-to-Host Copy
    hipMemcpyHostToDevice = 1,  ## < Host-to-Device Copy
    hipMemcpyDeviceToHost = 2,  ## < Device-to-Host Copy
    hipMemcpyDeviceToDevice = 3, ## < Device-to-Device Copy
    hipMemcpyDefault = 4        ## < Runtime will automatically determine copy-kind based on virtual addresses.

type
  Dim3* {.importcpp: "dim3", header: "hip/hip_runtime.h", bycopy.} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  BlockDim* {.importcpp: "const __HIP_Coordinates<__HIP_BlockDim>", header: "hip/hip_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  BlockIdx* {.importcpp: "const __HIP_Coordinates<__HIP_BlockIdx>", header: "hip/hip_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  GridDim* {.importcpp: "const __HIP_Coordinates<__HIP_GridDim>", header: "hip/hip_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  ThreadIdx* {.importcpp: "const __HIP_Coordinates<__HIP_ThreadIdx>", header: "hip/hip_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z

proc newDim3*(x: uint32_t = 1; y: uint32_t = 1; z: uint32_t = 1): Dim3 =
  result.x = x
  result.y = y
  result.z = z

proc hipMalloc*(`ptr`: ptr pointer; size: cint): hipError_t {.header: "hip/hip_runtime.h",importcpp: "hipMalloc(@)".}
proc hipMemcpy*(dst: pointer; src: pointer; size: cint; kind: hipMemcpyKind): hipError_t {.header: "hip/hip_runtime.h",importcpp: "hipMemcpy(@)".}
proc hipFree*(`ptr`: pointer): hipError_t {.header: "hip/hip_runtime.h",importcpp: "hipFree(@)".}

proc hipLaunchKernel*(function_address: pointer; numBlocks: Dim3; dimBlocks: Dim3;
                     args: ptr pointer): hipError_t {.
    importcpp: "hipLaunchKernel(@)", header: "hip/hip_runtime.h".}
# proc hipLaunchKernel*(function_address: pointer; numBlocks: dim3; dimBlocks: dim3;
#                      args: ptr pointer; sharedMemBytes: csize_t; stream: hipStream_t): cint {.
#     importcpp: "hipLaunchKernel(@)", header: "hip/hip_runtime.h".}
proc hipDeviceSynchronize*(): hipError_t {.header: "hip/hip_runtime.h",importcpp: "hipDeviceSynchronize(@)".}

proc hipLaunchKernelGGL*(
  function_address: proc;
  numBlocks: Dim3;
  dimBlocks: Dim3;
  sharedMemBytes: uint32_t;
  stream: hipStream_t;
  ) {.
    importcpp: "hipLaunchKernelGGL(@)", header: "hip/hip_runtime.h", varargs.}


type ConstCString* {.importc: "const char*".} = object
converter toCString*(self: ConstCString): cstring {.importc: "(char*)", noconv, nodecl.}
converter toConstCString*(self: cstring): ConstCString {.importc: "(const char*)", noconv, nodecl.}
proc `$`*(self: ConstCString): string = $(self.toCString())
proc hipGetErrorString*(err: hipError_t): ConstCString {.header: "hip/hip_runtime.h",importcpp: "hipGetErrorString(@)".}