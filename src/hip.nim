# HIP runtime C++ FFI
import std/strformat, macros

type
  size_t* = uint64
  uint8_t* = uint8
  uint16_t* = uint16
  uint32_t* = uint32
  uint64_t* = uint64
  hipStream_t* {.importcpp: "hipStream_t", header: "hip/hip_runtime.h".} = pointer
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
proc hipMemcpyToSymbol*(symbol: pointer; src: pointer; sizeBytes: csize_t;
                        offset: csize_t = 0;
                        kind: hipMemcpyKind = hipMemcpyHostToDevice): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipMemcpyToSymbol(@)".}
proc hipSymbol*[T](sym: var T): pointer {.header: "hip/hip_runtime.h", importcpp: "HIP_SYMBOL(@)".}
proc hipFree*(`ptr`: pointer): hipError_t {.header: "hip/hip_runtime.h",importcpp: "hipFree(@)".}

proc hipLaunchKernel*(function_address: pointer; numBlocks: Dim3; dimBlocks: Dim3;
                     args: ptr pointer): hipError_t {.
    importcpp: "hipLaunchKernel(@)", header: "hip/hip_runtime.h".}
proc hipLaunchKernel*(function_address: pointer; numBlocks: Dim3; dimBlocks: Dim3;
                     args: ptr pointer; sharedMemBytes: csize_t; stream: hipStream_t): cint {.
    importcpp: "hipLaunchKernel(@)", header: "hip/hip_runtime.h".}
proc hipDeviceSynchronize*(): hipError_t {.header: "hip/hip_runtime.h",importcpp: "hipDeviceSynchronize(@)".}
proc hipSyncthreads*() {.importcpp: "__syncthreads()", header: "hip/hip_runtime.h".}
proc hippoSyncthreads*() {.importcpp: "__syncthreads()", header: "hip/hip_runtime.h".}

# Atomics
proc hipAtomicAdd*(address: ptr int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicAdd(@)".}
proc hipAtomicAdd*(address: ptr uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicAdd(@)".}
proc hipAtomicSub*(address: ptr int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicSub(@)".}
proc hipAtomicSub*(address: ptr uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicSub(@)".}
proc hipAtomicExch*(address: ptr int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicExch(@)".}
proc hipAtomicExch*(address: ptr uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicExch(@)".}
proc hipAtomicCAS*(address: ptr int32; compare: int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicCAS(@)".}
proc hipAtomicCAS*(address: ptr uint32; compare: uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicCAS(@)".}
proc hipAtomicMin*(address: ptr int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicMin(@)".}
proc hipAtomicMin*(address: ptr uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicMin(@)".}
proc hipAtomicMax*(address: ptr int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicMax(@)".}
proc hipAtomicMax*(address: ptr uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicMax(@)".}
proc hipAtomicAnd*(address: ptr int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicAnd(@)".}
proc hipAtomicAnd*(address: ptr uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicAnd(@)".}
proc hipAtomicOr*(address: ptr int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicOr(@)".}
proc hipAtomicOr*(address: ptr uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicOr(@)".}
proc hipAtomicXor*(address: ptr int32; val: int32): int32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicXor(@)".}
proc hipAtomicXor*(address: ptr uint32; val: uint32): uint32 {.
  header: "hip/hip_runtime.h", importcpp: "atomicXor(@)".}

proc hipLaunchKernelGGL*(
  function_address: proc;
  numBlocks: Dim3;
  dimBlocks: Dim3;
  sharedMemBytes: uint32_t;
  stream: hipStream_t;
  ) {.
    importcpp: "hipLaunchKernelGGL(@)", header: "hip/hip_runtime.h", varargs.}

# Stream Management
proc hipStreamCreate*(stream: ptr hipStream_t): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipStreamCreate(@)".}
proc hipStreamDestroy*(stream: hipStream_t): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipStreamDestroy(@)".}
proc hipStreamSynchronize*(stream: hipStream_t): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipStreamSynchronize(@)".}

# Async Memory Operations
proc hipMemcpyAsync*(dst: pointer, src: pointer, sizeBytes: csize_t,
                     kind: hipMemcpyKind, stream: hipStream_t): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipMemcpyAsync(@)".}

# Page-locked Host Memory
when defined(HIP_CPU_RUNTIME):
  # HIP_CPU doesn't implement hipHostAlloc, use hipMalloc as fallback
  proc hipHostAlloc*(p: ptr pointer; size: csize_t;
                     flags: uint32_t): hipError_t =
    # Note: ignores flags for now, just use regular malloc
    hipMalloc(p, size.cint)
else:
  # Real HIP runtime has hipHostAlloc
  proc hipHostAlloc*(p: ptr pointer; size: csize_t;
                     flags: uint32_t): hipError_t {.
    header: "hip/hip_runtime.h", importcpp: "hipHostAlloc(@)".}

proc hipHostFree*(p: pointer): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipHostFree(@)".}

# Events for Timing
type hipEvent_t* {.importcpp: "hipEvent_t", header: "hip/hip_runtime.h".} = pointer
const
  HippoEventDefault* = 0'u32
  HippoEventBlockingSync* = 1'u32
  HippoEventDisableTiming* = 2'u32
  HippoEventInterprocess* = 4'u32
  HippoErrorNotReady* = 600

proc hipEventCreate*(event: ptr hipEvent_t): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipEventCreate(@)".}
when defined(HIP_CPU_RUNTIME):
  proc hipEventCreateWithFlags*(event: ptr hipEvent_t; flags: uint32_t): hipError_t =
    ## HIP-CPU currently exposes this API but does not use event flags in practice.
    discard flags
    hipEventCreate(event)
else:
  proc hipEventCreateWithFlags*(event: ptr hipEvent_t; flags: uint32_t): hipError_t {.
    header: "hip/hip_runtime.h", importcpp: "hipEventCreateWithFlags(@)".}
proc hipEventDestroy*(event: hipEvent_t): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipEventDestroy(@)".}
proc hipEventRecord*(event: hipEvent_t, stream: hipStream_t = nil): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipEventRecord(@)".}
proc hipEventSynchronize*(event: hipEvent_t): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipEventSynchronize(@)".}
when defined(HIP_CPU_RUNTIME):
  proc hipEventQuery*(event: hipEvent_t): hipError_t =
    ## HIP-CPU headers in this repo do not declare hipEventQuery.
    ## Emulate readiness checks by synchronizing and returning success.
    hipEventSynchronize(event)
else:
  proc hipEventQuery*(event: hipEvent_t): hipError_t {.
    header: "hip/hip_runtime.h", importcpp: "hipEventQuery(@)".}
proc hipEventElapsedTime*(ms: ptr cfloat, start: hipEvent_t, stop: hipEvent_t): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipEventElapsedTime(@)".}
when defined(HIP_CPU_RUNTIME):
  proc hipStreamWaitEvent*(stream: hipStream_t; event: hipEvent_t;
                           flags: uint32_t = 0'u32): hipError_t =
    ## HIP-CPU does not currently expose asynchronous stream wait semantics.
    ## We emulate wait behavior with a direct event synchronize call.
    discard stream
    discard flags
    hipEventSynchronize(event)
else:
  proc hipStreamWaitEvent*(stream: hipStream_t; event: hipEvent_t;
                           flags: uint32_t = 0'u32): hipError_t {.
    header: "hip/hip_runtime.h", importcpp: "hipStreamWaitEvent(@)".}

# Device Properties
type hipDeviceProp_t* {.importcpp: "hipDeviceProp_t", header: "hip/hip_runtime.h".} = object
  deviceOverlap*: cint
proc hipGetDevice*(device: ptr cint): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipGetDevice(@)".}
proc hipGetDeviceProperties*(prop: ptr hipDeviceProp_t; device: cint): hipError_t {.
  header: "hip/hip_runtime.h", importcpp: "hipGetDeviceProperties(@)".}

type ConstCString* {.importc: "const char*".} = object
converter toCString*(self: ConstCString): cstring {.importc: "(char*)", noconv, nodecl.}
converter toConstCString*(self: cstring): ConstCString {.importc: "(const char*)", noconv, nodecl.}
proc `$`*(self: ConstCString): string = $(self.toCString())
proc hipGetErrorString*(err: hipError_t): ConstCString {.header: "hip/hip_runtime.h",importcpp: "hipGetErrorString(@)".}
proc hipGetLastError*(): hipError_t {.header: "hip/hip_runtime.h",importcpp: "hipGetLastError()".}

# Error Helpers
proc handleError*(err: hipError_t) =
  if err != 0:
    var cstr = hipGetErrorString(err).toCString
    raise newException(Exception, &"HIP Error: " & $cstr)

# Hippo Types
type HippoStream* = hipStream_t
type HippoError* = hipError_t
type HippoMemcpyKind* = hipMemcpyKind

const
  HippoMemcpyHostToHost* = hipMemcpyHostToHost
  HippoMemcpyHostToDevice* = hipMemcpyHostToDevice
  HippoMemcpyDeviceToHost* = hipMemcpyDeviceToHost
  HippoMemcpyDeviceToDevice* = hipMemcpyDeviceToDevice
  HippoMemcpyDefault* = hipMemcpyDefault

# HIP Attributes
let
  blockDim* {.importc, inject, header: "hip/hip_runtime.h".}: BlockDim
  blockIdx* {.importc, inject, header: "hip/hip_runtime.h".}: BlockIdx
  gridDim* {.importc, inject, header: "hip/hip_runtime.h".}: GridDim
  threadIdx* {.importc, inject, header: "hip/hip_runtime.h".}: ThreadIdx

# HIP Math Functions
# Single-precision floating-point math functions available in device code
proc expf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "expf(@)".}
proc logf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "logf(@)".}
proc sinf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "sinf(@)".}
proc cosf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "cosf(@)".}
proc sqrtf*(x: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "sqrtf(@)".}
proc powf*(base: cfloat, exp: cfloat): cfloat {.header: "hip/hip_runtime.h", importcpp: "powf(@)".}

# Half-precision (float16) conversion intrinsics
proc halfToFloat*(h: uint16): cfloat {.header: "hip/hip_fp16.h",
    importcpp: "[&]{ __half_raw r; r.x = (#); return __half2float(r); }()".}
  ## Convert IEEE 754 half-precision (uint16) to float32 using HIP hardware intrinsic.
proc floatToHalf*(f: cfloat): uint16 {.header: "hip/hip_fp16.h",
    importcpp: "__half_raw(__float2half(#)).x".}
  ## Convert float32 to IEEE 754 half-precision (uint16) using HIP hardware intrinsic.

# Warp shuffle intrinsics
proc shflDown*(val: cfloat, delta: cint): cfloat {.header: "hip/hip_runtime.h",
    importcpp: "__shfl_down(@)".}
  ## Warp shuffle down for float32.
proc shflDown*(val: cint, delta: cint): cint {.header: "hip/hip_runtime.h",
    importcpp: "__shfl_down(@)".}
  ## Warp shuffle down for int32.
proc shfl*(val: cfloat, srcLane: cint): cfloat {.header: "hip/hip_runtime.h",
    importcpp: "__shfl(@)".}
  ## Warp shuffle: read float32 from srcLane (broadcast).
proc shfl*(val: cint, srcLane: cint): cint {.header: "hip/hip_runtime.h",
    importcpp: "__shfl(@)".}
  ## Warp shuffle: read int32 from srcLane (broadcast).

const WarpSize* {.intdefine.} = 32
  ## AMD wavefront size. Defaults to 32 (RDNA 3+).
  ## Set -d:WarpSize=64 for GCN/CDNA GPUs (e.g. MI250, MI300) which use wave64.

# Double-precision floating-point math functions
proc exp*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "exp(@)".}
proc log*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "log(@)".}
proc sin*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "sin(@)".}
proc cos*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "cos(@)".}
proc sqrt*(x: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "sqrt(@)".}
proc pow*(base: cdouble, exp: cdouble): cdouble {.header: "hip/hip_runtime.h", importcpp: "pow(@)".}
