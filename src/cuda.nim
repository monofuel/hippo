# CUDA runtime C++ FFI
import std/strformat, macros

type
  size_t* = uint64
  uint8_t* = uint8
  uint16_t* = uint16
  uint32_t* = uint32
  uint64_t* = uint64
  cudaStream_t* {.importcpp: "cudaStream_t", header: "cuda_runtime.h".} = pointer
  cudaError_t* {.importcpp: "cudaError_t", header: "cuda_runtime.h".} = cint

type
  cudaMemcpyKind* {.size: sizeof(cint), header: "cuda_runtime.h", importcpp: "cudaMemcpyKind".} = enum
    cudaMemcpyHostToHost = 0,    ## < Host-to-Host Copy
    cudaMemcpyHostToDevice = 1,  ## < Host-to-Device Copy
    cudaMemcpyDeviceToHost = 2,  ## < Device-to-Host Copy
    cudaMemcpyDeviceToDevice = 3, ## < Device-to-Device Copy
    cudaMemcpyDefault = 4        ## < Runtime will automatically determine copy-kind based on virtual addresses.

type
  Dim3* {.importcpp: "dim3", header: "cuda_runtime.h", bycopy.} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  BlockDim* {.importcpp: "const __cuda_Coordinates<__cuda_BlockDim>", header: "cuda_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  BlockIdx* {.importcpp: "const __cuda_Coordinates<__cuda_BlockIdx>", header: "cuda_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  GridDim* {.importcpp: "const __cuda_Coordinates<__cuda_GridDim>", header: "cuda_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z
  ThreadIdx* {.importcpp: "const __cuda_Coordinates<__cuda_ThreadIdx>", header: "cuda_runtime.h".} = object
    x* {.importc: "x".}: uint32_t ## < x
    y* {.importc: "y".}: uint32_t ## < y
    z* {.importc: "z".}: uint32_t ## < z

proc newDim3*(x: uint32_t = 1; y: uint32_t = 1; z: uint32_t = 1): Dim3 =
  result.x = x
  result.y = y
  result.z = z

proc cudaMalloc*(`ptr`: ptr pointer; size: cint): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaMalloc(@)".}
proc cudaMemcpy*(dst: pointer; src: pointer; size: cint; kind: cudaMemcpyKind): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaMemcpy(@)".}
proc cudaMemcpyToSymbol*(symbol: pointer; src: pointer; sizeBytes: csize_t;
                         offset: csize_t = 0;
                         kind: cudaMemcpyKind = cudaMemcpyHostToDevice): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaMemcpyToSymbol(@)".}
proc cudaFree*(`ptr`: pointer): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaFree(@)".}

proc cudaLaunchKernel*(function_address: pointer; numBlocks: Dim3; dimBlocks: Dim3;
                     args: ptr pointer): cudaError_t {.
    importcpp: "cudaLaunchKernel(@)", header: "cuda_runtime.h".}
proc cudaLaunchKernel*(function_address: pointer; numBlocks: Dim3; dimBlocks: Dim3;
                     args: ptr pointer, sharedMemBytes: uint32_t, stream: cudaStream_t): cudaError_t {.
    importcpp: "cudaLaunchKernel(@)", header: "cuda_runtime.h".}
proc cudaDeviceSynchronize*(): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaDeviceSynchronize(@)".}
proc cudaSyncthreads*() {.importcpp: "__syncthreads()", header: "cuda_runtime.h".}
proc hippoSyncthreads*() {.importcpp: "__syncthreads()", header: "cuda_runtime.h".}

# Atomics
proc cudaAtomicAdd*(address: ptr int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicAdd(@)".}
proc cudaAtomicAdd*(address: ptr uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicAdd(@)".}
proc cudaAtomicSub*(address: ptr int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicSub(@)".}
proc cudaAtomicSub*(address: ptr uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicSub(@)".}
proc cudaAtomicExch*(address: ptr int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicExch(@)".}
proc cudaAtomicExch*(address: ptr uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicExch(@)".}
proc cudaAtomicCAS*(address: ptr int32; compare: int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicCAS(@)".}
proc cudaAtomicCAS*(address: ptr uint32; compare: uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicCAS(@)".}
proc cudaAtomicMin*(address: ptr int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicMin(@)".}
proc cudaAtomicMin*(address: ptr uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicMin(@)".}
proc cudaAtomicMax*(address: ptr int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicMax(@)".}
proc cudaAtomicMax*(address: ptr uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicMax(@)".}
proc cudaAtomicAnd*(address: ptr int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicAnd(@)".}
proc cudaAtomicAnd*(address: ptr uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicAnd(@)".}
proc cudaAtomicOr*(address: ptr int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicOr(@)".}
proc cudaAtomicOr*(address: ptr uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicOr(@)".}
proc cudaAtomicXor*(address: ptr int32; val: int32): int32 {.
  header: "cuda_runtime.h", importcpp: "atomicXor(@)".}
proc cudaAtomicXor*(address: ptr uint32; val: uint32): uint32 {.
  header: "cuda_runtime.h", importcpp: "atomicXor(@)".}

proc cudaLaunchKernelGGL*(
  function_address: proc;
  numBlocks: Dim3;
  dimBlocks: Dim3;
  sharedMemBytes: uint32_t;
  stream: cudaStream_t;
  ) {.
    importcpp: "cudaLaunchKernelGGL(@)", header: "cuda_runtime.h", varargs.}

# Stream Management
proc cudaStreamCreate*(stream: ptr cudaStream_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaStreamCreate(@)".}
proc cudaStreamDestroy*(stream: cudaStream_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaStreamDestroy(@)".}
proc cudaStreamSynchronize*(stream: cudaStream_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaStreamSynchronize(@)".}

# Async Memory Operations
proc cudaMemcpyAsync*(dst: pointer; src: pointer; size: csize_t;
                      kind: cudaMemcpyKind; stream: cudaStream_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaMemcpyAsync(@)".}

# Page-locked Host Memory
proc cudaHostAlloc*(p: ptr pointer; size: csize_t;
                    flags: uint32_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaHostAlloc(@)".}
proc cudaFreeHost*(p: pointer): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaFreeHost(@)".}

# Events for Timing
type cudaEvent_t* {.importcpp: "cudaEvent_t", header: "cuda_runtime.h".} = pointer
const
  HippoEventDefault* = 0'u32
  HippoEventBlockingSync* = 1'u32
  HippoEventDisableTiming* = 2'u32
  HippoEventInterprocess* = 4'u32
  HippoErrorNotReady* = 600
proc cudaEventCreate*(event: ptr cudaEvent_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaEventCreate(@)".}
proc cudaEventCreateWithFlags*(event: ptr cudaEvent_t; flags: uint32_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaEventCreateWithFlags(@)".}
proc cudaEventDestroy*(event: cudaEvent_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaEventDestroy(@)".}
proc cudaEventRecord*(event: cudaEvent_t; stream: cudaStream_t = nil): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaEventRecord(@)".}
proc cudaEventQuery*(event: cudaEvent_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaEventQuery(@)".}
proc cudaEventSynchronize*(event: cudaEvent_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaEventSynchronize(@)".}
proc cudaEventElapsedTime*(ms: ptr cfloat; start: cudaEvent_t; stop: cudaEvent_t): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaEventElapsedTime(@)".}
proc cudaStreamWaitEvent*(stream: cudaStream_t; event: cudaEvent_t;
                          flags: uint32_t = 0'u32): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaStreamWaitEvent(@)".}

# Device Properties
type cudaDeviceProp* {.importcpp: "cudaDeviceProp", header: "cuda_runtime.h".} = object
  deviceOverlap*: cint
proc cudaGetDevice*(device: ptr cint): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaGetDevice(@)".}
proc cudaGetDeviceProperties*(prop: ptr cudaDeviceProp; device: cint): cudaError_t {.
  header: "cuda_runtime.h", importcpp: "cudaGetDeviceProperties(@)".}


type ConstCString* {.importc: "const char*".} = object
converter toCString*(self: ConstCString): cstring {.importc: "(char*)", noconv, nodecl.}
converter toConstCString*(self: cstring): ConstCString {.importc: "(const char*)", noconv, nodecl.}
proc `$`*(self: ConstCString): string = $(self.toCString())
proc cudaGetErrorString*(err: cudaError_t): ConstCString {.header: "cuda_runtime.h",importcpp: "cudaGetErrorString(@)".}
proc cudaGetLastError*(): cudaError_t {.header: "cuda_runtime.h",importcpp: "cudaGetLastError()".}

# Error Helpers
proc handleError*(err: cudaError_t) =
  if err != 0:
    var cstr = cudaGetErrorString(err).toCString
    raise newException(Exception, &"CUDA Error: " & $cstr)

# Hippo Types
type HippoStream* = cudaStream_t
type HippoError* = cudaError_t
type HippoMemcpyKind* = cudaMemcpyKind

const
  HippoMemcpyHostToHost* = cudaMemcpyHostToHost
  HippoMemcpyHostToDevice* = cudaMemcpyHostToDevice
  HippoMemcpyDeviceToHost* = cudaMemcpyDeviceToHost
  HippoMemcpyDeviceToDevice* = cudaMemcpyDeviceToDevice
  HippoMemcpyDefault* = cudaMemcpyDefault


# CUDA Attributes
let
  blockDim* {.importc, inject, header: "cuda_runtime.h".}: BlockDim
  blockIdx* {.importc, inject, header: "cuda_runtime.h".}: BlockIdx
  gridDim* {.importc, inject, header: "cuda_runtime.h".}: GridDim
  threadIdx* {.importc, inject, header: "cuda_runtime.h".}: ThreadIdx

# CUDA Math Functions
# Single-precision floating-point math functions available in device code
proc expf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "expf(@)".}
proc logf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "logf(@)".}
proc sinf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "sinf(@)".}
proc cosf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "cosf(@)".}
proc sqrtf*(x: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "sqrtf(@)".}
proc powf*(base: cfloat, exp: cfloat): cfloat {.header: "cuda_runtime.h", importcpp: "powf(@)".}

# Half-precision (float16) conversion intrinsics
proc halfToFloat*(h: uint16): cfloat {.header: "cuda_fp16.h",
    importcpp: "[&]{ __half_raw r; r.x = (#); return __half2float(r); }()".}
  ## Convert IEEE 754 half-precision (uint16) to float32 using CUDA hardware intrinsic.
proc floatToHalf*(f: cfloat): uint16 {.header: "cuda_fp16.h",
    importcpp: "__half_raw(__float2half(#)).x".}
  ## Convert float32 to IEEE 754 half-precision (uint16) using CUDA hardware intrinsic.

# Double-precision floating-point math functions
proc exp*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "exp(@)".}
proc log*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "log(@)".}
proc sin*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "sin(@)".}
proc cos*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "cos(@)".}
proc sqrt*(x: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "sqrt(@)".}
proc pow*(base: cdouble, exp: cdouble): cdouble {.header: "cuda_runtime.h", importcpp: "pow(@)".}
