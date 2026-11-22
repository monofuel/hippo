## Shared dot-product kernel and host checks for the Chapter 11 samples.
## Ordinary and mapped paths share the arithmetic, with independent CPU references.
## Device allocations stay within the worker that selected their owning device.

import hippo

when HippoRuntime != "HIP" and HippoRuntime != "HIP_CPU":
  {.error: "Chapter 11 currently targets HIP and HIP-CPU.".}

const
  ThreadsPerBlock* = 64
  MaxBlocks* = 8

proc dotKernel(a, b, partials: ptr cfloat; count: cint) {.hippoGlobal.} =
  ## Reduce strided products within each block, including partial and empty blocks.
  var shared {.hippoShared.}: array[ThreadsPerBlock, cfloat]
  let
    lane = int(threadIdx.x)
    stride = int(blockDim.x * gridDim.x)
    left = cast[ptr UncheckedArray[cfloat]](a)
    right = cast[ptr UncheckedArray[cfloat]](b)
  var
    index = int(blockIdx.x * blockDim.x + threadIdx.x)
    total = 0.0'f32
  while index < int(count):
    total = total + left[index] * right[index]
    index = index + stride
  shared[lane] = total
  hippoSyncthreads()
  var width = ThreadsPerBlock div 2
  while width > 0:
    if lane < width:
      shared[lane] = shared[lane] + shared[lane + width]
    hippoSyncthreads()
    width = width div 2
  if lane == 0:
    cast[ptr UncheckedArray[cfloat]](partials)[int(blockIdx.x)] = shared[0]

proc floatOffset*(data: pointer; offset: int): pointer =
  ## Offset a borrowed float32 pointer by elements, without changing ownership.
  addr cast[ptr UncheckedArray[cfloat]](data)[offset]

proc fillInputs*(a, b: pointer; count: int; seed = 0) =
  ## Use signed binary fractions so references are exact at the tested sizes.
  let
    left = cast[ptr UncheckedArray[cfloat]](a)
    right = cast[ptr UncheckedArray[cfloat]](b)
  for index in 0 ..< count:
    left[index] = cfloat((index * 17 + seed) mod 31 - 15) * 0.125'f32
    right[index] = cfloat((index * 13 + seed) mod 19 - 9) * 0.25'f32

proc referenceDot*(a, b: pointer; count: int): float64 =
  ## Compute a serial double-precision oracle on the host.
  let
    left = cast[ptr UncheckedArray[cfloat]](a)
    right = cast[ptr UncheckedArray[cfloat]](b)
  for index in 0 ..< count:
    result += float64(left[index]) * float64(right[index])

proc launchDot*(a, b, partials: pointer; count: int; stream: HippoStream) =
  ## Use one explicit stream and finish before host mutation or memory cleanup.
  var countArg = cint(count)
  try:
    hippoLaunchKernel(dotKernel, gridDim = newDim3(MaxBlocks),
      blockDim = newDim3(ThreadsPerBlock), stream = stream,
      args = hippoArgs(a, b, partials, countArg))
  finally:
    hippoStreamSynchronize(stream)

proc deviceDot*(a, b: pointer; count: int; stream: HippoStream): float64 =
  ## Reduce inputs already accessible by this device and copy back block sums.
  let partials = hippoMalloc(MaxBlocks * sizeof(cfloat))
  launchDot(a, b, partials.p, count, stream)
  var values: array[MaxBlocks, cfloat]
  hippoMemcpy(addr values[0], partials, sizeof(values), HippoMemcpyDeviceToHost)
  for value in values:
    result += float64(value)

proc copiedDot*(a, b: pointer; count: int; stream: HippoStream): float64 =
  ## Chapter 11.2/11.3: allocate device inputs, copy host data and reduce it.
  let
    capacity = max(count, 1) * sizeof(cfloat)
    deviceA = hippoMalloc(capacity)
    deviceB = hippoMalloc(capacity)
  if count > 0:
    hippoMemcpy(deviceA, a, count * sizeof(cfloat), HippoMemcpyHostToDevice)
    hippoMemcpy(deviceB, b, count * sizeof(cfloat), HippoMemcpyHostToDevice)
  result = deviceDot(deviceA.p, deviceB.p, count, stream)

proc mappedDot*(a, b: pointer; offset, count: int; stream: HippoStream): float64 =
  ## Chapter 11.2/11.4: obtain aliases on this device, then offset to its slice.
  let
    deviceA = floatOffset(hippoHostGetDevicePointer(a), offset)
    deviceB = floatOffset(hippoHostGetDevicePointer(b), offset)
  result = deviceDot(deviceA, deviceB, count, stream)
