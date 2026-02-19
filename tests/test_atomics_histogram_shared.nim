import
  std/unittest,
  hippo,
  ./utils

const
  NumBins = 256
  NumElements = 256 * 1024
  ThreadsPerBlock = 256
  BlocksPerGrid = min(64, (NumElements + ThreadsPerBlock - 1) div ThreadsPerBlock)

proc histogramSharedKernel(buffer: ptr[uint8], size: int32, histo: ptr[uint32]) {.hippoGlobal.} =
  ## Build a histogram using shared-memory accumulation plus global atomic merge.
  var temp {.hippoShared.}: array[NumBins, uint32]
  let tid = int(threadIdx.x)
  let buf = cast[ptr UncheckedArray[uint8]](buffer)
  let hist = cast[ptr UncheckedArray[uint32]](histo)

  temp[tid] = 0'u32
  hippoSyncthreads()

  var i = int(threadIdx.x + blockIdx.x * blockDim.x)
  let stride = int(blockDim.x * gridDim.x)
  while i < int(size):
    let bin = int(buf[i])
    discard hippoAtomicAdd(addr temp[bin], 1'u32)
    i += stride

  hippoSyncthreads()
  discard hippoAtomicAdd(addr hist[tid], temp[tid])

suite "atomics histogram shared":
  testSkipPlatforms "hist_gpu_shared style histogram", "SIMPLE":
    var hostBuffer = newSeq[uint8](NumElements)
    var expectedHist: array[NumBins, uint32]
    var hostHist: array[NumBins, uint32]
    var zeroHist: array[NumBins, uint32]

    for i in 0..<NumElements:
      let value = uint8((i * 17 + i div 7) mod NumBins)
      hostBuffer[i] = value
      expectedHist[int(value)] += 1'u32

    var devBuffer = hippoMalloc(sizeof(uint8) * NumElements)
    var devHist = hippoMalloc(sizeof(uint32) * NumBins)
    var sizeArg = NumElements.int32

    hippoMemcpy(devBuffer, addr hostBuffer[0], sizeof(uint8) * NumElements, HippoMemcpyHostToDevice)
    hippoMemcpy(devHist, addr zeroHist[0], sizeof(uint32) * NumBins, HippoMemcpyHostToDevice)

    hippoLaunchKernel(
      histogramSharedKernel,
      gridDim = newDim3(BlocksPerGrid.uint32),
      blockDim = newDim3(ThreadsPerBlock.uint32),
      args = hippoArgs(devBuffer.p, sizeArg, devHist.p)
    )

    hippoMemcpy(addr hostHist[0], devHist, sizeof(uint32) * NumBins, HippoMemcpyDeviceToHost)

    for i in 0..<NumBins:
      check hostHist[i] == expectedHist[i]
