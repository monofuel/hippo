import
  std/unittest,
  hippo

const
  PairCount = 4096
  ThreadsPerBlock = 256'u32
  BlocksPerGrid = uint32(PairCount div 256)
  BaseSeed = 0x5EED'u32

proc randKernel(outs: ptr uint32) {.hippoGlobal.} =
  ## Hash every (pos, seed) pair on the device with hippoRandUint32.
  let idx = int(threadIdx.x + blockIdx.x * blockDim.x)
  if idx < PairCount:
    let outArr = cast[ptr UncheckedArray[uint32]](outs)
    outArr[idx] = hippoRandUint32(uint32(idx), BaseSeed + uint32(idx))

suite "device rng":
  test "hippoRandUint32 matches between host and device":
    var hostOuts = newSeq[uint32](PairCount)
    let devOuts = hippoMalloc(PairCount * sizeof(uint32))

    hippoLaunchKernel(
      randKernel,
      gridDim = newDim3(BlocksPerGrid),
      blockDim = newDim3(ThreadsPerBlock),
      args = hippoArgs(devOuts.p)
    )
    hippoSynchronize()
    hippoMemcpy(addr hostOuts[0], devOuts, PairCount * sizeof(uint32), HippoMemcpyDeviceToHost)

    var mismatches = 0
    for i in 0 ..< PairCount:
      if hostOuts[i] != hippoRandUint32(uint32(i), BaseSeed + uint32(i)):
        mismatches.inc
    check mismatches == 0
