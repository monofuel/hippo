import
  std/unittest,
  hippo,
  ./utils

const
  ThreadsPerBlock = 128'u32
  BlocksPerGrid = 4'u32
  TotalThreads = int32(ThreadsPerBlock * BlocksPerGrid)
  IncrementsPerThread = 8'i32

proc atomicAddKernel(counter: ptr[int32]) {.hippoGlobal.} =
  ## Increment one global counter from many threads.
  let tid = threadIdx.x + blockIdx.x * blockDim.x
  if tid < TotalThreads.uint32:
    for _ in 0..<IncrementsPerThread:
      discard hippoAtomicAdd(counter, 1'i32)

proc atomicOpsKernel(signedData: ptr[int32], unsignedData: ptr[uint32]) {.hippoGlobal.} =
  ## Run single-thread deterministic checks for core atomic operations.
  let tid = threadIdx.x + blockIdx.x * blockDim.x
  if tid == 0:
    let sArr = cast[ptr UncheckedArray[int32]](signedData)
    let uArr = cast[ptr UncheckedArray[uint32]](unsignedData)

    discard hippoAtomicSub(addr sArr[0], 3'i32)
    discard hippoAtomicMin(addr sArr[1], 2'i32)
    discard hippoAtomicMax(addr sArr[2], 9'i32)

    discard hippoAtomicAnd(addr uArr[0], 0b1010'u32)
    discard hippoAtomicOr(addr uArr[1], 0b0011'u32)
    discard hippoAtomicXor(addr uArr[2], 0b0101'u32)
    uArr[7] = hippoAtomicCAS(addr uArr[3], 7'u32, 99'u32)
    uArr[8] = hippoAtomicCAS(addr uArr[3], 7'u32, 123'u32)
    sArr[3] = hippoAtomicCAS(addr sArr[4], 6'i32, 42'i32)
    sArr[5] = hippoAtomicExch(addr sArr[4], -7'i32)

suite "atomics":
  testSkipPlatforms "atomic add with contention", "SIMPLE":
    var hostCounter = int32(0)
    var devCounter = hippoMalloc(sizeof(int32))

    hippoMemcpy(devCounter, addr hostCounter, sizeof(int32), HippoMemcpyHostToDevice)
    hippoLaunchKernel(
      atomicAddKernel,
      gridDim = newDim3(BlocksPerGrid),
      blockDim = newDim3(ThreadsPerBlock),
      args = hippoArgs(devCounter.p)
    )
    hippoMemcpy(addr hostCounter, devCounter, sizeof(int32), HippoMemcpyDeviceToHost)

    check hostCounter == TotalThreads * IncrementsPerThread

  testSkipPlatforms "atomic sub min max cas exch and bitwise", "SIMPLE":
    var signedHost = [10'i32, 5'i32, 5'i32, 0'i32, 6'i32, 0'i32]
    var unsignedHost = [12'u32, 5'u32, 15'u32, 7'u32, 0'u32, 0'u32, 0'u32, 0'u32, 0'u32]
    var devSigned = hippoMalloc(sizeof(signedHost))
    var devUnsigned = hippoMalloc(sizeof(unsignedHost))

    hippoMemcpy(devSigned, addr signedHost[0], sizeof(signedHost), HippoMemcpyHostToDevice)
    hippoMemcpy(devUnsigned, addr unsignedHost[0], sizeof(unsignedHost), HippoMemcpyHostToDevice)
    hippoLaunchKernel(
      atomicOpsKernel,
      gridDim = newDim3(1'u32),
      blockDim = newDim3(1'u32),
      args = hippoArgs(devSigned.p, devUnsigned.p)
    )
    hippoMemcpy(addr signedHost[0], devSigned, sizeof(signedHost), HippoMemcpyDeviceToHost)
    hippoMemcpy(addr unsignedHost[0], devUnsigned, sizeof(unsignedHost), HippoMemcpyDeviceToHost)

    check signedHost[0] == 7'i32
    check signedHost[1] == 2'i32
    check signedHost[2] == 9'i32
    check signedHost[3] == 6'i32
    check signedHost[4] == -7'i32
    check signedHost[5] == 42'i32

    check unsignedHost[0] == 8'u32
    check unsignedHost[1] == 7'u32
    check unsignedHost[2] == 10'u32
    check unsignedHost[3] == 99'u32
    check unsignedHost[7] == 7'u32
    check unsignedHost[8] == 99'u32
