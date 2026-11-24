import
  std/[unittest, strutils],
  hippo

const
  LaneCount = 8
  RoundCount = 5

proc neighbors(output: ptr cint; seed: cint) {.hippoGlobal.} =
  ## Exchange neighboring lanes' values while retaining private accumulators.
  var
    values {.hippoShared.}: array[LaneCount, cint]
    weights {.hippoShared.}: array[LaneCount, cfloat]
    owner {.hippoShared.}: cint
  let
    lane = int(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
    blockId = int(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))
    lanes = int(blockDim.x * blockDim.y * blockDim.z)
    next = (lane + 1) mod lanes
    actual = cast[ptr UncheckedArray[cint]](output)
  var privateTotal = cint(lane)
  for round in 0 ..< RoundCount:
    values[lane] = seed + cint(blockId * 1000 + round * 100 + lane)
    weights[lane] = cfloat(lane) * 0.5'f32
    if lane == 0:
      owner = cint(blockId + 1)
    hippoSyncthreads()
    privateTotal = privateTotal + values[next]
    actual[(blockId * RoundCount + round) * lanes + lane] =
      privateTotal + owner + cint(weights[next] * 2.0'f32)
    hippoSyncthreads()

proc shadowed(output: ptr cint) {.hippoGlobal.} =
  ## Keep identically named shared declarations in distinct lexical scopes.
  var values {.hippoShared.}: array[LaneCount, cint]
  let
    lane = int(threadIdx.x)
    next = (lane + 1) mod LaneCount
    actual = cast[ptr UncheckedArray[cint]](output)
  values[lane] = cint(lane + 10)
  block:
    var values {.hippoShared.}: array[LaneCount, cint]
    values[lane] = cint(lane + 100)
    hippoSyncthreads()
    actual[lane] = values[next]
  hippoSyncthreads()
  actual[LaneCount + lane] = values[next]

proc independentBlocks(output: ptr cint) {.hippoGlobal.} =
  ## Blocks may execute different numbers of barriers and finish independently.
  var values {.hippoShared.}: array[LaneCount, cint]
  let
    lane = int(threadIdx.x)
    blockId = int(blockIdx.x)
    next = (lane + 1) mod LaneCount
    actual = cast[ptr UncheckedArray[cint]](output)
  for round in 0 .. blockId:
    values[lane] = cint(blockId * 100 + round * 10 + lane)
    hippoSyncthreads()
    actual[blockId * LaneCount + lane] = values[next]
    hippoSyncthreads()

proc helperStorage(): ptr cint {.hippoDeviceInline.} =
  ## Repeated helper calls from a block must borrow the same shared declaration.
  var values {.hippoShared.}: array[LaneCount, cint]
  result = addr values[0]

proc localTypesAndHelpers(output: ptr cint) {.hippoGlobal.} =
  ## Shared types may be local aliases and shared declarations may live in helpers.
  type Scratch = array[LaneCount, cint]
  var local {.hippoShared.}: Scratch
  let
    lane = int(threadIdx.x)
    blockId = int(blockIdx.x)
    next = (lane + 1) mod LaneCount
    first = cast[ptr UncheckedArray[cint]](helperStorage())
    actual = cast[ptr UncheckedArray[cint]](output)
  local[lane] = cint(blockId * 1000 + lane + 20)
  first[lane] = cint(blockId * 1000 + lane + 200)
  hippoSyncthreads()
  let second = cast[ptr UncheckedArray[cint]](helperStorage())
  actual[blockId * 2 * LaneCount + lane] = local[next]
  actual[blockId * 2 * LaneCount + LaneCount + lane] = second[next]

proc checkNeighbors(grid, blockSize: Dim3; seed: cint) =
  ## Compare all rounds against an independent block/lane calculation on the host.
  let
    blocks = int(grid.x * grid.y * grid.z)
    lanes = int(blockSize.x * blockSize.y * blockSize.z)
    count = blocks * RoundCount * lanes
    output = hippoMalloc(count * sizeof(cint))
  var
    seedArg = seed
    actual = newSeq[cint](count)
  hippoMemset(output, 0xFF, count * sizeof(cint))
  hippoLaunchKernel(neighbors, gridDim = grid, blockDim = blockSize,
    args = hippoArgs(output.p, seedArg))
  hippoSynchronize()
  hippoMemcpy(addr actual[0], output, count * sizeof(cint), HippoMemcpyDeviceToHost)
  for blockId in 0 ..< blocks:
    for lane in 0 ..< lanes:
      let next = (lane + 1) mod lanes
      var expected = cint(lane)
      for round in 0 ..< RoundCount:
        expected += seed + cint(blockId * 1000 + round * 100 + next)
        doAssert actual[(blockId * RoundCount + round) * lanes + lane] ==
          expected + cint(blockId + 1 + next)

suite "block shared memory ownership":
  test "neighbor reads, mixed shared types and private locals survive repeated barriers":
    checkNeighbors(newDim3(3), newDim3(LaneCount), 7)

  test "3D blocks and grids have separate shared allocations":
    checkNeighbors(newDim3(2, 2, 2), newDim3(2, 2, 2), 13)

  test "repeated launches with changed dimensions and inputs remain independent":
    for seed in [1, 37, -11]:
      checkNeighbors(newDim3(2), newDim3(4), cint(seed))
      checkNeighbors(newDim3(1), newDim3(LaneCount), cint(seed))

  test "nested shared declarations with the same name do not alias":
    let output = hippoMalloc(2 * LaneCount * sizeof(cint))
    var actual: array[2 * LaneCount, cint]
    hippoLaunchKernel(shadowed, blockDim = newDim3(LaneCount), args = hippoArgs(output.p))
    hippoSynchronize()
    hippoMemcpy(addr actual[0], output, sizeof(actual), HippoMemcpyDeviceToHost)
    for lane in 0 ..< LaneCount:
      check actual[lane] == cint((lane + 1) mod LaneCount + 100)
      check actual[LaneCount + lane] == cint((lane + 1) mod LaneCount + 10)

  test "local type aliases and repeated device helper calls retain block ownership":
    const Blocks = 2
    let output = hippoMalloc(Blocks * 2 * LaneCount * sizeof(cint))
    var actual: array[Blocks * 2 * LaneCount, cint]
    hippoLaunchKernel(localTypesAndHelpers, gridDim = newDim3(Blocks),
      blockDim = newDim3(LaneCount), args = hippoArgs(output.p))
    hippoSynchronize()
    hippoMemcpy(addr actual[0], output, sizeof(actual), HippoMemcpyDeviceToHost)
    for blockId in 0 ..< Blocks:
      for lane in 0 ..< LaneCount:
        check actual[blockId * 2 * LaneCount + lane] == cint(blockId * 1000 + (lane + 1) mod LaneCount + 20)
        check actual[blockId * 2 * LaneCount + LaneCount + lane] == cint(blockId * 1000 + (lane + 1) mod LaneCount + 200)

  test "a completed block does not prevent other blocks reaching later barriers":
    const Blocks = 3
    let output = hippoMalloc(Blocks * LaneCount * sizeof(cint))
    var actual: array[Blocks * LaneCount, cint]
    hippoLaunchKernel(independentBlocks, gridDim = newDim3(Blocks),
      blockDim = newDim3(LaneCount), args = hippoArgs(output.p))
    hippoSynchronize()
    hippoMemcpy(addr actual[0], output, sizeof(actual), HippoMemcpyDeviceToHost)
    for blockId in 0 ..< Blocks:
      for lane in 0 ..< LaneCount:
        check actual[blockId * LaneCount + lane] ==
          cint(blockId * 110 + (lane + 1) mod LaneCount)

when HippoRuntime == "SIMPLE":
  proc retiring(output: ptr cint; reverse: cint) {.hippoGlobal.} =
    ## Retire lanes both before and after peers have arrived at a later barrier.
    var values {.hippoShared.}: array[LaneCount, cint]
    let
      lane = int(threadIdx.x)
      rank = if reverse == 0: lane else: LaneCount - lane - 1
      blockId = int(blockIdx.x)
      actual = cast[ptr UncheckedArray[cint]](output)
    for round in 0 ..< LaneCount:
      if rank < round:
        return
      values[rank] = cint(rank + blockId * 100)
      hippoSyncthreads()
      var total: cint
      for other in round ..< LaneCount:
        total += values[other]
      actual[(blockId * LaneCount + round) * LaneCount + rank] = total
      hippoSyncthreads()

  proc failing() {.hippoGlobal.} =
    ## Exercise launch cleanup when a logical thread raises an exception.
    var values {.hippoShared.}: array[LaneCount, cint]
    values[int(threadIdx.x)] = 1
    hippoSyncthreads()
    raise newException(ValueError, "Intentional SIMPLE kernel failure.")

  proc ioFailure() {.hippoGlobal.} =
    ## Exercise value-only propagation of a user exception from a CPU worker.
    raise newException(IOError, "Intentional worker I/O failure.")

  proc defectFailure() {.hippoGlobal.} =
    ## A worker defect must remain a defect on the calling thread.
    raise newException(AssertionDefect, "Intentional worker defect.")

  suite "SIMPLE block cleanup":
    test "shared exchanges remain correct with fewer workers than lanes":
      for workers in [1'u, 2'u, 3'u, 4'u]:
        setThreads(workers)
        checkNeighbors(newDim3(3), newDim3(LaneCount), cint(workers))
        checkNeighbors(newDim3(1), newDim3(LaneCount), cint(workers + 10))

    test "exited lanes retire from barriers while the remaining lanes keep exchanging values":
      const Blocks = 3
      let output = hippoMalloc(Blocks * LaneCount * LaneCount * sizeof(cint))
      var actual: array[Blocks * LaneCount * LaneCount, cint]
      for workers in [1'u, 2'u, 3'u, 4'u]:
        setThreads(workers)
        for order in 0 .. 1:
          var reverse = cint(order)
          hippoMemset(output, 0xFF, sizeof(actual))
          hippoLaunchKernel(retiring, gridDim = newDim3(Blocks),
            blockDim = newDim3(LaneCount), args = hippoArgs(output.p, reverse))
          hippoMemcpy(addr actual[0], output, sizeof(actual), HippoMemcpyDeviceToHost)
          for blockId in 0 ..< Blocks:
            for round in 0 ..< LaneCount:
              let expected = cint((LaneCount - round) * (round + LaneCount - 1) div 2 +
                (LaneCount - round) * blockId * 100)
              for rank in 0 ..< LaneCount:
                check actual[(blockId * LaneCount + round) * LaneCount + rank] ==
                  (if rank < round: -1.cint else: expected)

    test "kernel exceptions release launch state before a later launch":
      expect ValueError:
        hippoLaunchKernel(failing, gridDim = newDim3(3),
          blockDim = newDim3(LaneCount), args = hippoArgs())
      checkNeighbors(newDim3(2), newDim3(LaneCount), 23)

    test "other worker exception types retain their diagnostic identity":
      try:
        hippoLaunchKernel(ioFailure, gridDim = newDim3(3), args = hippoArgs())
        fail()
      except SimpleKernelError as error:
        check error.originalType == "IOError"
        check "Intentional worker I/O failure." in error.msg
      checkNeighbors(newDim3(2), newDim3(LaneCount), 29)

    test "worker defects remain defects and release the launch state":
      expect Defect:
        hippoLaunchKernel(defectFailure, gridDim = newDim3(3), args = hippoArgs())
      checkNeighbors(newDim3(2), newDim3(LaneCount), 31)
