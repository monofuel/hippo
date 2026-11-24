import
  std/[unittest, sets, atomics, monotimes, times, os],
  hippo

when HippoRuntime == "SIMPLE":
  const
    Blocks = 12
    Lanes = 4
    ParticipationTimeoutMs = 5000

  proc currentWorker(): int =
    ## Identify the actual OS thread, or the only execution context without threading.
    when compileOption("threads"):
      result = getThreadId()
    else:
      result = 0

  proc workerIds(output: ptr int; arrivals: ptr Atomic[int]; required: int) {.hippoGlobal.} =
    ## Keep initial lanes active until every expected OS worker participates.
    let
      lane = int(threadIdx.x)
      blockId = int(blockIdx.x)
      actual = cast[ptr UncheckedArray[int]](output)
      owner = currentWorker()
    discard arrivals[].fetchAdd(1)
    let start = getMonoTime()
    while arrivals[].load() < required:
      if (getMonoTime() - start).inMilliseconds > ParticipationTimeoutMs:
        raise newException(ValueError, "Expected SIMPLE CPU workers did not participate.")
      sleep(1)
    hippoSyncthreads()
    if owner != currentWorker():
      raise newException(ValueError, "A logical lane moved between workers across a barrier.")
    actual[blockId * Lanes + lane] = owner

  proc checkWorkers(requested: uint; blocks: int) =
    ## Check worker bounds, within-block participation, lane affinity and completion.
    setThreads(requested)
    let
      expected = when compileOption("threads"): min(int(requested), blocks * Lanes)
                 else: 1
      output = hippoMalloc(blocks * Lanes * sizeof(int))
      caller = currentWorker()
    var
      required = expected
      arrivals: Atomic[int]
      actual = newSeq[int](blocks * Lanes)
    hippoMemset(output, 0xFF, actual.len * sizeof(int))
    hippoLaunchKernel(workerIds, gridDim = newDim3(blocks), blockDim = newDim3(Lanes),
      args = hippoArgs(output.p, addr arrivals, required))
    hippoMemcpy(addr actual[0], output, actual.len * sizeof(int), HippoMemcpyDeviceToHost)
    var observed = initHashSet[int]()
    for blockId in 0 ..< blocks:
      var blockWorkers = initHashSet[int]()
      for lane in 0 ..< Lanes:
        let owner = actual[blockId * Lanes + lane]
        doAssert owner != -1, "Launch returned before a lane wrote its output."
        observed.incl(owner)
        blockWorkers.incl(owner)
      check blockWorkers.len == min(expected, Lanes)
    check observed.len == expected
    if expected == 1:
      check caller in observed
    else:
      check caller notin observed
    check arrivals.load() == blocks * Lanes

  suite "SIMPLE CPU worker scheduling":
    test "one worker executes on the calling thread":
      checkWorkers(1, Blocks)

    test "two and four workers really participate without exceeding the bound":
      for requested in [2'u, 4'u]:
        checkWorkers(requested, Blocks)

    test "one block uses multiple workers even when lanes outnumber workers":
      checkWorkers(2, 1)

    test "worker count is capped by the number of logical lanes":
      checkWorkers(8, 1)

    test "repeated launches and worker-count changes finish before returning":
      for repetition in 0 ..< 10:
        for requested in [2'u, 1'u, 4'u]:
          checkWorkers(requested, Blocks)

    test "zero workers is rejected":
      expect ValueError:
        setThreads(0)
else:
  suite "SIMPLE CPU worker scheduling":
    test "SIMPLE-only scheduler checks":
      skip()
