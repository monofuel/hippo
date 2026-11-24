import
  std/[unittest, sets, atomics, monotimes, times, os],
  hippo

when HippoRuntime == "SIMPLE":
  const
    Blocks = 12
    Lanes = 4
    ParticipationTimeoutMs = 5000

  var workerVisits {.threadvar.}: int

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

  proc retainedWorkerState(output: ptr int) {.hippoGlobal.} =
    ## Observe worker identity and thread-local state surviving between launches.
    let
      lane = int(threadIdx.x)
      actual = cast[ptr UncheckedArray[int]](output)
    inc workerVisits
    actual[lane * 2] = currentWorker()
    actual[lane * 2 + 1] = workerVisits

  proc captureWorkers(lanes: int): seq[int] =
    ## Read one identity and visit count per lane without changing pool configuration.
    let output = hippoMalloc(lanes * 2 * sizeof(int))
    result = newSeq[int](lanes * 2)
    hippoLaunchKernel(retainedWorkerState, blockDim = newDim3(lanes), args = hippoArgs(output.p))
    hippoMemcpy(addr result[0], output, result.len * sizeof(int), HippoMemcpyDeviceToHost)

  proc oneWorkerFails() {.hippoGlobal.} =
    ## Cancel peers waiting at a barrier, leaving persistent workers reusable.
    if threadIdx.x == 0:
      raise newException(ValueError, "Intentional persistent worker failure.")
    hippoSyncthreads()

  proc workerPoolControl(mode: int) {.hippoGlobal.} =
    ## Reentrant pool control must fail instead of waiting for its own launch.
    if threadIdx.x == 0:
      if mode == 0:
        setThreads(1)
      else:
        shutdownSimpleWorkers()
    hippoSyncthreads()

  proc callerFailure() {.hippoGlobal.} =
    ## Exercise wrapped exceptions on the calling thread rather than a pool worker.
    raise newException(IOError, "Intentional caller kernel failure.")

  when compileOption("threads"):
    type HostStart = object
      output: ptr UncheckedArray[int]
      seed: int

    proc hostKernel(output: ptr int; seed: int) {.hippoGlobal.} =
      ## Write a caller-specific result through the shared persistent pool.
      let actual = cast[ptr UncheckedArray[int]](output)
      actual[int(threadIdx.x)] = seed + int(threadIdx.x)

    proc hostCaller(start: HostStart) {.thread.} =
      ## Submit independent synchronous launches from a short-lived host thread.
      for iteration in 0 ..< 30:
        var seed = start.seed + iteration * 10
        hippoLaunchKernel(hostKernel, blockDim = newDim3(Lanes),
          args = hippoArgs(cast[ptr int](start.output), seed))
        for lane in 0 ..< Lanes:
          doAssert start.output[lane] == seed + lane

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

    test "shutdown is idempotent and later launches recreate workers":
      checkWorkers(2, Blocks)
      shutdownSimpleWorkers()
      shutdownSimpleWorkers()
      checkWorkers(2, Blocks)

    test "pool control from a kernel is rejected without deadlocking":
      for choice in 0 .. 1:
        var mode = choice
        expect ValueError:
          hippoLaunchKernel(workerPoolControl, blockDim = newDim3(Lanes), args = hippoArgs(mode))
        checkWorkers(2, Blocks)

    test "wrapped caller failures release context and dispatch before the next launch":
      setThreads(1)
      expect SimpleKernelError:
        hippoLaunchKernel(callerFailure, args = hippoArgs())
      checkWorkers(1, Blocks)
      checkWorkers(2, Blocks)

  when compileOption("threads"):
    suite "SIMPLE persistent worker lifetime":
      test "worker identities and thread-local state survive smaller and repeated launches":
        shutdownSimpleWorkers()
        setThreads(4)
        let first = captureWorkers(4)
        for repetition in 2 .. 20:
          # Setting the same limit must not discard the existing pool.
          setThreads(4)
          let current = captureWorkers(4)
          for lane in 0 ..< 4:
            check current[lane * 2] == first[lane * 2]
            check current[lane * 2 + 1] == repetition
        let smaller = captureWorkers(2)
        for lane in 0 ..< 2:
          check smaller[lane * 2] == first[lane * 2]
          check smaller[lane * 2 + 1] == 21
        let larger = captureWorkers(4)
        for lane in 0 ..< 4:
          check larger[lane * 2] == first[lane * 2]
          check larger[lane * 2 + 1] == (if lane < 2: 22 else: 21)

      test "configuration changes and shutdown retire old worker state":
        shutdownSimpleWorkers()
        for workers in [4'u, 2'u, 8'u, 3'u]:
          setThreads(workers)
          let actual = captureWorkers(int(workers))
          for lane in 0 ..< int(workers):
            check actual[lane * 2 + 1] == 1
        shutdownSimpleWorkers()
        let recreated = captureWorkers(3)
        for lane in 0 ..< 3:
          check recreated[lane * 2 + 1] == 1
        setThreads(1)
        let caller = captureWorkers(1)
        check caller[0] == currentWorker()

      test "failure cancellation does not poison or replace persistent workers":
        setThreads(4)
        let first = captureWorkers(4)
        for repetition in 2 .. 20:
          expect ValueError:
            hippoLaunchKernel(oneWorkerFails, blockDim = newDim3(4), args = hippoArgs())
          let current = captureWorkers(4)
          for lane in 0 ..< 4:
            check current[lane * 2] == first[lane * 2]
            check current[lane * 2 + 1] == repetition

      test "host callers may exit while their pool remains reusable by another caller":
        shutdownSimpleWorkers()
        setThreads(4)
        var
          output: array[2, array[Lanes, int]]
          callers: array[2, Thread[HostStart]]
        for index in 0 ..< callers.len:
          createThread(callers[index], hostCaller,
            HostStart(output: cast[ptr UncheckedArray[int]](addr output[index][0]), seed: index * 10000))
        for index in 0 ..< callers.len:
          joinThread(callers[index])
        checkWorkers(4, Blocks)
        shutdownSimpleWorkers()
else:
  suite "SIMPLE CPU worker scheduling":
    test "SIMPLE-only scheduler checks":
      skip()
