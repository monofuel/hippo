import
  std/unittest,
  hippo

when HippoRuntime in ["HIP", "HIP_CPU", "CUDA"]:
  type
    HashEntry = object
      key: uint32
      value: uint32
      next: ptr HashEntry

  const
    ThreadsPerBlock = 128
    DotElements = 33 * 1024 + 17
    HashElements = 8 * 1024 + 19
    MaxBlocks = 64
    Repetitions = 3
    EmptySeed = 17.0'f32

  proc acquireLock(mutex: ptr int32) {.hippoDeviceInline.} =
    ## Acquire a device-wide lock with the fence required by the book's errata.
    while hippoAtomicCAS(mutex, 0'i32, 1'i32) != 0'i32:
      discard
    hippoThreadfence()

  proc releaseLock(mutex: ptr int32) {.hippoDeviceInline.} =
    ## Publish protected writes before another block acquires this lock.
    hippoThreadfence()
    discard hippoAtomicExch(mutex, 0'i32)

  proc lockedDot(a, b, total: ptr cfloat; mutex: ptr int32; count: int32) {.hippoGlobal.} =
    ## A.1.2: reduce each block, then add its partial sum under a global lock.
    var partials {.hippoShared.}: array[ThreadsPerBlock, cfloat]
    let
      aValues = cast[ptr UncheckedArray[cfloat]](a)
      bValues = cast[ptr UncheckedArray[cfloat]](b)
      lane = int(threadIdx.x)
      stride = int(blockDim.x * gridDim.x)
    var
      index = int(threadIdx.x + blockIdx.x * blockDim.x)
      partial = 0.0'f32
    while index < int(count):
      partial = partial + aValues[index] * bValues[index]
      index = index + stride
    partials[lane] = partial
    hippoSyncthreads()

    var remaining = int(blockDim.x) div 2
    while remaining > 0:
      if lane < remaining:
        partials[lane] = partials[lane] + partials[lane + remaining]
      hippoSyncthreads()
      remaining = remaining div 2

    if lane == 0:
      acquireLock(mutex)
      total[] = total[] + partials[0]
      releaseLock(mutex)

  proc insertEntries(keys: ptr uint32; pool: ptr HashEntry;
                     heads: ptr ptr HashEntry; locks: ptr int32;
                     count, bucketCount: int32) {.hippoGlobal.} =
    ## A.2.4: prepend pool entries to bucket lists protected by per-bucket locks.
    let
      keyValues = cast[ptr UncheckedArray[uint32]](keys)
      entries = cast[ptr UncheckedArray[HashEntry]](pool)
      buckets = cast[ptr UncheckedArray[ptr HashEntry]](heads)
      mutexes = cast[ptr UncheckedArray[int32]](locks)
      stride = int(blockDim.x * gridDim.x)
    var index = int(threadIdx.x + blockIdx.x * blockDim.x)
    # Serialize lanes within each wave so a spinning lane cannot strand its owner.
    # HIP-CPU does not emulate lockstep execution, but supports the same algorithm.
    while index - int(threadIdx.x) < int(count):
      for turn in 0 ..< HippoWarpSize:
        if int(threadIdx.x) mod HippoWarpSize == turn and index < int(count):
          let
            key = keyValues[index]
            bucket = int(key mod uint32(bucketCount))
            entry = addr entries[index]
            mutex = addr mutexes[bucket]
          entry.key = key
          entry.value = uint32(index)
          acquireLock(mutex)
          entry.next = buckets[bucket]
          buckets[bucket] = entry
          releaseLock(mutex)
      index = index + stride

  proc checkDot(blocks: int; count: int) =
    ## Check the lock-based GPU result against an independent host dot product.
    var
      a = newSeq[cfloat](DotElements)
      b = newSeq[cfloat](DotElements)
      expected = float64(EmptySeed)
      countArg = int32(count)
    # Positive integer products are exact in float32 and expose any lost block sum.
    for index in 0 ..< DotElements:
      a[index] = cfloat(index mod 7 + 1)
      b[index] = cfloat(index mod 5 + 1)
      if index < count:
        expected += float64(a[index]) * float64(b[index])
    let
      devA = hippoMalloc(a.len * sizeof(cfloat))
      devB = hippoMalloc(b.len * sizeof(cfloat))
      devTotal = hippoMalloc(sizeof(cfloat))
      devMutex = hippoMalloc(sizeof(int32))
    hippoMemcpy(devA, addr a[0], a.len * sizeof(cfloat), HippoMemcpyHostToDevice)
    hippoMemcpy(devB, addr b[0], b.len * sizeof(cfloat), HippoMemcpyHostToDevice)

    for repetition in 0 ..< Repetitions:
      var
        actual = EmptySeed
        mutex = 0'i32
      hippoMemcpy(devTotal, addr actual, sizeof(actual), HippoMemcpyHostToDevice)
      hippoMemcpy(devMutex, addr mutex, sizeof(mutex), HippoMemcpyHostToDevice)
      hippoLaunchKernel(lockedDot,
        gridDim = newDim3(uint32(blocks)),
        blockDim = newDim3(uint32(ThreadsPerBlock)),
        args = hippoArgs(devA.p, devB.p, devTotal.p, devMutex.p, countArg))
      hippoSynchronize()
      hippoMemcpy(addr actual, devTotal, sizeof(actual), HippoMemcpyDeviceToHost)
      hippoMemcpy(addr mutex, devMutex, sizeof(mutex), HippoMemcpyDeviceToHost)
      checkpoint "A.1 blocks=" & $blocks & " count=" & $count & " run=" & $repetition
      check float64(actual) == expected
      check mutex == 0'i32

  proc checkHashTable(blocks, bucketCount, count: int) =
    ## Verify every node, key, value and bucket after converting device pointers.
    var
      keys = newSeq[uint32](HashElements)
      entries = newSeq[HashEntry](HashElements)
      heads = newSeq[ptr HashEntry](bucketCount)
      locks = newSeq[int32](bucketCount)
      countArg = int32(count)
      bucketsArg = int32(bucketCount)
    for index in 0 ..< keys.len:
      # Duplicate keys are intentional: each input still needs its own node.
      keys[index] = uint32((index * 17 + index div 7) mod 257)
    let
      devKeys = hippoMalloc(keys.len * sizeof(uint32))
      devPool = hippoMalloc(entries.len * sizeof(HashEntry))
      devHeads = hippoMalloc(heads.len * sizeof(ptr HashEntry))
      devLocks = hippoMalloc(locks.len * sizeof(int32))
      poolStart = cast[uint](devPool.p)
    hippoMemcpy(devKeys, addr keys[0], keys.len * sizeof(uint32), HippoMemcpyHostToDevice)

    for repetition in 0 ..< Repetitions:
      hippoMemset(devPool, 0, entries.len * sizeof(HashEntry))
      hippoMemset(devHeads, 0, heads.len * sizeof(ptr HashEntry))
      hippoMemset(devLocks, 0, locks.len * sizeof(int32))
      hippoLaunchKernel(insertEntries,
        gridDim = newDim3(uint32(blocks)),
        blockDim = newDim3(uint32(ThreadsPerBlock)),
        args = hippoArgs(devKeys.p, devPool.p, devHeads.p, devLocks.p,
                        countArg, bucketsArg))
      hippoSynchronize()
      hippoMemcpy(addr entries[0], devPool, entries.len * sizeof(HashEntry), HippoMemcpyDeviceToHost)
      hippoMemcpy(addr heads[0], devHeads, heads.len * sizeof(ptr HashEntry), HippoMemcpyDeviceToHost)
      hippoMemcpy(addr locks[0], devLocks, locks.len * sizeof(int32), HippoMemcpyDeviceToHost)

      checkpoint "A.2 blocks=" & $blocks & " buckets=" & $bucketCount &
        " count=" & $count & " run=" & $repetition
      var
        seen = newSeq[bool](count)
        total = 0
      for bucket in 0 ..< bucketCount:
        var current = heads[bucket]
        while current != nil:
          let pointerValue = cast[uint](current)
          # Reject corrupt pointers before converting them to host pool indices.
          require pointerValue >= poolStart
          let offset = pointerValue - poolStart
          require offset mod uint(sizeof(HashEntry)) == 0
          require offset div uint(sizeof(HashEntry)) < uint(count)
          let index = int(offset div uint(sizeof(HashEntry)))
          require not seen[index]
          seen[index] = true
          inc total
          check entries[index].key == keys[index]
          check entries[index].value == uint32(index)
          check int(entries[index].key mod uint32(bucketCount)) == bucket
          current = entries[index].next
        check locks[bucket] == 0'i32
      check total == count
      for present in seen:
        check present

  suite "Appendix A.1: lock-based dot product":
    test "one block and multiple blocks match the host reference":
      for blocks in [1, 2, MaxBlocks]:
        checkDot(blocks, DotElements)
    test "empty and partial blocks preserve the initial accumulator":
      for count in [0, 1, ThreadsPerBlock + 3]:
        checkDot(MaxBlocks, count)

  suite "Appendix A.2: lock-based hash table":
    test "one block and multiple blocks retain every entry":
      for blocks in [1, 2, MaxBlocks]:
        checkHashTable(blocks, 32, HashElements)
    test "all keys colliding in one bucket retain every entry":
      checkHashTable(MaxBlocks, 1, HashElements)
    test "empty input and partial blocks retain exactly the supplied entries":
      for count in [0, 1, ThreadsPerBlock + 3]:
        checkHashTable(MaxBlocks, 32, count)
else:
  suite "Appendix A: Advanced Atomics":
    test "SIMPLE atomic locking is deferred":
      skip()
