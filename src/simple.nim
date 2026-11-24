## `simple` is a pure-nim backend for hippo.
## Logical GPU lanes run cooperatively as iterators, with storage owned by each block.
## Threaded builds distribute lanes, including those in one block, across bounded workers.
import
  std/[cpuinfo, macros, os, strutils, math, locks, typetraits, atomics, exitprocs]

## TERMINOLOGY: `threads` or `SingleThread` refers to the number of CPU threads we are using under the hood.
## Execution is cooperative; the threading option does not create an OS thread per lane.
## Logical GPU threads are lane iterators; each block owns shared storage and barrier state.
## GPU threads can sync with all the other threads in a block using `hippoSyncthreads()` just like in CUDA/HIP.
## Barriers yield worker capacity while other logical lanes remain queued.
## BlockDim, GridDim, ThreadIdx, BlockIdx, are all referring to GPU threads.


const SingleThread = defined(js) or not compileOption("threads")

type
  Dim3* = object
    x*: uint
    y*: uint
    z*: uint
  BlockDim* = Dim3
  GridDim* = Dim3
  ThreadIdx* = Dim3
  BlockIdx* = Dim3
  HippoStream* = pointer
  HippoError* = uint
  HippoMemcpyKind = enum 
    SimpleMemcpyHostToHost = 0,
    SimpleMemcpyHostToDevice = 1,
    SimpleMemcpyDeviceToHost = 2,
    SimpleMemcpyDeviceToDevice = 3,
    SimpleMemcpyDefault = 4

const
  HippoMemcpyHostToHost* = SimpleMemcpyHostToHost
  HippoMemcpyHostToDevice* = SimpleMemcpyHostToDevice
  HippoMemcpyDeviceToHost* = SimpleMemcpyDeviceToHost
  HippoMemcpyDeviceToDevice* = SimpleMemcpyDeviceToDevice
  HippoMemcpyDefault* = SimpleMemcpyDefault

proc newDim3*(x: SomeInteger = 1; y: SomeInteger = 1; z: SomeInteger = 1): Dim3 =
  result.x = uint(x)
  result.y = uint(y)
  result.z = uint(z)


var threads = 1.uint

proc simpleInit() =
  ## get the number of cpu cores and set the number of threads to use.
  when SingleThread:
    threads = 1
  else:
    let envThreads = getEnv("OMP_NUM_THREADS")
    if envThreads.len > 0:
      try: threads = parseInt(envThreads).uint
      except ValueError: threads = countProcessors().uint
    else:
      threads = countProcessors().uint
    echo "DEBUG: hippo/simple: Using up to ", threads, " CPU workers"

simpleInit()

proc simpleMalloc(p: ptr pointer, size: int) =
  p[] = allocShared(size)

proc simpleMemcpy(dst: pointer, src: pointer, size: int, kind: HippoMemcpyKind) =
  copyMem(dst, src, size)

proc simpleMemset(dst: pointer, value: cint, size: int) =
  ## SIMPLE device memory is host memory, so a plain host fill is enough.
  ## Zeroing takes the fast `zeroMem` path, any other byte fills in a loop.
  let fill = byte(value and 0xFF)
  if fill == 0:
    zeroMem(dst, size)
  else:
    let bytes = cast[ptr UncheckedArray[byte]](dst)
    for i in 0 ..< size:
      bytes[i] = fill

proc simpleFree(p: pointer) =
  deallocShared(p)

proc handleError*(err: HippoError) =
  ## Simple runtime raises errors as exceptions.
  discard

proc hipDeviceSynchronize*(): HippoError =
  ## SIMPLE backend executes synchronously, so there's nothing to wait for.
  0

macro callWithTuple(fn: typed, args: typed): untyped =
  ## Expand tuple arguments into a call, casting each to its formal parameter type.
  proc paramTypeAt(params: NimNode, idx: int): NimNode =
    var count = 0
    for i in 1..<params.len:
      let def = params[i]
      let typ = def[^2]
      for n in 0..<(def.len - 2):
        if count == idx:
          return typ
        inc count
    result = newEmptyNode()

  result = newCall(fn)
  var params = newEmptyNode()
  if fn.kind == nnkSym:
    let impl = fn.getImpl
    if impl.kind == nnkProcDef:
      params = impl[3]
  if params.kind == nnkEmpty:
    let fnType = fn.getType
    if fnType.kind == nnkProcTy:
      params = fnType[0]

  if args.getTypeImpl.kind in {nnkTupleTy, nnkTupleConstr}:
    for index in 0 ..< args.getTypeImpl.len:
      let
        child = newTree(nnkBracketExpr, args, newLit(index))
        paramType = if params.kind != nnkEmpty: paramTypeAt(params, index) else: newEmptyNode()
      if paramType.kind != nnkEmpty:
        result.add(newTree(nnkCast, paramType, child))
      else:
        result.add(child)
  elif args.kind in {nnkTupleConstr, nnkPar}:
    var i = 0
    for child in args:
      let paramType = if params.kind != nnkEmpty: paramTypeAt(params, i) else: newEmptyNode()
      if paramType.kind != nnkEmpty:
        result.add(newTree(nnkCast, paramType, child))
      else:
        result.add(child)
      inc i
  else:
    let paramType = if params.kind != nnkEmpty: paramTypeAt(params, 0) else: newEmptyNode()
    if paramType.kind != nnkEmpty:
      result.add(newTree(nnkCast, paramType, args))
    else:
      result.add(args)

type
  SimpleThread = iterator(): bool {.closure, gcsafe.}
  SimpleWorkerProc = proc(workerIndex, workerCount: int; cancelled: ptr Atomic[bool]) {.closure, gcsafe.}
  SimpleSharedSlot = object
    key, allocation, data: pointer
    next: ptr SimpleSharedSlot
  SimpleBlockState = object
    lock: Lock
    arrived, generation, completed, laneCount: int
    shared: ptr SimpleSharedSlot
  SimpleKernelError* = object of CatchableError
    ## User kernel failures with their original exception type and diagnostic text.
    originalType*: string
  SimpleFailureKind = enum
    valueFailure, defectFailure, kernelFailure
  SimpleFailure = object
    kind: SimpleFailureKind
    name, message: cstring

var
  blockIdx* {.threadvar.}: BlockIdx
  threadIdx* {.threadvar.}: ThreadIdx
  blockDim* {.threadvar.}: BlockDim
  gridDim* {.threadvar.}: GridDim
  currentSimpleBlock {.threadvar.}: ptr SimpleBlockState

proc simpleAllocate(count, size: csize_t): pointer {.importc: "calloc", header: "stdlib.h".}
  ## Allocate zeroed storage whose lifetime can extend past its creating CPU worker.
proc simpleRelease(memory: pointer) {.importc: "free", header: "stdlib.h".}
  ## Release storage allocated by simpleAllocate, on any CPU worker or the caller.

proc sharedErrorText(text: string): cstring =
  ## Copy diagnostics outside worker-owned Nim heaps before the worker exits.
  result = cast[cstring](simpleAllocate(1, csize_t(text.len + 1)))
  if result == nil:
    raise newException(OutOfMemDefect, "Unable to copy a SIMPLE worker failure.")
  if text.len > 0:
    copyMem(result, unsafeAddr text[0], text.len)

proc releaseFailure(failure: var SimpleFailure) =
  ## Release the value-only failure record after recreating caller-owned diagnostics.
  simpleRelease(failure.name)
  simpleRelease(failure.message)
  failure.name = nil
  failure.message = nil

template withSimpleBlock(state: ptr SimpleBlockState; body: untyped) =
  when not SingleThread:
    acquire(state.lock)
  try:
    body
  finally:
    when not SingleThread:
      release(state.lock)

proc simpleSharedMemory[T](site: static[string]): ptr T =
  ## Borrow one raw shared allocation per declaration and block until launch completion.
  when not supportsCopyMem(T):
    {.error: "SIMPLE shared memory requires a type without managed references or destructors.".}
  # The tag identifies a declaration; the allocation itself belongs to a block.
  var declarationTag {.global.}: byte
  let state = currentSimpleBlock
  if state == nil:
    raise newException(ValueError, "Shared memory requires an active SIMPLE block.")
  withSimpleBlock(state):
    var slot = state.shared
    while slot != nil and slot.key != addr declarationTag:
      slot = slot.next
    if slot == nil:
      slot = cast[ptr SimpleSharedSlot](simpleAllocate(1, sizeof(SimpleSharedSlot).csize_t))
      if slot == nil:
        raise newException(OutOfMemDefect, "Unable to allocate a SIMPLE shared-memory slot.")
      slot.allocation = simpleAllocate(1, csize_t(max(1, sizeof(T) + alignof(T) - 1)))
      if slot.allocation == nil:
        simpleRelease(slot)
        raise newException(OutOfMemDefect, "Unable to allocate SIMPLE block-shared memory.")
      slot.data = cast[pointer]((cast[uint](slot.allocation) + uint(alignof(T) - 1)) and
        not uint(alignof(T) - 1))
      slot.key = addr declarationTag
      slot.next = state.shared
      state.shared = slot
    result = cast[ptr T](slot.data)

proc describeFailure(error: ref Exception): SimpleFailure =
  ## Keep ORC-managed exception graphs on their worker and transfer diagnostic values.
  result.kind = if error of ValueError: valueFailure
                elif error of Defect: defectFailure
                else: kernelFailure
  result.name = sharedErrorText($error.name)
  result.message = sharedErrorText($error.name & ": " & error.msg & "\n" & getStackTrace(error))

proc raiseFailure(failure: var SimpleFailure) {.noreturn.} =
  ## Recreate failures on the caller without converting defects to recoverable errors.
  let
    kind = failure.kind
    originalType = $failure.name
    message = $failure.message
  releaseFailure(failure)
  case kind
  of valueFailure:
    raise newException(ValueError, message)
  of defectFailure:
    raise newException(Defect, message)
  of kernelFailure:
    let error = newException(SimpleKernelError, message)
    error.originalType = originalType
    raise error

proc simpleLaneFinished(state: ptr SimpleBlockState) =
  ## Retire a lane from later barriers, releasing peers already waiting for its exit.
  withSimpleBlock(state):
    inc state.completed
    if state.arrived > 0 and state.arrived == state.laneCount - state.completed:
      state.arrived = 0
      inc state.generation

proc simpleArrive(state: ptr SimpleBlockState): int =
  ## Publish this lane's writes and advance the generation after the last arrival.
  withSimpleBlock(state):
    result = state.generation
    inc state.arrived
    if state.arrived == state.laneCount - state.completed:
      state.arrived = 0
      inc state.generation

proc simpleBarrierPending(state: ptr SimpleBlockState; generation: int): bool =
  ## Acquire the block lock before observing a barrier release and its memory writes.
  withSimpleBlock(state):
    result = state.generation == generation

proc runAllTasks(tasks: seq[SimpleThread]; cancelled: ptr Atomic[bool] = nil) {.gcsafe.} =
  ## Keep lane iterators on their owning worker and yield capacity at barriers.
  var active = tasks
  while active.len > 0:
    var index = 0
    while index < active.len:
      if cancelled != nil and cancelled[].load():
        return
      if active[index]():
        inc index
      else:
        active.del(index)

when not SingleThread:
  type
    SimpleWorkState = object
      lock: Lock
      runWorker: ptr SimpleWorkerProc
      cancelled: Atomic[bool]
      failure: SimpleFailure
    SimplePoolState = object
      lock: Lock
      wake, finished: Cond
      generation: uint
      stopping: bool
      work: ptr SimpleWorkState
      activeCount, remaining, size: int
      workers: ptr UncheckedArray[Thread[SimpleWorkerStart]]
    SimpleWorkerStart = object
      pool: ptr SimplePoolState
      index: int

  var
    simpleDispatchLock: Lock
    simplePool: SimplePoolState

  initLock(simpleDispatchLock)
  initLock(simplePool.lock)
  initCond(simplePool.wake)
  initCond(simplePool.finished)

  proc simpleWorker(start: SimpleWorkerStart) {.thread.} =
    ## Sleep between launches and keep each launch's iterators on this worker.
    let pool = start.pool
    var generation = 0'u
    while true:
      acquire(pool.lock)
      while not pool.stopping and generation == pool.generation:
        wait(pool.wake, pool.lock)
      if pool.stopping:
        release(pool.lock)
        return
      generation = pool.generation
      let
        count = pool.activeCount
        work = if start.index < count: pool.work else: nil
      release(pool.lock)
      # An inactive worker may wake after the participating workers already finished.
      if work == nil:
        continue
      try:
        work.runWorker[](start.index, count, addr work.cancelled)
      except Exception as error:
        acquire(work.lock)
        try:
          if work.failure.name == nil:
            work.failure = describeFailure(error)
          work.cancelled.store(true)
        finally:
          release(work.lock)
      acquire(pool.lock)
      dec pool.remaining
      if pool.remaining == 0:
        pool.work = nil
        signal(pool.finished)
      release(pool.lock)

  proc stopSimpleWorkers() =
    ## Join idle pool workers while the caller holds the launch dispatch lock.
    acquire(simplePool.lock)
    simplePool.stopping = true
    broadcast(simplePool.wake)
    release(simplePool.lock)
    for index in 0 ..< simplePool.size:
      joinThread(simplePool.workers[index])
    simpleRelease(simplePool.workers)
    simplePool.workers = nil
    simplePool.size = 0
    acquire(simplePool.lock)
    simplePool.stopping = false
    simplePool.generation = 0
    release(simplePool.lock)

  proc ensureSimpleWorkers(count: int) =
    ## Grow lazily; smaller launches reuse a subset without moving thread handles.
    if simplePool.size >= count:
      return
    stopSimpleWorkers()
    # Nim's thread entry point borrows its Thread object, so handles must not move.
    simplePool.workers = cast[ptr UncheckedArray[Thread[SimpleWorkerStart]]](
      simpleAllocate(count.csize_t, sizeof(Thread[SimpleWorkerStart]).csize_t))
    if simplePool.workers == nil:
      raise newException(OutOfMemDefect, "Unable to allocate SIMPLE worker handles.")
    try:
      for index in 0 ..< count:
        createThread(simplePool.workers[index], simpleWorker,
          SimpleWorkerStart(pool: addr simplePool, index: index))
        inc simplePool.size
    except:
      stopSimpleWorkers()
      raise

  proc runParallelLanes(runWorker: SimpleWorkerProc; laneCount: int) =
    ## Borrow the callback until every participating persistent worker has finished.
    let workerCount = min(threads, uint(laneCount)).int
    ensureSimpleWorkers(workerCount)
    var work = SimpleWorkState(runWorker: unsafeAddr runWorker)
    initLock(work.lock)
    try:
      acquire(simplePool.lock)
      try:
        simplePool.work = addr work
        simplePool.activeCount = workerCount
        simplePool.remaining = workerCount
        inc simplePool.generation
        broadcast(simplePool.wake)
        while simplePool.remaining > 0:
          wait(simplePool.finished, simplePool.lock)
      finally:
        release(simplePool.lock)
      if work.failure.name != nil:
        raiseFailure(work.failure)
    finally:
      releaseFailure(work.failure)
      deinitLock(work.lock)

template withSimpleLaunch(body: untyped) =
  if currentSimpleBlock != nil:
    raise newException(ValueError, "SIMPLE launches and worker configuration require host code.")
  when not SingleThread:
    acquire(simpleDispatchLock)
  try:
    body
  finally:
    when not SingleThread:
      release(simpleDispatchLock)

proc shutdownSimpleWorkers*() =
  ## Wait for any launch and stop idle SIMPLE workers; a later launch recreates them.
  ## Host launches share one pool and execute synchronously under the dispatch lock.
  ## Call from host code; workers also stop automatically at normal process exit.
  withSimpleLaunch:
    when not SingleThread:
      stopSimpleWorkers()

proc setThreads*(n: uint) =
  ## Set the maximum CPU workers; changing it stops the pool between launches.
  ## One uses the calling thread. Without threading, every launch uses the caller.
  ## Call from host code; setting the existing limit preserves the pool.
  if n == 0:
    raise newException(ValueError, "SIMPLE needs at least one CPU worker.")
  withSimpleLaunch:
    when not SingleThread:
      if n != threads:
        stopSimpleWorkers()
    threads = n

when not SingleThread:
  proc quitSimpleWorkers() {.noconv.} =
    ## Release persistent workers at normal process exit.
    shutdownSimpleWorkers()

  addExitProc(quitSimpleWorkers)

template simpleLaunchKernel(fn: untyped; gridDimArg, blockDimArg: Dim3; args: tuple) =
  withSimpleLaunch:
    let
      launchGrid = gridDimArg
      launchBlock = blockDimArg
      launchArgs = args
      totalBlocks = int(launchGrid.x * launchGrid.y * launchGrid.z)
      lanesPerBlock = int(launchBlock.x * launchBlock.y * launchBlock.z)
      totalLanes = totalBlocks * lanesPerBlock
      previousBlock = currentSimpleBlock
      states = cast[ptr UncheckedArray[SimpleBlockState]](
        allocShared0(totalBlocks * sizeof(SimpleBlockState)))
    for index in 0 ..< totalBlocks:
      states[index].laneCount = lanesPerBlock
      when not SingleThread:
        initLock(states[index].lock)

    proc makeWorkerTasks(workerIndex, workerCount: int; blockArgs: typeof(launchArgs);
                         blockGrid, blockSize: Dim3;
                         blockStates: ptr UncheckedArray[SimpleBlockState]): seq[SimpleThread] {.nimcall.} =
      ## Create every iterator on its worker, borrowing only raw block-shared memory.
      let
        blocksPerPlane = int(blockGrid.x * blockGrid.y)
        lanesPerPlane = int(blockSize.x * blockSize.y)
        lanesPerBlock = int(blockSize.x * blockSize.y * blockSize.z)
        count = int(blockGrid.x * blockGrid.y * blockGrid.z) * lanesPerBlock
      proc makeLaneTask(index: int): SimpleThread =
        ## Preserve lane affinity and restore its identity at every resume.
        let
          blockIndex = index div lanesPerBlock
          laneIndex = index mod lanesPerBlock
          bz = uint(blockIndex div blocksPerPlane)
          by = uint((blockIndex mod blocksPerPlane) div int(blockGrid.x))
          bx = uint(blockIndex mod int(blockGrid.x))
          tz = uint(laneIndex div lanesPerPlane)
          ty = uint((laneIndex mod lanesPerPlane) div int(blockSize.x))
          tx = uint(laneIndex mod int(blockSize.x))
          state = addr blockStates[blockIndex]
          kernelIter = callWithTuple(fn, blockArgs)
        result = iterator(): bool {.gcsafe.} =
          while true:
            blockIdx = newDim3(bx, by, bz)
            threadIdx = newDim3(tx, ty, tz)
            gridDim = blockGrid
            blockDim = blockSize
            currentSimpleBlock = state
            if not kernelIter():
              simpleLaneFinished(state)
              break
            yield true
      var index = workerIndex
      while index < count:
        result.add(makeLaneTask(index))
        index += workerCount

    var serialFailure: SimpleFailure
    try:
      when not SingleThread:
        if threads > 1 and totalLanes > 1:
          proc runWorker(index, count: int; cancelled: ptr Atomic[bool]) {.gcsafe.} =
            ## Keep iterator allocation, execution and destruction on the same worker.
            let previous = currentSimpleBlock
            try:
              runAllTasks(makeWorkerTasks(index, count, launchArgs,
                launchGrid, launchBlock, states), cancelled)
            finally:
              currentSimpleBlock = previous
          runParallelLanes(runWorker, totalLanes)
        else:
          runAllTasks(makeWorkerTasks(0, 1, launchArgs, launchGrid, launchBlock, states))
      else:
        runAllTasks(makeWorkerTasks(0, 1, launchArgs, launchGrid, launchBlock, states))
    except ValueError, Defect, SimpleKernelError:
      raise
    except Exception as error:
      serialFailure = describeFailure(error)
    finally:
      currentSimpleBlock = previousBlock
      for index in 0 ..< totalBlocks:
        var slot = states[index].shared
        while slot != nil:
          let next = slot.next
          simpleRelease(slot.allocation)
          simpleRelease(slot)
          slot = next
        when not SingleThread:
          deinitLock(states[index].lock)
      deallocShared(states)
    # Restore launch state before propagating a replacement exception.
    if serialFailure.name != nil:
      raiseFailure(serialFailure)

template hippoSyncthreads*() =
  ## Release worker capacity until all logical lanes in this block arrive.
  block:
    let state = currentSimpleBlock
    if state == nil:
      raise newException(ValueError, "SIMPLE barriers require an active kernel block.")
    let generation = simpleArrive(state)
    while simpleBarrierPending(state, generation):
      yield true


# Warp shuffle stubs — no-op in SIMPLE backend (single-threaded per "warp")
proc shflDown*(val: cfloat, delta: cint): cfloat = val
  ## Warp shuffle down stub; returns val unchanged.
proc shflDown*(val: cint, delta: cint): cint = val
  ## Warp shuffle down stub; returns val unchanged.
proc shfl*(val: cfloat, srcLane: cint): cfloat = val
  ## Warp shuffle stub; returns val unchanged.
proc shfl*(val: cint, srcLane: cint): cint = val
  ## Warp shuffle stub; returns val unchanged.
proc shflDown*(val: cuint, delta: cint): cuint = val
  ## Warp shuffle down stub; returns val unchanged.
proc shflDown*(val: cfloat, delta: cint, width: cint): cfloat = val
  ## Warp shuffle down stub with width; returns val unchanged.
proc shflDown*(val: cint, delta: cint, width: cint): cint = val
  ## Warp shuffle down stub with width; returns val unchanged.
proc shflDown*(val: cuint, delta: cint, width: cint): cuint = val
  ## Warp shuffle down stub with width; returns val unchanged.
proc shflXor*(val: cfloat, laneMask: cint): cfloat = val
  ## Warp shuffle xor stub; returns val unchanged.
proc shflXor*(val: cint, laneMask: cint): cint = val
  ## Warp shuffle xor stub; returns val unchanged.

const WarpSize* = 1
  ## SIMPLE backend has no real warps.

# Half-precision (float16) conversion — software fallback via bit manipulation
proc halfToFloat*(h: uint16): cfloat =
  ## Convert IEEE 754 half-precision (uint16) to float32.
  let s = uint32(h shr 15)
  let e = uint32((h shr 10) and 0x1F'u16)
  let f = uint32(h and 0x3FF'u16)
  if e == 0'u32:
    if f == 0'u32:
      return cast[cfloat](s shl 31)
    # Subnormal: val = ±f * 2^-24
    let fBits = (s shl 31) or ((127'u32 - 24) shl 23)
    return cast[cfloat](fBits) * cfloat(f)
  elif e == 31'u32:
    # Inf/NaN → preserve sign, set float32 inf/nan exponent
    let bits = (s shl 31) or 0x7F800000'u32 or (f shl 13)
    return cast[cfloat](bits)
  let bits = (s shl 31) or ((e + 112) shl 23) or (f shl 13)
  cast[cfloat](bits)

proc floatToHalf*(f: cfloat): uint16 =
  ## Convert float32 to IEEE 754 half-precision (uint16).
  let bits = cast[uint32](f)
  let s = uint16((bits shr 16) and 0x8000'u32)
  let e = int((bits shr 23) and 0xFF'u32) - 127 + 15
  let m = bits and 0x7FFFFF'u32
  if e <= 0:
    return s  # flush to ±zero
  if e >= 31:
    return s or 0x7C00'u16  # ±infinity
  s or uint16(e shl 10) or uint16(m shr 13)

proc loadU32*(p: ptr uint8): uint32 =
  ## Safe uint32 load from a byte pointer.
  copyMem(addr result, p, 4)

# Math functions matching HIP/CUDA interface
# Single-precision floating-point math functions
template sinf*(x: cfloat): cfloat =
  ## Single-precision sine function
  math.sin(x.float).cfloat

template cosf*(x: cfloat): cfloat =
  ## Single-precision cosine function
  math.cos(x.float).cfloat

# Double-precision floating-point math functions
template sin*(x: cdouble): cdouble =
  ## Double-precision sine function
  math.sin(x.float64).cdouble

template cos*(x: cdouble): cdouble =
  ## Double-precision cosine function
  math.cos(x.float64).cdouble
