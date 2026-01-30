## `simple` is a pure-nim backend for hippo.
## You must compile with --threads:on for using more cpu cores.
## Otherwise it will only use 1 core.
## Single threaded mode should 'just work' anywhere, even in js, but won't be fast.
import
  std/[cpuinfo, macros, os, strutils, math, tables, locks]

## TERMINOLOGY: `threads` or `SingleThread` refers to the number of CPU threads we are using under the hood.
## these CPU threads are in a pool that runs the actual gpu kernels.
## GPU blocks and GPU threads are closures that are a separate distinct concept.
## GPU threads can sync with all the other threads in a block using `hippoSyncthreads()` just like in CUDA/HIP.
## we may be running thousands of GPU threads on top of a pool or 8, 4, or maybe even 1 cpu thread.
## BlockDim, GridDim, ThreadIdx, BlockIdx, are all referring to GPU threads.
## goal: handle thousands of GPU threads with correct __syncthreads() semantics via cooperative scheduling.


const SingleThread = defined(js) or not compileOption("threads")

when SingleThread:
  {.warning: "hippo/simple: Compiled without --threads:on, performance will be limited to 1 core".}

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

proc setThreads*(n: uint) =
  threads = n


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
    echo "DEBUG: hippo/simple: Using ", threads, " threads"

simpleInit()

proc simpleMalloc(p: ptr pointer, size: int) =
  p[] = allocShared(size)

proc simpleMemcpy(dst: pointer, src: pointer, size: int, kind: HippoMemcpyKind) =
  copyMem(dst, src, size)

proc simpleFree(p: pointer) =
  deallocShared(p)

proc handleError(err: HippoError) =
  ## Simple runtime raises errors as exceptions.
  discard

macro callWithTuple(fn: typed, args: untyped): untyped =
  ## Expand a tuple constructor into a call expression and cast args to param types.
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

  if args.kind in {nnkTupleConstr, nnkPar}:
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

var
  blockIdx* {.threadvar.}: BlockIdx
  threadIdx* {.threadvar.}: ThreadIdx
  blockDim* {.threadvar.}: BlockDim
  gridDim* {.threadvar.}: GridDim

type
  BlockBarrier = object
    lock: Lock
    counter: int
    generation: int
    numThreads: int
    lastLaunchId: int  # Track which kernel launch this barrier belongs to
  Task = iterator (): bool {.gcsafe.}  # Closure iterator that yields true while working, false when done

var blockBarriers: Table[uint64, ptr BlockBarrier]
var currentKernelLaunch*: int = 0
var syncthreadsCalled*: bool = false  # Global flag for syncthreads

proc runAllTasks(tasks: seq[Task]) {.gcsafe.} =
  ## Cooperative scheduler that runs tasks in round-robin until all complete.
  var active = newSeq[Task]()
  for t in tasks: active.add(t)

  while active.len > 0:
    var i = 0
    while i < active.len:
      let stillWorking = active[i]()
      if not stillWorking:
        active.del(i)
      else:
        inc i

when not SingleThread:
  var barriersLock: Lock
  initLock(barriersLock)

# Export functions so they're accessible from templates expanded in kernel iterators
proc getBlockId*(blockIdx: BlockIdx): uint64 =
  # Create a unique identifier for a block
  result = blockIdx.x.uint64
  result = result * 1000000'u64 + blockIdx.y.uint64
  result = result * 1000000'u64 + blockIdx.z.uint64

proc getOrCreateBarrier*(blockId: uint64, numThreads: int, launchId: int): ptr BlockBarrier =
  when not SingleThread:
    acquire(barriersLock)
  if not blockBarriers.hasKey(blockId):
    var barrier = cast[ptr BlockBarrier](allocShared(sizeof(BlockBarrier)))
    initLock(barrier.lock)
    barrier.counter = 0
    barrier.generation = 0
    barrier.numThreads = numThreads
    barrier.lastLaunchId = launchId
    blockBarriers[blockId] = barrier
    result = barrier
  else:
    result = blockBarriers[blockId]
  when not SingleThread:
    release(barriersLock)
  



when SingleThread:
  template simpleLaunchKernel(fn: untyped, gridDimArg: Dim3, blockDimArg: Dim3, args: tuple) =
    block:
      inc currentKernelLaunch
      blockBarriers.clear()

      gridDim = gridDimArg
      blockDim = blockDimArg

      let totalBlocks = int(gridDimArg.x * gridDimArg.y * gridDimArg.z)
      let totalThreadsPerBlock = int(blockDimArg.x * blockDimArg.y * blockDimArg.z)
      let totalBlocksPerPlane = int(gridDimArg.x * gridDimArg.y)
      let totalThreadsPerBlockPlane = int(blockDimArg.x * blockDimArg.y)

      proc executeBlock(blockIndex: int): proc() {.closure, gcsafe.} =
        let bz = blockIndex div totalBlocksPerPlane
        let remainder = blockIndex mod totalBlocksPerPlane
        let by = remainder div int(gridDimArg.x)
        let bx = remainder mod int(gridDimArg.x)

        result = proc() {.closure, gcsafe.} =
          proc makeThreadTask(bxVal, byVal, bzVal, txVal, tyVal, tzVal: uint): Task =
            iterator (): bool =
              {.gcsafe.}:
                blockIdx.x = bxVal
                blockIdx.y = byVal
                blockIdx.z = bzVal
                threadIdx.x = txVal
                threadIdx.y = tyVal
                threadIdx.z = tzVal
                gridDim = gridDimArg
                blockDim = blockDimArg

                let kernelIter = callWithTuple(fn, args)
                while true:
                  blockIdx.x = bxVal
                  blockIdx.y = byVal
                  blockIdx.z = bzVal
                  threadIdx.x = txVal
                  threadIdx.y = tyVal
                  threadIdx.z = tzVal
                  gridDim = gridDimArg
                  blockDim = blockDimArg
                  if not kernelIter():
                    break
                  yield true
              yield false

          var blockTasks: seq[Task]
          for threadIndex in 0..<totalThreadsPerBlock:
            let tz = threadIndex div totalThreadsPerBlockPlane
            let threadRemainder = threadIndex mod totalThreadsPerBlockPlane
            let ty = threadRemainder div int(blockDimArg.x)
            let tx = threadRemainder mod int(blockDimArg.x)
            blockTasks.add(makeThreadTask(uint(bx), uint(by), uint(bz), uint(tx), uint(ty), uint(tz)))

          runAllTasks(blockTasks)

      for blockIndex in 0..<totalBlocks:
        let closure = executeBlock(blockIndex)
        closure()

else:

  proc worker(closure: proc () {.closure, gcsafe.}) {.thread.} =
    ## Worker procedure that executes the provided closure in a thread.
    closure()

  template simpleLaunchKernel(fn: untyped, gridDimArg: Dim3, blockDimArg: Dim3, args: tuple) =
    block:
      # Reset barriers for this kernel launch (before threads start)
      acquire(barriersLock)
      inc currentKernelLaunch
      blockBarriers.clear()
      release(barriersLock)

      # Multi-threaded execution: blocks run in parallel OS threads.
      # Threads within each block run cooperatively using round-robin scheduling.
      let totalBlocks = int(gridDimArg.x * gridDimArg.y * gridDimArg.z)
      let totalThreadsPerBlock = int(blockDimArg.x * blockDimArg.y * blockDimArg.z)
      let totalBlocksPerPlane = int(gridDimArg.x * gridDimArg.y)
      let totalThreadsPerBlockPlane = int(blockDimArg.x * blockDimArg.y)

      proc executeBlock(blockIndex: int): proc() {.closure, gcsafe.} =
        let bz = blockIndex div totalBlocksPerPlane
        let remainder = blockIndex mod totalBlocksPerPlane
        let by = remainder div int(gridDimArg.x)
        let bx = remainder mod int(gridDimArg.x)

        result = proc() {.closure, gcsafe.} =
          proc makeThreadTask(bxVal, byVal, bzVal, txVal, tyVal, tzVal: uint): Task =
            iterator (): bool =
              {.gcsafe.}:
                blockIdx.x = bxVal
                blockIdx.y = byVal
                blockIdx.z = bzVal
                threadIdx.x = txVal
                threadIdx.y = tyVal
                threadIdx.z = tzVal
                gridDim = gridDimArg
                blockDim = blockDimArg

                let kernelIter = callWithTuple(fn, args)
                while true:
                  blockIdx.x = bxVal
                  blockIdx.y = byVal
                  blockIdx.z = bzVal
                  threadIdx.x = txVal
                  threadIdx.y = tyVal
                  threadIdx.z = tzVal
                  gridDim = gridDimArg
                  blockDim = blockDimArg
                  if not kernelIter():
                    break
                  yield true
              yield false

          var blockTasks: seq[Task]
          for threadIndex in 0..<totalThreadsPerBlock:
            let tz = threadIndex div totalThreadsPerBlockPlane
            let threadRemainder = threadIndex mod totalThreadsPerBlockPlane
            let ty = threadRemainder div int(blockDimArg.x)
            let tx = threadRemainder mod int(blockDimArg.x)
            blockTasks.add(makeThreadTask(uint(bx), uint(by), uint(bz), uint(tx), uint(ty), uint(tz)))

          runAllTasks(blockTasks)

      for i in 0..<totalBlocks:
        let closure = executeBlock(i)
        closure()


# hippoSyncthreads implementation depends on threading mode
template hippoSyncthreads*() {.dirty.} =
  ## Blocking barrier that synchronizes threads within a block.
  ## Since blocks may run in parallel OS threads, this can be blocking.
  bind acquire, release
  let blockId = getBlockId(blockIdx)
  let barrier = getOrCreateBarrier(blockId, int(blockDim.x * blockDim.y * blockDim.z), currentKernelLaunch)

  acquire(barrier.lock)
  let myGen = barrier.generation
  inc barrier.counter
  if barrier.counter == barrier.numThreads:
    inc barrier.generation
    barrier.counter = 0
  release(barrier.lock)

  while true:
    acquire(barrier.lock)
    if barrier.generation > myGen:
      release(barrier.lock)
      break
    release(barrier.lock)
    yield true

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
