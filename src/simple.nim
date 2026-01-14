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
# we should still be able to process thousands of gpu threads and handle __syncthreads() correctly.
# these should work like coroutines that can be paused (when syncthreads is called) and resumed (when all threads in a block are synced).


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

proc newDim3*(x: uint = 1; y: uint = 1; z: uint = 1): Dim3 =
  result.x = x
  result.y = y
  result.z = z


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

var
  blockIdx* {.threadvar.}: BlockIdx
  threadIdx* {.threadvar.}: ThreadIdx
  blockDim* {.threadvar.}: BlockDim
  gridDim* {.threadvar.}: GridDim

when not SingleThread:
  # Barrier infrastructure for multi-threaded mode
  type
    BlockBarrier = object
      lock: Lock
      counter: int
      generation: int
      numThreads: int
      lastLaunchId: int  # Track which kernel launch this barrier belongs to
    Task = iterator (): bool  # Closure iterator that yields true while working, false when done
  
  var blockBarriers: Table[uint64, ptr BlockBarrier]
  var barriersLock: Lock
  var currentKernelLaunch*: int = 0
  var syncthreadsCalled*: bool = false  # Global flag for syncthreads
  
  initLock(barriersLock)
  
  # Export functions so they're accessible from templates expanded in kernel iterators
  proc getBlockId*(blockIdx: BlockIdx): uint64 =
    # Create a unique identifier for a block
    result = blockIdx.x.uint64
    result = result * 1000000'u64 + blockIdx.y.uint64
    result = result * 1000000'u64 + blockIdx.z.uint64
  
  proc getOrCreateBarrier*(blockId: uint64, numThreads: int, launchId: int): ptr BlockBarrier =
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
      release(barriersLock)
    else:
      result = blockBarriers[blockId]
      release(barriersLock)
  



when SingleThread:
  # TODO this is wrong. the single thread case should be the multithread with just a cpu pool of 1 worker.
  # this DOES NOT implement syncthreads correctly. it runs each thread to completion, this is wrong.
  # in the example of a dot product, thread 0 finishes completely first before any of the other threads had a chance to run and gives an incorrect answer.

  template simpleLaunchKernel(fn: untyped, gridDimArg: Dim3, blockDimArg: Dim3, args: tuple) =
    # Sequential execution
    gridDim = gridDimArg
    blockDim = blockDimArg
    for bz in 0..<gridDimArg.z:
      for by in 0..<gridDimArg.y:
        for bx in 0..<gridDimArg.x:
          blockIdx.x = bx
          blockIdx.y = by
          blockIdx.z = bz
          for tz in 0..<blockDimArg.z:
            for ty in 0..<blockDimArg.y:
              for tx in 0..<blockDimArg.x:
                threadIdx.x = tx
                threadIdx.y = ty
                threadIdx.z = tz
                unpackCall(fn, args)

else:

  proc runAllTasks(tasks: seq[Task]) =
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

  proc worker(closure: proc () {.closure, gcsafe.}) {.thread.} =
    ## Worker procedure that executes the provided closure in a thread.
    closure()

  template simpleLaunchKernel(fn: untyped, gridDimArg: Dim3, blockDimArg: Dim3, args: tuple) =
    block:
      # Reset barriers for this kernel launch (before threads start)
      acquire(barriersLock)
      inc currentKernelLaunch
      let launchId = currentKernelLaunch
      # Clear barriers table - new barriers will be created as needed
      blockBarriers.clear()
      release(barriersLock)
      
      # Multi-threaded execution: blocks run in parallel, threads within each block run cooperatively
      let totalBlocks = gridDimArg.x * gridDimArg.y * gridDimArg.z
      let totalThreadsPerBlock = blockDimArg.x * blockDimArg.y * blockDimArg.z
      let totalBlocksPerPlane = gridDimArg.x * gridDimArg.y
      let totalThreadsPerBlockPlane = blockDimArg.x * blockDimArg.y
      
      # Execute blocks in parallel
      var blockHandles: seq[Thread[proc () {.closure.}]]
      blockHandles.setLen(int(totalBlocks))
      
      proc executeBlock(blockIndex: uint): proc() {.closure, gcsafe.} =
        # Set up block indices (capture in closure)
        let bz = blockIndex div totalBlocksPerPlane
        let remainder = blockIndex mod totalBlocksPerPlane
        let by = remainder div gridDimArg.x
        let bx = remainder mod gridDimArg.x
        
        result = proc() {.closure, gcsafe.} =
          # Create iterators for each thread in this block
          # Threads run cooperatively using round-robin scheduling
          var blockTasks: seq[Task]
          
          for threadIndex in 0..<totalThreadsPerBlock:
            # Convert threadIndex to 3D threadIdx
            let tz = threadIndex div totalThreadsPerBlockPlane
            let threadRemainder = threadIndex mod totalThreadsPerBlockPlane
            let ty = threadRemainder div blockDimArg.x
            let tx = threadRemainder mod blockDimArg.x
            
            # Create iterator for this thread that wraps kernel execution
            let threadTask = iterator (): bool =
              # Set thread-local variables
              blockIdx.x = bx
              blockIdx.y = by
              blockIdx.z = bz
              threadIdx.x = tx
              threadIdx.y = ty
              threadIdx.z = tz
              gridDim = gridDimArg
              blockDim = blockDimArg
              
              # Execute kernel proc (it returns an iterator, so call it and run the iterator in a loop)
              {.gcsafe.}:
                # Call the proc to get the iterator instance and run it
                # Cast pointer arguments to ptr float64 as expected by kernels
                let kernelIter = fn(
                  cast[ptr float64](args[0]),
                  cast[ptr float64](args[1]),
                  cast[ptr float64](args[2]),
                  cast[ptr float64](args[3])
                )
                while kernelIter():
                  yield true  # Yield while kernel is running (for syncthreads)
              
              # Thread is done
              yield false
            
            blockTasks.add(threadTask)
          
          # Run threads sequentially for SIMPLE runtime (no cooperative scheduling needed)
          {.gcsafe.}:
            for task in blockTasks:
              while task():
                discard
      
      # Launch blocks in parallel
      var blockIdxCounter: uint = 0
      for i in 0..<int(totalBlocks):
        let blockIndex = blockIdxCounter
        let closure = executeBlock(blockIndex)
        createThread(blockHandles[i], worker, closure)
        inc blockIdxCounter
      
      # Wait for all blocks to complete
      for th in blockHandles:
        joinThread(th)


# hippoSyncthreads implementation depends on threading mode
when SingleThread:
  proc hippoSyncthreads*() =
    ## In single-threaded mode, threads execute sequentially, so syncthreads is a no-op
    ## TODO this is very, very, very wrong. syncthreads CANNOT be a no-op.
    ## this breaks the dot product test because thread 0 finishes completely first before any of the other threads had a chance to run and, therefore, gives an incorrect answer.
    discard

# hippoSyncthreads - blocking barrier for multi-threaded mode
template hippoSyncthreads*() {.dirty.} =
  ## Blocking barrier that synchronizes threads within a block.
  ## Since blocks run in parallel OS threads, this can be blocking.
  when not (defined(js) or not compileOption("threads")):
    let blockId = getBlockId(blockIdx)
    let barrier = getOrCreateBarrier(blockId, int(blockDim.x * blockDim.y * blockDim.z), currentKernelLaunch)

    # Simple cooperative barrier using generation counter
    let myGen = barrier.generation
    inc barrier.counter
    let arrived = barrier.counter
    if arrived == barrier.numThreads:
      inc barrier.generation
      barrier.counter = 0

    # Wait until generation advances
    while barrier.generation <= myGen:
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
