## `simple` is a pure-nim backend for hippo.
## You must compile with --threads:on for using more cpu cores.
## Otherwise it will only use 1 core.
## Single threaded mode should 'just work' anywhere, even in js, but won't be fast.
import std/[cpuinfo, macros]

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

macro unpackCall(fn: untyped, args: untyped): untyped =
  ## Unpack the tuple and call the function with individual arguments, forcing type casting.
  let fnType = fn.getTypeInst()
  assert fnType.kind == nnkProcTy, "Expected a procedure type"
  let params = fnType[0]
  result = newCall(fn)
  for i in 1..<params.len:
    let paramType = params[i][1]
    let argExpr = newTree(nnkBracketExpr, args, newLit(i - 1))
    let castedArg = newTree(nnkCast, paramType, argExpr)
    result.add(castedArg)

when SingleThread:

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

  # NB. currently we are only divving up blocks, but not threads.
  # if you execute a single block with many threads, it runs them all on one cpu thread
  # should improve that eventually, but computers are hard.
  # TODO a thread pool would be better, but `std/threadpool` is deprecated and I don't want to pull in dependencies.
  # There are some fantastic multithreading libraries for nim, but I'm feeling stubborn about not pulling in dependencies.

  proc worker(closure: proc () {.closure, gcsafe.}) {.thread.} =
    ## Worker procedure that executes the provided closure in a thread.
    closure()

  template simpleLaunchKernel(fn: untyped, gridDimArg: Dim3, blockDimArg: Dim3, args: tuple) =
    block:
      # Multi-threaded execution
      let totalBlocks = gridDimArg.x * gridDimArg.y * gridDimArg.z
      let blocksPerThread = totalBlocks div threads
      let extraBlocks = totalBlocks mod threads

      var threadHandles: seq[Thread[proc () {.closure.}]]
      threadHandles.setLen(threads)

      proc makeClosure(tid: uint, startBlock: uint, endBlock: uint): proc() {.closure, gcsafe.} =
        result = proc() {.closure, gcsafe.} =
          # echo "Thread ", tid, " startBlock=", startBlock, " endBlock=", endBlock
          gridDim = gridDimArg
          blockDim = blockDimArg
          let totalBlocksPerPlane = gridDimArg.x * gridDimArg.y
          for blockIndex in startBlock..<endBlock:
            let bz = blockIndex div totalBlocksPerPlane
            let remainder = blockIndex mod totalBlocksPerPlane
            let by = remainder div gridDimArg.x
            let bx = remainder mod gridDimArg.x
            blockIdx.x = bx
            blockIdx.y = by
            blockIdx.z = bz
            for tz in 0..<blockDimArg.z:
              for ty in 0..<blockDimArg.y:
                for tx in 0..<blockDimArg.x:
                  threadIdx.x = tx
                  threadIdx.y = ty
                  threadIdx.z = tz
                  # echo "threadId", getThreadId(), " Thread ", tid, " blockIdx=", blockIdx, " threadIdx=", threadIdx, " startBlock=", startBlock, " endBlock=", endBlock
                  {.gcsafe.}: # HACK
                    unpackCall(fn, args)

      var startBlock: uint = 0
      for i in 0..<threads.uint:
        let numBlocks = if i < extraBlocks: blocksPerThread + 1 else: blocksPerThread
        let myStartBlock = startBlock
        let myEndBlock = startBlock + numBlocks
        let closure = makeClosure(i, myStartBlock, myEndBlock)
        createThread(threadHandles[i], worker, closure)
        startBlock = myEndBlock

      for th in threadHandles:
        joinThread(th)


proc hippoSyncthreads*() =
  # could probably use nim iterators to implement syncthreads?
  # I want something simple that can work anywhere
  raise newException(Exception, "hippoSyncthreads not implemented yet")
