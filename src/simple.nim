## `simple` is a pure-nim backend for hippo.
## You must compile with --threads:on for using more gpu cores.
## Otherwise it will only use 1 core.
import std/[cpuinfo, macros]

const SingleThread = defined(js) or not compileOption("threads")

when SingleThread:
  echo "singlethread"
  {.warning: "hippo/simple: Compiled without --threads:on, performance will be limited to 1 core".}
else:
  echo "multithread"
  {.warning: "hippo/simple: Compiling with --threads:on".}

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
  p[] = alloc(size)

proc simpleMemcpy(dst: pointer, src: pointer, size: int, kind: HippoMemcpyKind) =
  copyMem(dst, src, size)

proc simpleFree(p: pointer) =
  dealloc(p)

proc handleError(err: HippoError) =
  ## Simple runtime raises errors as exceptions.
  discard

var
  blockIdx* {.threadvar.}: BlockIdx
  threadIdx* {.threadvar.}: ThreadIdx

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

  template simpleLaunchKernel(fn: untyped, gridDim: Dim3, blockDim: Dim3, args: tuple) =
    # Sequential execution
    for bz in 0..<gridDim.z:
      for by in 0..<gridDim.y:
        for bx in 0..<gridDim.x:
          blockIdx.x = bx
          blockIdx.y = by
          blockIdx.z = bz
          for tz in 0..<blockDim.z:
            for ty in 0..<blockDim.y:
              for tx in 0..<blockDim.x:
                threadIdx.x = tx
                threadIdx.y = ty
                threadIdx.z = tz
                unpackCall(fn, args)

else:

  proc worker(closure: proc () {.closure.}) {.thread.} =
    ## Worker procedure that executes the provided closure in a thread.
    {.gcsafe.}:
      closure()

  template simpleLaunchKernel(fn: untyped, gridDim: Dim3, blockDim: Dim3, args: tuple) =
    # Multi-threaded execution
    let totalBlocks = gridDim.x * gridDim.y * gridDim.z
    let blocksPerThread = totalBlocks div threads
    let extraBlocks = totalBlocks mod threads

    var threadHandles: seq[Thread[proc () {.closure.}]]
    threadHandles.setLen(threads)
    var startBlock = 0.uint

    for i in 0..<threads.uint:
      let numBlocks = if i < extraBlocks: blocksPerThread + 1 else: blocksPerThread
      let endBlock = startBlock + numBlocks
      proc threadClosure() {.closure.} =
        for bz in 0..<gridDim.z:
          for by in 0..<gridDim.y:
            for bx in 0..<gridDim.x:
              let blockIndex = bx + by * gridDim.x + bz * gridDim.x * gridDim.y
              if blockIndex >= startBlock and blockIndex < endBlock:
                blockIdx.x = bx
                blockIdx.y = by
                blockIdx.z = bz
                for tz in 0..<blockDim.z:
                  for ty in 0..<blockDim.y:
                    for tx in 0..<blockDim.x:
                      threadIdx.x = tx
                      threadIdx.y = ty
                      threadIdx.z = tz
                      unpackCall(fn, args)
      createThread(threadHandles[i], worker, threadClosure)
      startBlock = endBlock

    for th in threadHandles:
      joinThread(th)


proc hippoSyncthreads*() =
  # could probably use nim iterators to implement syncthreads?
  # I want something simple that can work anywhere
  raise newException(Exception, "hippoSyncthreads not implemented yet")
