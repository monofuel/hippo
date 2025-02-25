## `simple` is a pure-nim backend for hippo.
## You must compile with --threads:on for using more gpu cores.
## Otherwise it will only use 1 core.
import std/[cpuinfo, macros]

const SingleThread = defined(js) or not defined(threadsOn)

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


var threads = 1

proc setThreads*(n: int) =
  threads = n


proc simpleInit() =
  ## get the number of cpu cores and set the number of threads to use.
  when SingleThread:
    threads = countProcessors()
  else:
    threads = 1

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

# TODO thread local variables
var
  blockIdx*: BlockIdx
  threadIdx*: ThreadIdx

macro unpackCall(fn: untyped, args: untyped): untyped =
  ## Unpack the tuple and call the function with individual arguments, forcing type casting.
  #TODO this would be better without type casting but computers are hard.
  let fnType = fn.getTypeInst()
  assert fnType.kind == nnkProcTy, "Expected a procedure type"
  let params = fnType[0]
  result = newCall(fn)
  for i in 1..<params.len:
    let paramType = params[i][1]
    let argExpr = newTree(nnkBracketExpr, args, newLit(i - 1))
    let castedArg = newTree(nnkCast, paramType, argExpr)
    result.add(castedArg)

# Updated template for the CPU backend kernel launch
template simpleLaunchKernel(fn: untyped, gridDim: Dim3, blockDim: Dim3, args: tuple) =
  for bx in 0..<gridDim.x:
    blockIdx.x = bx.uint32
    for tx in 0..<blockDim.x:
      threadIdx.x = tx.uint32
      unpackCall(fn, args)



proc hippoSyncthreads*() =
  # could probably use nim iterators to implement syncthreads?
  # I want something simple that can work anywhere
  raise newException(Exception, "hippoSyncthreads not implemented yet")
