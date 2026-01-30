import
  hippo,
  std/[unittest, strformat],
  ./utils

# Isolated syncthreads test that does not involve shared memory.
# thread 0 reduces the partial sums from all threads in the block, and writes the result to the global array.

const
  N = 33 * 32
  ThreadsPerBlock: int = 1  # TODO this is wrong, the whole point of this is to test many gpu threads
  BlocksPerGrid: int = min(32, ((N + ThreadsPerBlock - 1) div ThreadsPerBlock))

proc dot(a, b, c, scratch: ptr[float64]){.hippoGlobal.} =
  let aArray = cast[ptr UncheckedArray[float64]](a)
  let bArray = cast[ptr UncheckedArray[float64]](b)
  let cArray = cast[ptr UncheckedArray[float64]](c)
  let scratchArray = cast[ptr UncheckedArray[float64]](scratch)

  let cacheIndex = threadIdx.x
  var tid = threadIdx.x + blockIdx.x * blockDim.x
  var temp: float64 = 0
  while tid < N:
    temp = temp + (aArray[tid] * bArray[tid])
    tid += blockDim.x * gridDim.x

  # Write to global scratch instead of shared memory
  scratchArray[blockIdx.x * blockDim.x + threadIdx.x] = temp

  # Synchronize to ensure all threads have written to global scratch
  hippoSyncthreads()

  # Thread 0 performs a linear sum over the block's partials in global scratch
  if cacheIndex == 0:
    var sum: float64 = 0
    for i in 0 ..< blockDim.x:
      sum = sum + scratchArray[blockIdx.x * blockDim.x + i]
    cArray[blockIdx.x] = sum

# Limit CPU threads for SIMPLE backend to make testing less intensive (otherwise it can hog all the cores)
when HippoRuntime == "SIMPLE":
  setThreads(4)

suite "syncthreads":
  test "syncthreads":
    var a, b, partial_c: array[N, float64]

    var dev_a = hippoMalloc(sizeof(float64)*N)
    var dev_b = hippoMalloc(sizeof(float64)*N)
    var dev_partial_c = hippoMalloc(BlocksPerGrid * sizeof(float64))
    var dev_scratch = hippoMalloc(BlocksPerGrid * ThreadsPerBlock * sizeof(float64))  # Global scratch space

    for i in 0 ..< N:
      a[i] = i.float
      b[i] = (i * 2).float

    hippoMemcpy(dev_a, addr a[0], sizeof(float64)*N, HippoMemcpyHostToDevice)
    hippoMemcpy(dev_b, addr b[0], sizeof(float64)*N, HippoMemcpyHostToDevice)

    hippoLaunchKernel(
      dot,
      gridDim = newDim3(BlocksPerGrid.uint32),
      blockDim = newDim3(ThreadsPerBlock.uint32),
      args = hippoArgs(dev_a.p, dev_b.p, dev_partial_c.p, dev_scratch.p)
    )

    hippoMemcpy(addr partial_c[0], dev_partial_c, BlocksPerGrid * sizeof(float64), HippoMemcpyDeviceToHost)

    var c: float64 = 0
    for i in 0 ..< BlocksPerGrid:
      c += partial_c[i]

    proc sum_squares(x: float64): float64 =
      result = x * (x + 1) * (2 * x + 1) / 6

    let expected = 2 * sum_squares((N - 1).float)
    assertAlmostEqual(c, expected)
