import
  hippo,
  std/unittest,
  ./utils

# isolated test to check if shared memory is working without relying on syncthreads

const
  N = 33 * 1024
  ThreadsPerBlock: int = 256
  BlocksPerGrid: int = min(32, ((N + ThreadsPerBlock - 1) div ThreadsPerBlock))

proc dot(a, b, partial_c: ptr[float64]){.hippoGlobal.} =
  let aArray = cast[ptr UncheckedArray[float64]](a)
  let bArray = cast[ptr UncheckedArray[float64]](b)
  let cArray = cast[ptr UncheckedArray[float64]](partial_c)

  var sdata {.hippoShared.}: array[ThreadsPerBlock, float64]  # Shared memory array

  let tid = threadIdx.x + blockIdx.x * blockDim.x
  var local_tid = threadIdx.x
  var temp: float64 = 0
  var idx = tid
  while idx < N:
    temp = temp + (aArray[idx] * bArray[idx])
    idx += blockDim.x * gridDim.x

  # Store partial sum in shared memory at thread's own index (no inter-thread dependency)
  sdata[local_tid] = temp

  # Immediately read back own value and write to global partials (no sync needed)
  cArray[tid] = sdata[local_tid]

suite "dot product with shared memory but no syncthreads":
  testSkipPlatforms "dot", "SIMPLE":
    var a, b: array[N, float64]
    var partial_c: array[N, float64]  # Now per-thread partials since no block reduction

    var dev_a = hippoMalloc(sizeof(float64)*N)
    var dev_b = hippoMalloc(sizeof(float64)*N)
    var dev_partial_c = hippoMalloc(N * sizeof(float64))  # Expanded for per-thread outputs

    for i in 0 ..< N:
      a[i] = i.float
      b[i] = (i * 2).float

    hippoMemcpy(dev_a, addr a[0], sizeof(float64)*N, HippoMemcpyHostToDevice)
    hippoMemcpy(dev_b, addr b[0], sizeof(float64)*N, HippoMemcpyHostToDevice)

    hippoLaunchKernel(
      dot,
      gridDim = newDim3(BlocksPerGrid.uint32),
      blockDim = newDim3(ThreadsPerBlock.uint32),
      args = hippoArgs(dev_a.p, dev_b.p, dev_partial_c.p)
    )

    hippoMemcpy(addr partial_c[0], dev_partial_c, N * sizeof(float64), HippoMemcpyDeviceToHost)

    var c: float64 = 0
    for i in 0 ..< N:
      c += partial_c[i]

    proc sum_squares(x: float64): float64 =
      result = x * (x + 1) * (2 * x + 1) / 6

    let expected = 2 * sum_squares((N - 1).float)
    assertAlmostEqual(c, expected)
