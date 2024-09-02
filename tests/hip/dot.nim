import hippo, std/strformat

# GPU Dot product

const 
  N = 33 * 1024
  ThreadsPerBlock: int = 256
  BlocksPerGrid: int = min(32, ((N + ThreadsPerBlock - 1) div ThreadsPerBlock))

proc dot(a, b, c: ptr[float64]){.hippoGlobal.} =
  var cache {.hippoShared.}: array[256, float]

  # TODO figure out how to do this properly
  let aArray = cast[ptr UncheckedArray[float64]](a)
  let bArray = cast[ptr UncheckedArray[float64]](b)
  let cArray = cast[ptr UncheckedArray[float64]](c)

  let cacheIndex = threadIdx.x
  var tid = threadIdx.x + blockIdx.x * blockDim.x
  var temp: float64 = 0
  while tid < N:
    # TODO not sure how to handle functions like `pluseq___dot_u17` with nim / cuda
    temp = temp + (aArray[tid] * bArray[tid])
    tid += blockDim.x * gridDim.x
  
  # set the cache values
  cache[cacheIndex] = temp

  # synchronize threads in this block
  hippoSyncthreads()

  # for reductions, threadsPerBlock must be a power of 2
  # because of the following code
  var i = blockDim.x div 2
  while i != 0:
    if cacheIndex < i:
      # TODO not sure how to handle functions like `pluseq___dot_u17` with nim / cuda
      cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i]
    hippoSyncthreads()
    i = i div 2

  if cacheIndex == 0:
    cArray[blockIdx.x] = cache[0]


proc main() =
  var a, b, partial_c: array[N, float64]
  var dev_a, dev_b, dev_partial_c: pointer

  # allocate gpu memory
  handleError(hipMalloc(addr dev_a, sizeof(float64)*N))
  handleError(hipMalloc(addr dev_b, sizeof(float64)*N))
  handleError(hipMalloc(addr dev_partial_c, BlocksPerGrid * sizeof(float64)))

  # fill in host memory with data
  for i in 0 ..< N:
    a[i] = i.float
    b[i] = (i * 2).float

  # copy data to device
  handleError(hipMemcpy(dev_a, addr a[0], sizeof(float64)*N, hipMemcpyHostToDevice))
  handleError(hipMemcpy(dev_b, addr b[0], sizeof(float64)*N, hipMemcpyHostToDevice))

  # launch kernel
  hippoLaunchKernel(
    dot,
    gridDim = newDim3(BlocksPerGrid.uint32),
    blockDim = newDim3(ThreadsPerBlock.uint32),
    args = (dev_a, dev_b, dev_partial_c)
  )

  # copy memory back from GPU to CPU
  handleError(hipMemcpy(addr partial_c[0], dev_partial_c, BlocksPerGrid * sizeof(float64), hipMemcpyDeviceToHost))

  # finish up on the CPU
  var c: float64 = 0
  for i in 0 ..< BlocksPerGrid:
    c += partial_c[i]

  proc sum_squares(x: float64): float64 =
    result = x * (x + 1) * (2 * x + 1) / 6
  
  echo fmt"Does GPU value {c:e} = {2 * sum_squares((N - 1)):e}?"

  handleError(hipFree(dev_a))
  handleError(hipFree(dev_b))
  handleError(hipFree(dev_partial_c))


when isMainModule:
  main()