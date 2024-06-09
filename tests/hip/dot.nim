import hippo

# GPU Dot product

const 
  N = 33 * 1024
  ThreadsPerBlock: int = 256
  BlocksPerGrid: int = min(32, ((N + ThreadsPerBlock - 1) div ThreadsPerBlock))

proc dot(a, b, c: ptr[cfloat]){.hippoGlobal.} =
  
  var cache {.hippoShared.}: ptr[cfloat]

  # TODO figure out how to do this properly
  let aArray = cast[ptr UncheckedArray[cfloat]](a)
  let bArray = cast[ptr UncheckedArray[cfloat]](b)
  let cArray = cast[ptr UncheckedArray[cfloat]](c)
  let cacheArray = cast[ptr UncheckedArray[cfloat]](cache)

  let cacheIndex = threadIdx.x
  var tid = threadIdx.x + blockIdx.x * blockDim.x
  var temp: cfloat = 0
  while tid < N:
    temp += aArray[tid] * bArray[tid]
    tid += blockDim.x * gridDim.x
  
  # set the cache values
  cacheArray[cacheIndex] = temp

  # synchronize threads in this block
  hippoSyncthreads()

  # for reductions, threadsPerBlock must be a power of 2
  # because of the following code
  var i = blockDim.x div 2
  while i != 0:
    if cacheIndex < i:
      cacheArray[cacheIndex] += cacheArray[cacheIndex + i]
    hippoSyncthreads()
    i = i div 2

  if cacheIndex == 0:
    cArray[blockIdx.x] = cacheArray[0]


proc main() =
  var a, b, partial_c: array[N, int32]
  var dev_a, dev_b, dev_partial_c: pointer

  # allocate gpu memory
  handleError(hipMalloc(addr dev_a, sizeof(float)*N))
  handleError(hipMalloc(addr dev_b, sizeof(float)*N))
  handleError(hipMalloc(addr dev_partial_c, BlocksPerGrid * sizeof(float)))

  # fill in host memory with data
  for i in 0 ..< N:
    a[i] = i.int32
    b[i] = (i * 2).int32

  # copy data to device
  handleError(hipMemcpy(dev_a, addr a[0], sizeof(float)*N, hipMemcpyHostToDevice))
  handleError(hipMemcpy(dev_b, addr b[0], sizeof(float)*N, hipMemcpyHostToDevice))

  # launch kernel
  handleError(launchKernel(
    dot,
    gridDim = newDim3(BlocksPerGrid.uint32),
    blockDim = newDim3(ThreadsPerBlock.uint32),
    args = (dev_a, dev_b, dev_partial_c)
  ))

  # copy memory back from GPU to CPU
  handleError(hipMemcpy(addr partial_c[0], dev_partial_c, BlocksPerGrid * sizeof(float), hipMemcpyDeviceToHost))

  # finish up on the CPU
  var c: int32 = 0
  for i in 0 ..< BlocksPerGrid:
    c += partial_c[i]


  proc sum_squares(x: float): float =
    result = x * (x + 1) * (2 * x + 1) / 6
  
  echo "Does GPU value ", c.float, " = ", 2 * sum_squares((N - 1)), "?"

  handleError(hipFree(dev_a))
  handleError(hipFree(dev_b))
  handleError(hipFree(dev_partial_c))


when isMainModule:
  main()