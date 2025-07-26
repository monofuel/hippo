import hippo, std/[unittest, strformat]

const 
  N = 33 * 1024
  ThreadsPerBlock: int = 256
  BlocksPerGrid: int = min(32, ((N + ThreadsPerBlock - 1) div ThreadsPerBlock))

# requires --stackTrace:off

{.emit: """
#ifdef __HIPCC__
  // Redefine N_INLINE before nimbase.h uses it
  #ifdef N_INLINE
    #undef N_INLINE
  #endif
  #define N_INLINE(rettype, name) __device__ __host__ inline rettype name
#endif
""".}

# Comprehensive device annotation for all Nim runtime functions
# {.emit: """
# #ifdef __HIPCC__
#   // Redefine ALL Nim calling conventions for device compatibility
#   #ifdef N_INLINE
#     #undef N_INLINE
#   #endif
#   #define N_INLINE(rettype, name) __device__ __host__ inline rettype name
  
#   #ifdef N_NOINLINE  
#     #undef N_NOINLINE
#   #endif
#   #define N_NOINLINE(rettype, name) __device__ __host__ rettype name
  
#   #ifdef N_NIMCALL
#     #undef N_NIMCALL
#   #endif
#   #define N_NIMCALL(rettype, name) __device__ __host__ rettype name
  
#   // Debug: Verify our redefinitions worked
#   #pragma message("HIP: Redefined Nim calling conventions for device compatibility")
# #endif
# """.}

# NOT TESTED
# my gtx 1070 gpu server is turned off and I'm lazy.
# {.emit: """
# #ifdef __CUDACC__
# #define N_INLINE(rettype, name) __device__ __host__ inline rettype name
# #endif
# """.}

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
    temp += aArray[tid] * bArray[tid]
    #temp = temp + (aArray[tid] * bArray[tid])
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
      cache[cacheIndex] += cache[cacheIndex + i]
      #cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i]
    hippoSyncthreads()
    i = i div 2

  if cacheIndex == 0:
    cArray[blockIdx.x] = cache[0]

suite "dot product":
  test "dot":
    var a, b, partial_c: array[N, float64]

    # allocate gpu memory
    var dev_a = hippoMalloc(sizeof(float64)*N)
    var dev_b = hippoMalloc(sizeof(float64)*N)
    var dev_partial_c = hippoMalloc(BlocksPerGrid * sizeof(float64))

    # fill in host memory with data
    for i in 0 ..< N:
      a[i] = i.float
      b[i] = (i * 2).float

    # copy data to device
    hippoMemcpy(dev_a, addr a[0], sizeof(float64)*N, HippoMemcpyHostToDevice)
    hippoMemcpy(dev_b, addr b[0], sizeof(float64)*N, HippoMemcpyHostToDevice)

    # launch kernel
    hippoLaunchKernel(
      dot,
      gridDim = newDim3(BlocksPerGrid.uint32),
      blockDim = newDim3(ThreadsPerBlock.uint32),
      args = hippoArgs(dev_a.p, dev_b.p, dev_partial_c.p)
    )

    # copy memory back from GPU to CPU
    hippoMemcpy(addr partial_c[0], dev_partial_c, BlocksPerGrid * sizeof(float64), HippoMemcpyDeviceToHost)

    # finish up on the CPU
    var c: float64 = 0
    for i in 0 ..< BlocksPerGrid:
      c += partial_c[i]

    proc sum_squares(x: float64): float64 =
      result = x * (x + 1) * (2 * x + 1) / 6
    
    echo fmt"Does GPU value {c:e} = {2 * sum_squares((N - 1)):e}?"