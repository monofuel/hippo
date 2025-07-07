import
  std/[strformat, math],
  benchy,
  hippo

## parallel math benchmark
## run with `nix-shell -p tbb --command "nim cpp -r tests/bench_math.nim"`
## 
## test various levels of parallelism to see how it scales with larger vectors

# Simple addition kernels
proc addKernelBlock(a, b, c: ptr[cint]){.hippoGlobal.} =
  let tid = blockIdx.x
  let aArray = cast[ptr UncheckedArray[cint]](a)
  let bArray = cast[ptr UncheckedArray[cint]](b)
  let cArray = cast[ptr UncheckedArray[cint]](c)
  cArray[tid] = aArray[tid] + bArray[tid]

# Complex math kernels - more compute per element
proc mathKernelBlock(a, b, c: ptr[cfloat]){.hippoGlobal.} =
  let tid = blockIdx.x
  let aArray = cast[ptr UncheckedArray[cfloat]](a)
  let bArray = cast[ptr UncheckedArray[cfloat]](b)
  let cArray = cast[ptr UncheckedArray[cfloat]](c)
  # Complex computation: sin(a) * cos(b) + sqrt(abs(a + b))
  let val = sin(aArray[tid]) * cos(bArray[tid]) + sqrt(abs(aArray[tid] + bArray[tid]))
  cArray[tid] = val

proc mathKernelBlockThread(a, b, c: ptr[cfloat]){.hippoGlobal.} =
  let tid = blockIdx.x * blockDim.x + threadIdx.x
  let aArray = cast[ptr UncheckedArray[cfloat]](a)
  let bArray = cast[ptr UncheckedArray[cfloat]](b)
  let cArray = cast[ptr UncheckedArray[cfloat]](c)
  # Complex computation: sin(a) * cos(b) + sqrt(abs(a + b))
  let val = sin(aArray[tid]) * cos(bArray[tid]) + sqrt(abs(aArray[tid] + bArray[tid]))
  cArray[tid] = val

# Serial implementations
proc addSerial(a, b, c: ptr[cint], n: int) =
  let aArray = cast[ptr UncheckedArray[cint]](a)
  let bArray = cast[ptr UncheckedArray[cint]](b)
  let cArray = cast[ptr UncheckedArray[cint]](c)
  for i in 0..<n:
    cArray[i] = aArray[i] + bArray[i]

proc mathSerial(a, b, c: ptr[cfloat], n: int) =
  let aArray = cast[ptr UncheckedArray[cfloat]](a)
  let bArray = cast[ptr UncheckedArray[cfloat]](b)
  let cArray = cast[ptr UncheckedArray[cfloat]](c)
  for i in 0..<n:
    let val = sin(aArray[i]) * cos(bArray[i]) + sqrt(abs(aArray[i] + bArray[i]))
    cArray[i] = val

# Matrix multiplication kernel - O(n^2) operations
proc matMulKernel(a, b, c: ptr[cfloat], n: int32){.hippoGlobal.} =
  let tid = blockIdx.x * blockDim.x + threadIdx.x
  if tid < n.uint32:
    let aArray = cast[ptr UncheckedArray[cfloat]](a)
    let bArray = cast[ptr UncheckedArray[cfloat]](b)
    let cArray = cast[ptr UncheckedArray[cfloat]](c)
    # Simplified: each element does a mini dot product
    var sum: cfloat = 0.0
    for i in 0..<min(n, 32):  # Limit to avoid too much compute
      sum += aArray[tid] * bArray[(tid.int32 + i.int32) mod n]
    cArray[tid] = sum

# Serial matrix multiplication
proc matMulSerial(a, b, c: ptr[cfloat], n: int) =
  let aArray = cast[ptr UncheckedArray[cfloat]](a)
  let bArray = cast[ptr UncheckedArray[cfloat]](b)
  let cArray = cast[ptr UncheckedArray[cfloat]](c)
  for i in 0..<n:
    var sum: cfloat = 0.0
    for j in 0..<min(n, 32):
      sum += aArray[i] * bArray[(i + j) mod n]
    cArray[i] = sum

proc benchVectorSum(n: int32, label: string) =
  echo &"\n=== Vector Size: {n} ({label}) ==="
  
  var a_int = newSeq[int32](n)
  var b_int = newSeq[int32](n)
  var c_int = newSeq[int32](n)
  
  var a_float = newSeq[float32](n)
  var b_float = newSeq[float32](n)
  var c_float = newSeq[float32](n)
  
  # Initialize test data - avoid overflow with large i values
  for i in 0..<n:
    a_int[i] = -(i mod 1000).int32
    b_int[i] = (i mod 100).int32
    a_float[i] = (i mod 1000).float32 / 100.0
    b_float[i] = (i mod 100).float32 / 10.0
  
  # Simple addition benchmarks
  timeIt &"CPU Serial Add {label}":
    addSerial(cast[ptr cint](addr a_int[0]), cast[ptr cint](addr b_int[0]), cast[ptr cint](addr c_int[0]), n.int)
  
  # Complex math benchmarks  
  timeIt &"CPU Serial Math {label}":
    mathSerial(cast[ptr cfloat](addr a_float[0]), cast[ptr cfloat](addr b_float[0]), cast[ptr cfloat](addr c_float[0]), n.int)
  
  # Matrix multiplication benchmarks (only for reasonable sizes)
  if n <= 65536:
    timeIt &"CPU Serial MatMul {label}":
      matMulSerial(cast[ptr cfloat](addr a_float[0]), cast[ptr cfloat](addr b_float[0]), cast[ptr cfloat](addr c_float[0]), n.int)
  
  # GPU Simple addition
  var dev_a_int = hippoMalloc(sizeof(int32) * n.int)
  var dev_b_int = hippoMalloc(sizeof(int32) * n.int)
  var dev_c_int = hippoMalloc(sizeof(int32) * n.int)
  
  hippoMemcpy(dev_a_int, addr a_int[0], sizeof(int32) * n.int, HippoMemcpyHostToDevice)
  hippoMemcpy(dev_b_int, addr b_int[0], sizeof(int32) * n.int, HippoMemcpyHostToDevice)
  
  timeIt &"GPU Add {label}":
    hippoLaunchKernel(
      addKernelBlock,
      gridDim = newDim3(n.uint32),
      args = hippoArgs(dev_a_int.p, dev_b_int.p, dev_c_int.p)
    )
    hippoSynchronize()
  
  # GPU Complex math
  var dev_a_float = hippoMalloc(sizeof(float32) * n.int)
  var dev_b_float = hippoMalloc(sizeof(float32) * n.int)
  var dev_c_float = hippoMalloc(sizeof(float32) * n.int)
  
  hippoMemcpy(dev_a_float, addr a_float[0], sizeof(float32) * n.int, HippoMemcpyHostToDevice)
  hippoMemcpy(dev_b_float, addr b_float[0], sizeof(float32) * n.int, HippoMemcpyHostToDevice)
  
  timeIt &"GPU Math {label}":
    hippoLaunchKernel(
      mathKernelBlock,
      gridDim = newDim3(n.uint32),
      args = hippoArgs(dev_a_float.p, dev_b_float.p, dev_c_float.p)
    )
    hippoSynchronize()
  
  # GPU Matrix multiplication (only for reasonable sizes)
  if n <= 65536:
    let threadsPerBlock = 256.uint32
    let blocksPerGrid = ((n.uint32 + threadsPerBlock - 1) div threadsPerBlock)
    timeIt &"GPU MatMul {label}":
      hippoLaunchKernel(
        matMulKernel,
        gridDim = newDim3(blocksPerGrid),
        blockDim = newDim3(threadsPerBlock),
        args = hippoArgs(dev_a_float.p, dev_b_float.p, dev_c_float.p, n)
      )
      hippoSynchronize()
  
  # GPU Block+Thread combination for larger vectors
  if n > 1024:
    let threadsPerBlock = 256.uint32
    let blocksPerGrid = ((n.uint32 + threadsPerBlock - 1) div threadsPerBlock)
    timeIt &"GPU Math Blocks+Threads {label}":
      hippoLaunchKernel(
        mathKernelBlockThread,
        gridDim = newDim3(blocksPerGrid),
        blockDim = newDim3(threadsPerBlock),
        args = hippoArgs(dev_a_float.p, dev_b_float.p, dev_c_float.p)
      )
      hippoSynchronize()
  
  # Hippo GpuRefs do not need to be freed manually! done automatically.

# Test different vector sizes - focusing on where parallelism should win
when isMainModule:
  echo "Vector Sum Benchmarks - Serial CPU vs Parallel GPU"
  echo "=================================================="
  
  benchVectorSum(1024, "Small")
  benchVectorSum(4096, "Medium")
  benchVectorSum(16384, "Large")
  benchVectorSum(65536, "X-Large")
  benchVectorSum(262144, "XX-Large")
  benchVectorSum(1048576, "XXX-Large")