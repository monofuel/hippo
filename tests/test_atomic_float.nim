import
  std/unittest,
  hippo,
  ./utils

const
  BlocksPerGrid = 256'u32
  ThreadsPerBlock = 256'u32
  ExpectedTotal = float32(BlocksPerGrid * ThreadsPerBlock)

proc atomicAddFloatKernel(total: ptr[float32]) {.hippoGlobal.} =
  ## Add 1.0 to a single global float accumulator from every thread.
  discard hippoAtomicAdd(total, 1.0'f32)

suite "atomic float":
  testSkipPlatforms "atomic add float32 with contention", "SIMPLE":
    var hostTotal = 0.0'f32
    let devTotal = hippoMalloc(sizeof(float32))

    hippoMemcpy(devTotal, addr hostTotal, sizeof(float32), HippoMemcpyHostToDevice)
    hippoLaunchKernel(
      atomicAddFloatKernel,
      gridDim = newDim3(BlocksPerGrid),
      blockDim = newDim3(ThreadsPerBlock),
      args = hippoArgs(devTotal.p)
    )
    hippoSynchronize()
    hippoMemcpy(addr hostTotal, devTotal, sizeof(float32), HippoMemcpyDeviceToHost)

    check hostTotal == ExpectedTotal
