import
  hippo,
  std/[unittest, strformat],
  ./utils

# TODO
# Test hippoDevice functions called from global kernels

proc testDeviceFunctions(output: ptr[int32]){.hippoGlobal.} =
  ## TODO this should call a device function
  let outputArray = cast[ptr UncheckedArray[int32]](output)

  # Each thread writes its thread ID
  let tid = threadIdx.x + blockIdx.x * blockDim.x
  outputArray[tid] = int32(tid)

suite "device functions":
  testSkipPlatforms "basic_functionality", "SIMPLE":
    skip()
    # const NumThreads = 16
    # const TotalResults = NumThreads

    # var output = newSeq[int32](TotalResults)

    # # Allocate GPU memory
    # var dev_output = hippoMalloc(sizeof(int32) * TotalResults)

    # # Launch kernel
    # hippoLaunchKernel(
    #   testDeviceFunctions,
    #   gridDim = newDim3(1.uint32),
    #   blockDim = newDim3(NumThreads.uint32),
    #   args = hippoArgs(dev_output.p)
    # )

    # # Copy results back
    # hippoMemcpy(addr output[0], dev_output, sizeof(int32) * TotalResults, HippoMemcpyDeviceToHost)

    # # Verify results - each thread should write its thread ID
    # for i in 0..<NumThreads:
    #   assert(output[i] == int32(i), fmt"Thread {i}: expected {i}, got {output[i]}")
