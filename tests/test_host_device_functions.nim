import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test basic host-device interaction (hippoHostDevice)
# TODO
# host device functions should work on both host and on device


proc useHostDeviceFunctions(output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that performs basic computation
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  let tid = threadIdx.x + blockIdx.x * blockDim.x

  # Simple computation: (tid + 10) * 2
  let sum = int32(tid) + 10
  let product = sum * 2

  outputArray[tid] = product

suite "host device functions":
  testSkipPlatforms "basic_host_device_interaction", "SIMPLE":
    skip()
    # const NumThreads = 8
    # var output = newSeq[int32](NumThreads)

    # # Allocate GPU memory
    # var dev_output = hippoMalloc(sizeof(int32) * NumThreads)

    # # Launch kernel
    # hippoLaunchKernel(
    #   useHostDeviceFunctions,
    #   gridDim = newDim3(1.uint32),
    #   blockDim = newDim3(NumThreads.uint32),
    #   args = hippoArgs(dev_output.p)
    # )

    # # Copy results back
    # hippoMemcpy(addr output[0], dev_output, sizeof(int32) * NumThreads, HippoMemcpyDeviceToHost)

    # # Verify device results: (tid + 10) * 2
    # for i in 0..<NumThreads:
    #   let expected = int32((i + 10) * 2)
    #   assert(output[i] == expected, fmt"Thread {i}: expected {expected}, got {output[i]}")
