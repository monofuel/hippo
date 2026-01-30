import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test hippoSynchronize() device synchronization

proc writeValues(output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that writes values to global memory
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  let tid = threadIdx.x + blockIdx.x * blockDim.x
  outputArray[tid] = int32(tid * 10)

proc readAndVerify(input: ptr[int32], output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that reads from global memory and writes verification
  let inputArray = cast[ptr UncheckedArray[int32]](input)
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  let tid = threadIdx.x + blockIdx.x * blockDim.x

  # Read the value written by the first kernel
  let value = inputArray[tid]
  # Write verification: add 1 to the original value
  outputArray[tid] = value + 1

suite "synchronize":
  test "hippo_synchronize":
    const NumThreads = 16
    const TotalElements = NumThreads

    var inputData = newSeq[int32](TotalElements)
    var outputData = newSeq[int32](TotalElements)

    # Allocate GPU memory
    var dev_input = hippoMalloc(sizeof(int32) * TotalElements)
    var dev_output = hippoMalloc(sizeof(int32) * TotalElements)

    # Launch first kernel to write values
    hippoLaunchKernel(
      writeValues,
      gridDim = newDim3(1.uint32),
      blockDim = newDim3(NumThreads.uint32),
      args = hippoArgs(dev_input.p)
    )

    # Synchronize to ensure first kernel completes
    hippoSynchronize()

    # Now launch second kernel that reads from the first kernel's output
    hippoLaunchKernel(
      readAndVerify,
      gridDim = newDim3(1.uint32),
      blockDim = newDim3(NumThreads.uint32),
      args = hippoArgs(dev_input.p, dev_output.p)
    )

    # Synchronize again before copying back
    hippoSynchronize()

    # Copy results back
    hippoMemcpy(addr outputData[0], dev_output, sizeof(int32) * TotalElements, HippoMemcpyDeviceToHost)

    # Verify results: each thread should have (tid * 10) + 1
    for i in 0..<NumThreads:
      let expected = int32(i * 10 + 1)
      assert(outputData[i] == expected, fmt"Thread {i}: expected {expected}, got {outputData[i]}")
