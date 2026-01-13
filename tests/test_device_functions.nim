import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test hippoDevice functions called from global kernels

proc addTwo(a, b: int32): int32 {.hippoDevice.} =
  ## Simple device function that adds two numbers
  return a + b

proc multiplyByTwo(value: int32): int32 {.hippoDevice.} =
  ## Device function that multiplies by two
  return value * 2

proc complexOperation(a, b: int32): int32 {.hippoDevice.} =
  ## Device function that calls other device functions
  let sum = addTwo(a, b)
  return multiplyByTwo(sum)

proc testDeviceFunctions(output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that calls device functions
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  let tid = threadIdx.x + blockIdx.x * blockDim.x

  # Call device functions
  let baseValue = int32(tid)
  let result1 = addTwo(baseValue, 10)
  let result2 = multiplyByTwo(baseValue)
  let result3 = complexOperation(baseValue, baseValue + 1)

  # Store results
  outputArray[tid * 3 + 0] = result1
  outputArray[tid * 3 + 1] = result2
  outputArray[tid * 3 + 2] = result3

suite "device functions":
  test "basic_functionality":
    const NumThreads = 16
    const ResultsPerThread = 3
    const TotalResults = NumThreads * ResultsPerThread

    var output = newSeq[int32](TotalResults)

    # Allocate GPU memory
    var dev_output = hippoMalloc(sizeof(int32) * TotalResults)

    # Launch kernel
    hippoLaunchKernel(
      testDeviceFunctions,
      gridDim = newDim3(1.uint32),
      blockDim = newDim3(NumThreads.uint32),
      args = hippoArgs(dev_output.p)
    )

    # Copy results back
    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * TotalResults, HippoMemcpyDeviceToHost)

    # Verify results
    for i in 0..<NumThreads:
      let baseValue = int32(i)

      # Check addTwo(baseValue, 10)
      assert(output[i * 3 + 0] == baseValue + 10, fmt"Thread {i}: addTwo({baseValue}, 10) expected {baseValue + 10}, got {output[i * 3 + 0]}")

      # Check multiplyByTwo(baseValue)
      assert(output[i * 3 + 1] == baseValue * 2, fmt"Thread {i}: multiplyByTwo({baseValue}) expected {baseValue * 2}, got {output[i * 3 + 1]}")

      # Check complexOperation(baseValue, baseValue + 1) = multiplyByTwo(addTwo(baseValue, baseValue + 1)) = multiplyByTwo(baseValue * 2 + 1) = (baseValue * 2 + 1) * 2
      let expected3 = (baseValue * 2 + 1) * 2
      assert(output[i * 3 + 2] == expected3, fmt"Thread {i}: complexOperation({baseValue}, {baseValue + 1}) expected {expected3}, got {output[i * 3 + 2]}")
