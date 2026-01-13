import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test hippoHostDevice functions that work on both host and device

proc addNumbers(a, b: int32): int32 {.hippoHostDevice.} =
  ## Function that can be called from both host and device
  return a + b

proc multiplyByTwo(value: int32): int32 {.hippoHostDevice.} =
  ## Another host-device function
  return value * 2

proc computeValue(base: int32): int32 {.hippoHostDevice.} =
  ## Host-device function that calls other host-device functions
  let sum = addNumbers(base, 10)
  return multiplyByTwo(sum)

proc useHostDeviceFunctions(output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that calls host-device functions from device code
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  let tid = threadIdx.x + blockIdx.x * blockDim.x

  # Call host-device functions from device code
  let baseValue = int32(tid)
  let result = computeValue(baseValue)

  outputArray[tid] = result

suite "host device functions":
  test "basic_host_device_interaction":
    const NumThreads = 8

    # First, test calling host-device functions from host code
    let hostResult1 = addNumbers(5, 3)
    assert(hostResult1 == 8, fmt"Host call to addNumbers(5, 3) expected 8, got {hostResult1}")

    let hostResult2 = multiplyByTwo(7)
    assert(hostResult2 == 14, fmt"Host call to multiplyByTwo(7) expected 14, got {hostResult2}")

    let hostResult3 = computeValue(5)
    assert(hostResult3 == 30, fmt"Host call to computeValue(5) expected 30, got {hostResult3}")

    # Now test calling host-device functions from device code
    var output = newSeq[int32](NumThreads)

    # Allocate GPU memory
    var dev_output = hippoMalloc(sizeof(int32) * NumThreads)

    # Launch kernel that uses host-device functions
    hippoLaunchKernel(
      useHostDeviceFunctions,
      gridDim = newDim3(1.uint32),
      blockDim = newDim3(NumThreads.uint32),
      args = hippoArgs(dev_output.p)
    )

    # Copy results back
    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * NumThreads, HippoMemcpyDeviceToHost)

    # Verify device results: computeValue(tid) = multiplyByTwo(addNumbers(tid, 10)) = (tid + 10) * 2
    for i in 0..<NumThreads:
      let expected = int32((i + 10) * 2)
      assert(output[i] == expected, fmt"Thread {i}: expected {expected}, got {output[i]}")
