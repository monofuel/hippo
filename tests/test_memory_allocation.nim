import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test memory allocation edge cases

proc writeValue(output: ptr[int32], value: int32){.hippoGlobal.} =
  ## Kernel that writes a value to memory
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  outputArray[threadIdx.x] = value

suite "memory allocation":
  test "basic_allocation":
    const Size = 16
    var output = newSeq[int32](Size)
    var dev_output = hippoMalloc(sizeof(int32) * Size)
    var value = 42'i32

    hippoLaunchKernel(
      writeValue,
      gridDim = newDim3(1, 1, 1),
      blockDim = newDim3(Size.uint32, 1, 1),
      args = hippoArgs(dev_output.p, value)
    )

    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * Size, HippoMemcpyDeviceToHost)

    for i in 0..<Size:
      assert(output[i] == 42)

  test "large_allocation":
    const Size = 1024 * 10  # 10KB
    var output = newSeq[int32](Size)
    var dev_output = hippoMalloc(sizeof(int32) * Size)
    var value = 123'i32

    hippoLaunchKernel(
      writeValue,
      gridDim = newDim3(1, 1, 1),
      blockDim = newDim3(1, 1, 1),  # Only use first thread
      args = hippoArgs(dev_output.p, value)
    )

    hippoMemcpy(addr output[0], dev_output, sizeof(int32), HippoMemcpyDeviceToHost)
    assert(output[0] == 123)

  test "zero_size_allocation":
    # Test allocating zero bytes (should not crash)
    var dev_zero = hippoMalloc(0)
    # Just verify it doesn't crash - we can't really use zero-sized allocations
