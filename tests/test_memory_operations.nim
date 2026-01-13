import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test device-to-device memory operations (memcpy)

proc copyData(src: ptr[int32], dst: ptr[int32]){.hippoGlobal.} =
  ## Kernel that copies data from src to dst on device
  let srcArray = cast[ptr UncheckedArray[int32]](src)
  let dstArray = cast[ptr UncheckedArray[int32]](dst)
  let tid = threadIdx.x + blockIdx.x * blockDim.x

  # Copy with some transformation
  dstArray[tid] = srcArray[tid] * 2

suite "memory operations":
  testSkipPlatforms "device_to_device_memcpy", "SIMPLE":
    const NumElements = 16
    var hostData = newSeq[int32](NumElements)
    var resultData = newSeq[int32](NumElements)

    # Initialize host data
    for i in 0..<NumElements:
      hostData[i] = int32(i + 10)

    # Allocate device memory
    var dev_src = hippoMalloc(sizeof(int32) * NumElements)
    var dev_dst = hippoMalloc(sizeof(int32) * NumElements)

    # Copy data to first device buffer
    hippoMemcpy(dev_src, addr hostData[0], sizeof(int32) * NumElements, HippoMemcpyHostToDevice)

    # Perform device-to-device copy via kernel
    hippoLaunchKernel(
      copyData,
      gridDim = newDim3(1.uint32),
      blockDim = newDim3(NumElements.uint32),
      args = hippoArgs(dev_src.p, dev_dst.p)
    )

    # Copy result back to host
    hippoMemcpy(addr resultData[0], dev_dst, sizeof(int32) * NumElements, HippoMemcpyDeviceToHost)

    # Verify results: each element should be doubled
    for i in 0..<NumElements:
      let expected = int32((i + 10) * 2)
      assert(resultData[i] == expected, fmt"Element {i}: expected {expected}, got {resultData[i]}")

  testSkipPlatforms "multiple_allocations", "SIMPLE":
    # Test multiple sequential allocations and deallocations
    const NumBuffers = 3
    const BufferSize = 16

    # Test allocating and using multiple buffers sequentially
    for i in 0..<NumBuffers:
      var hostData = newSeq[int32](BufferSize)
      for j in 0..<BufferSize:
        hostData[j] = int32(i * 100 + j)

      # Allocate buffer for this iteration
      var dev_buffer = hippoMalloc(sizeof(int32) * BufferSize)

      # Copy to device
      hippoMemcpy(dev_buffer, addr hostData[0], sizeof(int32) * BufferSize, HippoMemcpyHostToDevice)

      # Copy back and verify
      var resultData = newSeq[int32](BufferSize)
      hippoMemcpy(addr resultData[0], dev_buffer, sizeof(int32) * BufferSize, HippoMemcpyDeviceToHost)

      for j in 0..<BufferSize:
        let expected = int32(i * 100 + j)
        assert(resultData[j] == expected, fmt"Buffer {i}, element {j}: expected {expected}, got {resultData[j]}")

      # Buffer is automatically freed when dev_buffer goes out of scope
