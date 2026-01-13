import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test blockIdx, threadIdx, gridDim, blockDim indices

proc writeIndices(output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that writes thread and block indices to verify correctness
  let outputArray = cast[ptr UncheckedArray[int32]](output)

  # Use 1D indexing for simplicity
  let globalIdx = threadIdx.x + blockIdx.x * blockDim.x

  # Encode indices: blockIdx.x * 10000 + threadIdx.x
  let encoded = int32(blockIdx.x * 10000 + threadIdx.x)
  outputArray[globalIdx] = encoded

suite "block thread indices":
  testSkipPlatforms "indices_1d", "SIMPLE":
    const GridX = 3.uint32
    const BlockX = 4.uint32

    let totalThreads = int(GridX * BlockX)
    var output = newSeq[int32](totalThreads)

    # Allocate GPU memory
    var dev_output = hippoMalloc(sizeof(int32) * totalThreads)

    # Launch 1D kernel
    hippoLaunchKernel(
      writeIndices,
      gridDim = newDim3(GridX, 1, 1),
      blockDim = newDim3(BlockX, 1, 1),
      args = hippoArgs(dev_output.p)
    )

    # Copy results back
    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * totalThreads, HippoMemcpyDeviceToHost)

    # Verify all indices
    for blockIdx in 0..<GridX:
      for threadIdx in 0..<BlockX:
        let globalIdx = int(threadIdx + blockIdx * BlockX)
        let expected = int32(blockIdx * 10000 + threadIdx)
        assert(output[globalIdx] == expected,
               fmt"Global idx {globalIdx}: expected {expected}, got {output[globalIdx]}")
