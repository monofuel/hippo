import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test various kernel launch parameter combinations

proc writeGlobalIndex(output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that writes its global thread index
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  let globalIdx = threadIdx.x + blockIdx.x * blockDim.x
  outputArray[globalIdx] = int32(globalIdx)

suite "kernel launch params":
  testSkipPlatforms "single_thread", "SIMPLE":
    var output = newSeq[int32](1)
    var dev_output = hippoMalloc(sizeof(int32))
    hippoLaunchKernel(
      writeGlobalIndex,
      gridDim = newDim3(1, 1, 1),
      blockDim = newDim3(1, 1, 1),
      args = hippoArgs(dev_output.p)
    )
    hippoMemcpy(addr output[0], dev_output, sizeof(int32), HippoMemcpyDeviceToHost)
    assert(output[0] == 0)

  testSkipPlatforms "single_block", "SIMPLE":
    const Threads = 8
    var output = newSeq[int32](Threads)
    var dev_output = hippoMalloc(sizeof(int32) * Threads)
    hippoLaunchKernel(
      writeGlobalIndex,
      gridDim = newDim3(1, 1, 1),
      blockDim = newDim3(Threads.uint32, 1, 1),
      args = hippoArgs(dev_output.p)
    )
    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * Threads, HippoMemcpyDeviceToHost)
    for i in 0..<Threads:
      assert(output[i] == int32(i))

  testSkipPlatforms "multiple_blocks", "SIMPLE":
    const Blocks = 3
    const ThreadsPerBlock = 4
    const Total = Blocks * ThreadsPerBlock
    var output = newSeq[int32](Total)
    var dev_output = hippoMalloc(sizeof(int32) * Total)
    hippoLaunchKernel(
      writeGlobalIndex,
      gridDim = newDim3(Blocks.uint32, 1, 1),
      blockDim = newDim3(ThreadsPerBlock.uint32, 1, 1),
      args = hippoArgs(dev_output.p)
    )
    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * Total, HippoMemcpyDeviceToHost)
    for i in 0..<Total:
      assert(output[i] == int32(i), fmt"Index {i}: expected {i}, got {output[i]}")
