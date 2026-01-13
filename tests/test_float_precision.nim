import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test floating point operations and precision

proc floatOperations(output: ptr[float32]){.hippoGlobal.} =
  ## Kernel that performs various float operations
  let outputArray = cast[ptr UncheckedArray[float32]](output)
  let tid = threadIdx.x

  # Simple arithmetic
  let base = float32(tid + 1)
  let result1 = base * 2.0f
  let result2 = result1 + 1.0f
  let result3 = result2 / 3.0f

  # Store results in sequence
  outputArray[tid * 3 + 0] = result1
  outputArray[tid * 3 + 1] = result2
  outputArray[tid * 3 + 2] = result3

proc doubleOperations(output: ptr[float64]){.hippoGlobal.} =
  ## Kernel that performs double precision operations
  let outputArray = cast[ptr UncheckedArray[float64]](output)
  let tid = threadIdx.x

  # More complex operations
  let base = float64(tid + 1)
  let result1 = sin(base)
  let result2 = cos(base)
  let result3 = result1 * result1 + result2 * result2  # Should be close to 1

  outputArray[tid * 3 + 0] = result1
  outputArray[tid * 3 + 1] = result2
  outputArray[tid * 3 + 2] = result3

suite "float precision":
  testSkipPlatforms "float32_operations", "SIMPLE":
    const NumThreads = 4
    const ResultsPerThread = 3
    var output = newSeq[float32](NumThreads * ResultsPerThread)

    var dev_output = hippoMalloc(sizeof(float32) * NumThreads * ResultsPerThread)

    hippoLaunchKernel(
      floatOperations,
      gridDim = newDim3(1, 1, 1),
      blockDim = newDim3(NumThreads.uint32, 1, 1),
      args = hippoArgs(dev_output.p)
    )

    hippoMemcpy(addr output[0], dev_output, sizeof(float32) * NumThreads * ResultsPerThread, HippoMemcpyDeviceToHost)

    # Verify results with tolerance
    for i in 0..<NumThreads:
      let base = float32(i + 1)
      let expected1 = base * 2.0f
      let expected2 = expected1 + 1.0f
      let expected3 = expected2 / 3.0f

      assertAlmostEqual(output[i * 3 + 0], expected1, 1e-6)
      assertAlmostEqual(output[i * 3 + 1], expected2, 1e-6)
      assertAlmostEqual(output[i * 3 + 2], expected3, 1e-6)

  testSkipPlatforms "float64_operations", "SIMPLE":
    const NumThreads = 4
    const ResultsPerThread = 3
    var output = newSeq[float64](NumThreads * ResultsPerThread)

    var dev_output = hippoMalloc(sizeof(float64) * NumThreads * ResultsPerThread)

    hippoLaunchKernel(
      doubleOperations,
      gridDim = newDim3(1, 1, 1),
      blockDim = newDim3(NumThreads.uint32, 1, 1),
      args = hippoArgs(dev_output.p)
    )

    hippoMemcpy(addr output[0], dev_output, sizeof(float64) * NumThreads * ResultsPerThread, HippoMemcpyDeviceToHost)

    # Verify trigonometric identity: sin^2(x) + cos^2(x) = 1
    for i in 0..<NumThreads:
      let base = float64(i + 1)
      let sin_val = output[i * 3 + 0]
      let cos_val = output[i * 3 + 1]
      let identity = output[i * 3 + 2]

      # Check that sin^2 + cos^2 ≈ 1
      assertAlmostEqual(identity, 1.0, 1e-10)
