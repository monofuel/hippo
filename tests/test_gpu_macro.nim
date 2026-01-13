import
  std/[macros, unittest, strformat],
  hippo,
  ./utils

# Concrete types (standard Nim types to match int, long, float, double)
type
  Int = int32 # Equivalent to C int

# Outer macro: takes type and body (for map); simplified: no fixed size, no blockSize param
macro generateForLoopMacro(theType: typedesc): untyped =
  let typeName = $theType # e.g., "Int"
  let macroName = ident("customForLoop_" & typeName) # e.g., customForLoop_Int
  let kernelName = ident("gpuKernel_" & typeName) # Top-level kernel name
  let bufType = nnkBracketExpr.newTree(ident"seq", theType) # Only seq for simplicity

  # Generate top-level GPU kernel for map operations
  let kernelDef = quote do:
    proc `kernelName`(inPtr: ptr `theType`, outPtr: ptr `theType`, n: int) {.hippoGlobal.} =
      let idx = threadIdx.x + blockIdx.x * blockDim.x
      if int(idx) < n:
        let inArray = cast[ptr UncheckedArray[`theType`]](inPtr)
        let outArray = cast[ptr UncheckedArray[`theType`]](outPtr)
        let it = inArray[int(idx)]
        var outIt: `theType`
        outIt = it * 2 # Hardcoded body for now
        outArray[int(idx)] = outIt

  # Generate inner macro that handles GPU launch; fixed blockSize=256
  let innerMacro = quote do:
    proc `macroName`(arr: `bufType`): seq[`theType`] =
      let n = arr.len
      let inputSize = n * sizeof(`theType`)
      let devIn = hippoMalloc(inputSize)
      let devOut = hippoMalloc(inputSize)
      hippoMemcpy(devIn, unsafeAddr(arr[0]), inputSize, HippoMemcpyHostToDevice)
      let blockDim = newDim3(256)
      let gridDim = newDim3(uint32((n + 255) div 256))
      hippoLaunchKernel(`kernelName`, gridDim=gridDim, blockDim=blockDim, args=hippoArgs(devIn.p, devOut.p, n))
      hippoSynchronize()
      var output = newSeq[`theType`](n)
      hippoMemcpy(unsafeAddr(output[0]), devOut, inputSize, HippoMemcpyDeviceToHost)
      # No explicit hippoFree needed; handled by Nim GC via GpuRef
      output

  result = newStmtList(kernelDef, innerMacro)

# Generate macro for Int type with map body (hardcoded to map by 2 for now)
generateForLoopMacro(Int)

suite "GPU macro map operations":
  testSkipPlatforms "map multiply by 2", "SIMPLE":
    # This test runs on: HIP, CUDA, HIP_CPU
    # This test skips on: SIMPLE and SIMPLE_NO_THREADS (both thread variants)
    let seq_int: seq[Int] = @[Int(1), Int(2), Int(3)]
    let result_int = customForLoop_Int(seq_int)
    # Verify results: each element should be doubled
    assert result_int.len == 3
    assert result_int[0] == Int(2)
    assert result_int[1] == Int(4)
    assert result_int[2] == Int(6)

  testSkipPlatforms "map with different values", "SIMPLE":
    # Test with different input values
    let seq_int: seq[Int] = @[Int(10), Int(0), Int(-5)]
    let result_int = customForLoop_Int(seq_int)
    # Verify results: each element should be doubled
    assert result_int.len == 3
    assert result_int[0] == Int(20)
    assert result_int[1] == Int(0)
    assert result_int[2] == Int(-10)
