import
  std/[macros, unittest, strformat],
  hippo,
  ./utils

# Concrete types (standard Nim types to match int, long, float, double)
type
  Int = int32 # Equivalent to C int
  Float = float32 # Equivalent to C float
  Long = int64 # Equivalent to C long
  Double = float64 # Equivalent to C double

# Helper proc for AST replacement (used in macro)
proc replaceSym(node: NimNode; oldSym: string; newNode: NimNode): NimNode =
  result = node.copyNimTree()
  for i in 0..<result.len:
    if result[i].kind == nnkIdent and $result[i] == oldSym:
      result[i] = newNode
    else:
      result[i] = replaceSym(result[i], oldSym, newNode)

# Outer macro: takes type and body (for map); simplified: no fixed size, no blockSize param
macro generateForLoopMacro(theType: typedesc, body: untyped): untyped =
  let typeName = $theType # e.g., "Int"
  let macroName = ident("customForLoop_" & typeName) # e.g., customForLoop_Int
  let kernelName = ident("gpuKernel_" & typeName) # Top-level kernel name
  let bufType = nnkBracketExpr.newTree(ident"seq", theType) # Only seq for simplicity

  # Prepare modified body with placeholders replaced
  var modifiedBody = body
  let idxExpr = newCall(ident"int", ident"idx") # int(idx)
  let inArray = ident"inArray"
  let outArray = ident"outArray"
  let itExpr = nnkBracketExpr.newTree(inArray, idxExpr)
  let outItExpr = nnkBracketExpr.newTree(outArray, idxExpr)
  modifiedBody = modifiedBody.replaceSym("it", itExpr)
  modifiedBody = modifiedBody.replaceSym("outIt", outItExpr)

  # I tried injecting the body inside a `quote do` block but this made nim very unhappy.
  # Manually build the kernel proc AST to avoid expansion issues
  let params = nnkFormalParams.newTree(
    newEmptyNode(),  # return type void
    nnkIdentDefs.newTree(ident"inPtr", nnkPtrTy.newTree(theType), newEmptyNode()),
    nnkIdentDefs.newTree(ident"outPtr", nnkPtrTy.newTree(theType), newEmptyNode()),
    nnkIdentDefs.newTree(ident"n", ident"int", newEmptyNode())
  )
  let pragmas = nnkPragma.newTree(ident"hippoGlobal")

  # Build kernel body AST
  let kernelBody = newStmtList(
    nnkLetSection.newTree(
      nnkIdentDefs.newTree(
        ident"idx",
        newEmptyNode(),
        nnkInfix.newTree(
          ident"+",
          nnkDotExpr.newTree(ident"threadIdx", ident"x"),
          nnkInfix.newTree(
            ident"*",
            nnkDotExpr.newTree(ident"blockIdx", ident"x"),
            nnkDotExpr.newTree(ident"blockDim", ident"x")
          )
        )
      )
    ),
    nnkIfStmt.newTree(
      nnkElifBranch.newTree(
        nnkInfix.newTree(
          ident"<",
          newCall(ident"int", ident"idx"),
          ident"n"
        ),
        newStmtList(
          nnkLetSection.newTree(
            nnkIdentDefs.newTree(
              ident"inArray",
              newEmptyNode(),
              nnkCast.newTree(
                nnkPtrTy.newTree(nnkBracketExpr.newTree(ident"UncheckedArray", theType)),
                ident"inPtr"
              )
            ),
            nnkIdentDefs.newTree(
              ident"outArray",
              newEmptyNode(),
              nnkCast.newTree(
                nnkPtrTy.newTree(nnkBracketExpr.newTree(ident"UncheckedArray", theType)),
                ident"outPtr"
              )
            )
          ),
          modifiedBody  # Insert replaced body here
        )
      )
    )
  )

  let kernelDef = nnkProcDef.newTree(
    kernelName,
    newEmptyNode(),
    newEmptyNode(),
    params,
    pragmas,
    newEmptyNode(),
    kernelBody
  )

  # Generate inner proc that handles GPU launch; fixed blockSize=256
  let innerProc = quote do:
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

  result = newStmtList(kernelDef, innerProc)

# Generate macro for Int type with map body
generateForLoopMacro(Int):
  outIt = it * 2 # Map: double

# Generate macro for Float type with map body
generateForLoopMacro(Float):
  outIt = it * 1.5f # Map: multiply by 1.5

# Generate macro for Long type with map body
generateForLoopMacro(Long):
  outIt = it + 1'i64 # Map: add 1

# Generate macro for Double type with map body
generateForLoopMacro(Double):
  outIt = it * it  # Square

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

  testSkipPlatforms "map Float multiply by 1.5", "SIMPLE":
    # Test with Float values
    let seq_float: seq[Float] = @[Float(1.0), Float(2.0)]
    let result_float = customForLoop_Float(seq_float)
    # Verify results: each element should be multiplied by 1.5
    assert result_float.len == 2
    assert result_float[0] == Float(1.5)
    assert result_float[1] == Float(3.0)

  testSkipPlatforms "map Long add 1", "SIMPLE":
    # Test with Long values
    let seq_long: seq[Long] = @[Long(10), Long(0)]
    let result_long = customForLoop_Long(seq_long)
    # Verify results: each element should be incremented by 1
    assert result_long.len == 2
    assert result_long[0] == Long(11)
    assert result_long[1] == Long(1)

  testSkipPlatforms "map Double square", "SIMPLE":
    # Test with Double values
    let seq_double: seq[Double] = @[Double(4.0), Double(5.0)]
    let result_double = customForLoop_Double(seq_double)
    # Verify results: each element should be squared
    assert result_double.len == 2
    assert result_double[0] == Double(16.0)
    assert result_double[1] == Double(25.0)
