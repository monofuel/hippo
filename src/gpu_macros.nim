import std/macros

## Experimental Nim GPU 'macro macros' for generating GPU kernels and helpers to use them.
## these macros accept a kernel `body` and produce a macro for easily using the kernel.


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

export generateForLoopMacro
