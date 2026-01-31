import std/macros


## Experimental Nim GPU 'macro macros' for generating GPU kernels and helpers to use them.
## these macros accept a kernel `body` and produce a macro for easily using the kernel.


proc replaceSym(node: NimNode; oldSym: string; newNode: NimNode): NimNode =
  ## Recursively replace identifier uses of `oldSym` with `newNode`.
  result = node.copyNimTree()
  for i in 0..<result.len:
    if result[i].kind == nnkIdent and $result[i] == oldSym:
      result[i] = newNode
    else:
      result[i] = replaceSym(result[i], oldSym, newNode)

proc buildKernelParams(theType: NimNode): NimNode =
  ## Build the formal parameter list for the GPU kernel proc.
  nnkFormalParams.newTree(
    newEmptyNode(),  # return type void
    nnkIdentDefs.newTree(ident"inPtr", nnkPtrTy.newTree(theType), newEmptyNode()),
    nnkIdentDefs.newTree(ident"outPtr", nnkPtrTy.newTree(theType), newEmptyNode()),
    nnkIdentDefs.newTree(ident"n", ident"int", newEmptyNode())
  )

proc buildKernelBody(theType: NimNode; body: NimNode): NimNode =
  ## Build the kernel body with `it`/`outIt` substitutions.
  let idxSym = genSym(nskLet, "idx")
  let inArraySym = genSym(nskLet, "inArray")
  let outArraySym = genSym(nskLet, "outArray")
  let idxExpr = newCall(ident"int", idxSym)
  let itExpr = nnkBracketExpr.newTree(inArraySym, idxExpr)
  let outItExpr = nnkBracketExpr.newTree(outArraySym, idxExpr)

  var modifiedBody = replaceSym(body, "it", itExpr)
  modifiedBody = replaceSym(modifiedBody, "outIt", outItExpr)

  result = quote do:
    let `idxSym` = threadIdx.x + blockIdx.x * blockDim.x
    if int(`idxSym`) < n:
      let `inArraySym` = cast[ptr UncheckedArray[`theType`]](inPtr)
      let `outArraySym` = cast[ptr UncheckedArray[`theType`]](outPtr)
      `modifiedBody`

proc buildKernelDef(kernelName: NimNode; theType: NimNode; body: NimNode): NimNode =
  ## Define the `hippoGlobal` kernel proc.
  let pragmas = nnkPragma.newTree(ident"hippoGlobal")
  nnkProcDef.newTree(
    kernelName,
    newEmptyNode(),
    newEmptyNode(),
    buildKernelParams(theType),
    pragmas,
    newEmptyNode(),
    buildKernelBody(theType, body)
  )

proc buildLauncherProc(macroName: NimNode; kernelName: NimNode; bufType: NimNode; theType: NimNode): NimNode =
  ## Build the host-side launcher proc that allocates, copies, and launches.
  quote do:
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

# Outer macro: takes type and body (for map); simplified: no fixed size, no blockSize param
macro generateForLoopMacro(name: static[string], theType: typedesc, body: untyped): untyped =
  ## Generate a GPU kernel and a host launcher proc using a custom name.
  let typeName = $theType
  let macroName = ident(name)
  let kernelName = ident("gpuKernel_" & name & "_" & typeName)
  let bufType = nnkBracketExpr.newTree(ident"seq", theType) # Only seq for simplicity
  let kernelDef = buildKernelDef(kernelName, theType, body)
  let innerProc = buildLauncherProc(macroName, kernelName, bufType, theType)
  result = newStmtList(kernelDef, innerProc)

export generateForLoopMacro
