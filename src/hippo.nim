import
  std/[strformat, os, macros, strutils, tables, sequtils]


# supported runtimes: HIP, HIP_CPU, CUDA, and SIMPLE
const HippoRuntime* {.strdefine.} = "HIP"

{.warning: "DEBUG: Using Hippo Runtime: " & HippoRuntime.}
echo &"DEBUG: Using Hippo Runtime: {HippoRuntime}"

when not defined(Nimdoc) and ((defined(c) or defined(js)) and HippoRuntime != "SIMPLE"):
  {.error: "The HIP, CUDA and HIP_CPU backends require compiling with cpp".}

## Nim Library to enable writing CUDA and HIP kernels in Nim
## All cuda and hip structures and functions are re-exported and can be used
##
## - There are 3 sets of function prefixes.
## - `hippo*` prefixed functions are friendly nim interfaces for either HIP or CUDA
##   - This is the recommended way to use this library, as it is the most nim-like
##   - These functions check for errors and raise them as exceptions
## - `hip*` prefixed functions are the raw HIP C++ functions
## - `cuda*` prefixed functions are the raw CUDA C functions


# -------------------
# Compiler Specifics
# this section is for special cases to add libraries or change includes depending on what compiler and settings are being used.

proc getHipPlatform(): string =
  ## getHipPlatform is a compile time specific function, and gets the target platform for hipcc.
  ## NVCC and HIPCC (when building for nvidia) require that we pass compiler args in -Xcompiler="".
  ## hipcc uses HIP_PLATFORM to determine amd / nvidia.
  ## hipcc defaults to amd if amdclang++ or hip clang is found.
  ## https://github.com/ROCm/llvm-project/blob/00fdfae9aeef14c905550601c2218a6b5962f48c/amd/hipcc/bin/hipvars.pm#L131
  ## 
  #TODO: this check does not work on nixos because things are not in the normal locations
  let 
    clangPath = getEnv("HIP_CLANG_PATH", "") / "clang++"
    amdClangPath = "/opt/rocm/bin/amdclang++"
    defaultPlatform = if (fileExists(clangPath) or fileExists(amdClangPath)): "amd" else: "nvidia"
    hipPlatform = getEnv("HIP_PLATFORM", defaultPlatform)
  echo &"DEBUG: HIP Platform: {hipPlatform}"
  result = hipPlatform

const HipPlatform = getHipPlatform()

# HippoRuntime can be set to "HIP", "HIP_CPU", or "CUDA"
# HIP hipcc will auto detect the runtime of the build system
# HIP_CPU will use the HIP-CPU runtime header
# CUDA will use nvcc

if HippoRuntime == "HIP":
  echo &"DEBUG: Using HIP Platform: {HipPlatform}"

when HippoRuntime == "HIP_CPU":
  # Intel TBB is required for HIP-CPU
  when defined(linux):
    {.passL: "-ltbb".}
  elif defined(windows):
    # using msys2 pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-tbb
    {.passL: "-ltbb12".}
  # I forgot when I had to use stdc++? maybe it's needed?
  # {.passL: "-lstdc++".}

  # hip.nim expects hip/hip_runtime.h to be in the include path
  # for HIP-CPU, we want to dynamically set the include path
  # the include path is in this library, but it needs to be imported into the user's project
  const
    hipIncludePath = staticExec("pwd") & "/../HIP-CPU/include"
  {.passC: "-I" & hipIncludePath.}
  echo "DEBUG: Using HIP CPU runtime"
  include hip
elif HippoRuntime == "CUDA":
  # nvcc loads the CUDA runtime header automatically
  echo "DEBUG: Using CUDA runtime"
  include cuda
elif HippoRuntime == "SIMPLE":
  echo "DEBUG: Using SIMPLE runtime"
  include simple
else:
  echo "DEBUG: Using HIP runtime"
  include hip

# -------------------
# Hippo Templates
# nim wrappers around hip and cuda functions.
# these hippo* functions need to be nim-friendly and warp around the raw hip and cuda functions.

type
  GpuMemory = object ## Wrapper around gpu memory for automatic cleanup
    p*: pointer
  GpuRef = ref GpuMemory

template hippoMalloc*(size: int): GpuRef =
  ## Allocate memory on the GPU and return a GpuRef object.
  ## GpuMemory is a wrapper around Gpu allocated pointers.
  ## It will automatically free the memory when it goes out of scope.
  var g = GpuRef()
  when HippoRuntime == "CUDA":
    handleError(cudaMalloc(addr g.p, size.cint))
  elif HippoRuntime == "SIMPLE":
    simpleMalloc(addr g.p, size)
  else:
    handleError(hipMalloc(addr g.p, size.cint))
  g

template hippoMemcpy*(dst: pointer, src: pointer, size: int, kind: HippoMemcpyKind) =
  ## host -> host memory copy
  ## hippoMemcpy is broken out as 4 separate templates to make it easier to work with GpuRef objects
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst, src, size.cint, kind))
  elif HippoRuntime == "SIMPLE":
    simpleMemcpy(dst, src, size, kind)
  else:
    handleError(hipMemcpy(dst, src, size.cint, kind))

template hippoMemcpy*(dst: pointer, src: GpuRef, size: int, kind: HippoMemcpyKind) =
  ## host -> device memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst, src.p, size.cint, kind))
  elif HippoRuntime == "SIMPLE":
    simpleMemcpy(dst, src.p, size, kind)
  else:
    handleError(hipMemcpy(dst, src.p, size.cint, kind))

template hippoMemcpy*(dst: GpuRef, src: pointer, size: int, kind: HippoMemcpyKind) =
  ## device -> host memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst.p, src, size.cint, kind))
  elif HippoRuntime == "SIMPLE":
    simpleMemcpy(dst.p, src, size, kind)
  else:
    handleError(hipMemcpy(dst.p, src, size.cint, kind))

template hippoMemcpy*(dst: GpuRef, src: GpuRef, size: int, kind: HippoMemcpyKind) =
  ## device -> device memory copy
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`.
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst.p, src.p, size.cint, kind))
  elif HippoRuntime == "SIMPLE":
    simpleMemcpy(dst.p, src.p, size, kind)
  else:
    handleError(hipMemcpy(dst.p, src.p, size.cint, kind))

template hippoFree*(p: pointer) =
  ## Free memory on the GPU
  when HippoRuntime == "CUDA":
    handleError(cudaFree(p))
  elif HippoRuntime == "SIMPLE":
    simpleFree(p)
  else:
    handleError(hipFree(p))

# TODO maybe this should be called hippoDeviceSynchronize to be more consistent?
template hippoSynchronize*() =
  ## Synchronize the device
  when HippoRuntime == "CUDA":
    handleError(cudaDeviceSynchronize())
  elif HippoRuntime == "SIMPLE":
    # in cpu mode, kernels are performed syncronously
    discard
  else:
    handleError(hipDeviceSynchronize())

proc `=destroy`*(mem: var GpuMemory) =
  ## Automatically free device memory when the object goes out of scope
  if mem.p != nil:
    hippoFree(mem.p)
    mem.p = nil

proc hippoRefcopy[T](obj: ref T): GpuRef =
  ## Performs a shallow copy of a ref object to the GPU.
  let size = sizeof(T)
  result = hippoMalloc(size)
  hippoMemcpy(result, addr obj[], size, HippoMemcpyHostToDevice)

proc hippoRefcopy[T](gpuref: GpuRef): ref T =
  ## Copies gpu memory to a new ref object on the host
  let size = sizeof(T)
  result = new T
  hippoMemcpy(addr result[], gpuref, size, HippoMemcpyDeviceToHost)

proc hippoRefcopy[T](gpuref: GpuRef, target: ref T) =
  ## Copies gpu memory to a ref object on the host
  let size = sizeof(T)
  hippoMemcpy(addr target[], gpuref, size, HippoMemcpyDeviceToHost)

# -------------------
# Kernel Execution

when HippoRuntime == "HIP" or HippoRuntime == "HIP_CPU":
  macro hipLaunchKernelGGLWithTuple(
    kernel: proc,
    gridDim: Dim3 = newDim3(1,1,1),
    blockDim: Dim3 = newDim3(1,1,1),
    sharedMemBytes: uint32 = 0,
    stream: HippoStream = nil,
    args: tuple
    ): untyped =

    var callNode = newCall(bindSym"hipLaunchKernelGGL")

    # add the fixed vars
    callNode.add kernel
    callNode.add gridDim
    callNode.add blockDim
    callNode.add sharedMemBytes
    callNode.add stream

    # add every value of the tuple
    for child in args:
      callNode.add child
    result = callNode

template hippoLaunchKernel*(
  kernel: proc,                     ## The GPU kernel procedure to launch
  gridDim: Dim3 = newDim3(1,1,1),   ## default to a grid of 1 block
  blockDim: Dim3 = newDim3(1,1,1),  ## default to 1 thread per block
  sharedMemBytes: uint32 = 0,       ## dynamic shared memory amount to allocate
  stream: HippoStream = nil,        ## Which device stream to run under (defaults to null)
  args: untyped,     ## tuple of pointers to arguments (pointers to arguments! not arguments!) to pass to the GPU kernel
) =
  var result: HippoError
  ## Launch a kernel on the GPU.
  ## also checks if launchKernel() returns an error.
  ## Important: this only checks if the kernel launch was successful, not the kernel itself.
  ## If you need help debugging, you can call hippoSynchronize() to wait for the kernel to finish and report errors.

  when HippoRuntime == "HIP" and HipPlatform == "amd":
    var kernelArgs: seq[pointer] = cast[seq[pointer]](args)
    result = hipLaunchKernel(
      cast[pointer](kernel),
      gridDim,
      blockDim,
      addr kernelArgs[0],
      sharedMemBytes,
      stream
    )
  elif (HippoRuntime == "HIP" and HipPlatform == "nvidia") or HippoRuntime == "HIP_CPU":
    hipLaunchKernelGGLWithTuple(
      kernel,
      gridDim,
      blockDim,
      sharedMemBytes,
      stream,
      args
    )
    result = hipGetLastError()
  elif HippoRuntime == "CUDA":
    var kernelArgs: seq[pointer] = cast[seq[pointer]](args)
    result = cudaLaunchKernel(
      kernel,
      gridDim,
      blockDim,
      addr kernelArgs[0],
      sharedMemBytes,
      stream
    )
  elif HippoRuntime == "SIMPLE":
    simpleLaunchKernel(kernel, gridDim, blockDim, args)
  else:
    raise newException(Exception, &"Unknown runtime: {HippoRuntime}")

  handleError(result)
  


# -------------------
# Macros
# these Nim macros wrap around the attributes required by cuda and hip (which are identical, and as such are here and not in hip.nim or cuda.nim).

macro hippoGlobal*(fn: untyped): untyped =
  ## Declare a function as `__global__`. global functions are called from the host and run on the device.
  when HippoRuntime != "SIMPLE":
    let globalPragma: NimNode = quote:
      {. exportc, codegenDecl: "__global__ $# $#$#".}

    fn.addPragma(globalPragma[0])
    fn.addPragma(globalPragma[1])
  quote do:
    {.push stackTrace: off, checks: off.}
    `fn`
    {.pop.}

macro hippoDevice*(fn: untyped): untyped =
  ## Declare fuctions for use on the `__device__` (the gpu),
  ## to be called by either `device` or `global` functions.
  when HippoRuntime != "SIMPLE":
    let globalPragma: NimNode = quote:
      {. exportc, codegenDecl: "__device__ $# $#$#".}

    fn.addPragma(globalPragma[0])
    fn.addPragma(globalPragma[1])
  quote do:
    {.push stackTrace: off, checks: off.}
    `fn`
    {.pop.}


macro hippoHost*(fn: untyped): untyped =
  ## Explicitly declare a function as a `__host__` function (cpu side).
  ## All functions default to `host` functions, so this is not required.
  when HippoRuntime != "SIMPLE":
    let globalPragma: NimNode = quote:
      {. exportc, codegenDecl: "__host__ $# $#$#".}

    fn.addPragma(globalPragma[0])
    fn.addPragma(globalPragma[1])
  quote do:
    {.push stackTrace: off, checks: off.}
    `fn`
    {.pop.}

macro hippoHostDevice*(fn: untyped): untyped =
  ## Declare a function as both `__host__` and `__device__`.
  ## This is useful for functions that are usable from either the host and the device.
  ## eg: `proc add(a: int, b: int) {.hippoHostDevice.} = a + b`
  when HippoRuntime != "SIMPLE":
    let globalPragma: NimNode = quote:
      {. exportc, codegenDecl: "__device__ __host__ $# $#$#".}

    fn.addPragma(globalPragma[0])
    fn.addPragma(globalPragma[1])
  quote do:
    {.push stackTrace: off, checks: off.}
    `fn`
    {.pop.}

macro hippoShared*(v: untyped): untyped =
  ## Declared a variable as static shared memory `__shared__`.
  ## Shared memory is shared between threads in the same block.
  ## It is faster than global memory, but is limited in size. They are located on-chip.
  ## eg: `var cache {.hippoShared.}: array[256, float]`
  when HippoRuntime != "SIMPLE":
    quote do:
      {.push stackTrace: off, checks: off, noinit, exportc, codegenDecl: "__shared__ $# $#".}
      `v`
      {.pop.}
  else:
    # TODO proper thread vars
    quote do:
      `v`

macro hippoConstant*(v: untyped): untyped =
  ## Declared a variable as `__constant__`.
  ## Constants are read-only globals that are cached on-chip.
  ## constants are useful for data that is being read by all threads in a warp at the same time.
  ## if each thread in a warp accesses different addresses in constant memory,
  ## the accesses are serialized and this may cause a 16x slowdown.
  ## eg: `const N {.hippoConstant.} = 1024`
  quote do:
    {.push stackTrace: off, checks: off, noinit, exportc, codegenDecl: "__constant__ $# $#".}
    `v`
    {.pop.}


var deviceAnnotatedFunctions {.compileTime.}: seq[string] = @[]

proc findCalledFunctions(n: NimNode): seq[NimNode] =
  ## Recursively walk AST to find all function calls
  result = newSeq[NimNode]()
  case n.kind:
  of nnkCall, nnkCommand:
    if n.len > 0 and n[0].kind in {nnkIdent, nnkSym}:
      result.add(n[0])
    for child in n:
      result.add(findCalledFunctions(child))
  of nnkDotExpr:
    if n.len > 1 and n[1].kind in {nnkIdent, nnkSym}:
      result.add(n[1])
    for child in n:
      result.add(findCalledFunctions(child))
  else:
    for child in n:
      result.add(findCalledFunctions(child))

proc shouldWrapFunction(funcName: string): bool =
  ## Determine if a function should be wrapped for device execution
  if funcName in deviceAnnotatedFunctions:
    return false  # Already processed
  
  # Skip system functions, operators, and built-ins
  if funcName in ["len", "high", "low", "int", "uint8", "float", "cfloat", "addr", "cast", "sizeof"]:
    return false
  if funcName.startsWith("nim"):
    return false
  if funcName.startsWith("system"):
    return false
  if funcName.startsWith("=") or funcName.startsWith("@"):
    return false
  if funcName in [".", "[]", "+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">="]:
    return false
  
  # Skip single character identifiers (likely variables)
  if funcName.len == 1:
    return false
  
  # Skip common variable names
  if funcName in ["x", "y", "z", "i", "j", "k", "offset", "res"]:
    return false
  
  return true

proc transformFunctionCalls(node: NimNode, wrapperMap: Table[string, string]): NimNode =
  ## Transform function calls to use device wrappers
  result = copyNimNode(node)
  
  case node.kind
  of nnkCall, nnkCommand:
    if node.len > 0:
      case node[0].kind
      of nnkIdent:
        let funcName = node[0].strVal
        if wrapperMap.hasKey(funcName):
          result.add(ident(wrapperMap[funcName]))
        else:
          result.add(transformFunctionCalls(node[0], wrapperMap))
      of nnkSym:
        let funcName = node[0].strVal
        if wrapperMap.hasKey(funcName):
          result.add(ident(wrapperMap[funcName]))
        else:
          result.add(transformFunctionCalls(node[0], wrapperMap))
      else:
        result.add(transformFunctionCalls(node[0], wrapperMap))
    
    # Transform arguments
    for i in 1 ..< node.len:
      result.add(transformFunctionCalls(node[i], wrapperMap))
  else:
    # Transform all children
    for child in node:
      result.add(transformFunctionCalls(child, wrapperMap))

macro autoDeviceKernel*(fn: typed): untyped =
  ## Automatically create device wrapper functions for called functions and transform kernel to use them
  
  # Extract the actual function definition
  let actualFunc = 
    if fn.kind == nnkStmtList and fn.len >= 2:
      fn[1] # The function is at index 1
    else:
      fn
  
  if actualFunc.kind notin {nnkProcDef, nnkFuncDef}:
    error("autoDeviceKernel expects a function definition")
    return fn
  
  # Get the function body
  let functionBody = actualFunc[6]
  
  # Find all function calls recursively
  let calledFunctions = findCalledFunctions(functionBody)
  
  # Create device versions and mapping
  var deviceFunctions: seq[NimNode] = @[]
  var functionMapping = initTable[string, string]()
  var processedFunctions: seq[string] = @[]
  
  # Process called functions to create device wrappers
  proc processFunction(funcNode: NimNode, funcName: string) =
    if funcName in processedFunctions or not shouldWrapFunction(funcName):
      return
    
    processedFunctions.add(funcName)
    
    if funcNode.kind == nnkSym:
      try:
        let funcImpl = funcNode.getImpl()
        if funcImpl.kind in {nnkProcDef, nnkFuncDef, nnkMethodDef}:
          let deviceName = "gpu_" & funcName
          
          # Extract function signature
          let funcSignature = funcImpl[3] # FormalParams
          let returnType = funcSignature[0]
          
          # Create new parameter symbols and build wrapper signature
          var wrapperParams = newNimNode(nnkFormalParams)
          wrapperParams.add(returnType)
          
          var paramMapping = newSeq[(NimNode, NimNode)]() # (original, new)
          var callArgs = newSeq[NimNode]()
          
          for i in 1 ..< funcSignature.len:
            let paramDef = funcSignature[i]
            if paramDef.kind == nnkIdentDefs:
              let paramType = paramDef[paramDef.len - 2]
              
              # Create new parameter definitions with fresh symbols
              var newParamDef = newNimNode(nnkIdentDefs)
              for j in 0 ..< paramDef.len - 2:
                let originalParam = paramDef[j]
                let newParam = genSym(nskParam, $originalParam)
                paramMapping.add((originalParam, newParam))
                newParamDef.add(newParam)
                callArgs.add(newParam)
              
              newParamDef.add(paramType)
              newParamDef.add(newEmptyNode())
              wrapperParams.add(newParamDef)
          
          # Create the call to the original function
          var originalCall = newCall(ident(funcName))
          for arg in callArgs:
            originalCall.add(arg)
          
          # Create wrapper body
          let wrapperBody = newStmtList(
            newNimNode(nnkReturnStmt).add(originalCall)
          )
          
          # Create the device wrapper function
          var deviceWrapper = newNimNode(nnkProcDef)
          deviceWrapper.add(ident(deviceName))  # name
          deviceWrapper.add(newEmptyNode())     # generics
          deviceWrapper.add(newEmptyNode())     # unused
          deviceWrapper.add(wrapperParams)      # params
          
          # Add device pragmas
          var pragmas = newNimNode(nnkPragma)
          when HippoRuntime != "SIMPLE":
            pragmas.add(ident("exportc"))
            
            let codegenDecl = newNimNode(nnkExprColonExpr)
            codegenDecl.add(ident("codegenDecl"))
            codegenDecl.add(newStrLitNode("__host__ __device__ $# $#$#"))
            pragmas.add(codegenDecl)
          
          deviceWrapper.add(pragmas)            # pragmas
          deviceWrapper.add(newEmptyNode())     # reserved
          deviceWrapper.add(wrapperBody)        # body
          
          # Wrap with push/pop pragmas for stackTrace and checks
          let wrappedDeviceFunc = quote do:
            {.push stackTrace: off, checks: off.}
            `deviceWrapper`
            {.pop.}
          
          deviceFunctions.add(wrappedDeviceFunc)
          functionMapping[funcName] = deviceName
          
          # Recursively process functions called by the original function
          let nestedCalls = findCalledFunctions(funcImpl[6])
          for nestedCall in nestedCalls:
            if nestedCall.kind in {nnkSym, nnkIdent}:
              let nestedName = $nestedCall
              processFunction(nestedCall, nestedName)
      except:
        # If we can't get the implementation, skip this function
        discard
  
  # Process all called functions
  for funcNode in calledFunctions:
    if funcNode.kind in {nnkSym, nnkIdent}:
      let funcName = $funcNode
      processFunction(funcNode, funcName)
  
  # Build the result
  result = newStmtList()
  
  # Add all device functions first
  for deviceFunc in deviceFunctions:
    result.add(deviceFunc)
  
  # Transform the kernel to use device functions
  if functionMapping.len > 0:
    var transformedKernel = copyNimTree(fn)
    let transformedBody = transformFunctionCalls(functionBody, functionMapping)
    
    # Update the kernel body to use device function calls
    if transformedKernel.kind == nnkStmtList and transformedKernel.len >= 2:
      transformedKernel[1][6] = transformedBody
    else:
      transformedKernel[6] = transformedBody
    
    result.add(transformedKernel)
    
    # Log what was created
    let deviceNames = functionMapping.values.toSeq.join(", ")
    hint("autoDeviceKernel created device wrappers: " & deviceNames)
  else:
    # No functions to transform, return original
    result.add(fn)


macro hippoArgs*(args: varargs[untyped]): untyped =
  ## Automatically convert varargs for use with CUDA/HIP.
  ## CUDA/HIP expects an array of arguments or pointers depending on platform.
  when (HippoRuntime == "HIP" and HipPlatform == "nvidia") or HippoRuntime == "HIP_CPU" or HippoRuntime == "SIMPLE":
    # Create a tuple constructor with original arguments
    var tupleNode = newNimNode(nnkTupleConstr)
    for arg in args:
      tupleNode.add(arg)
    result = tupleNode
  else:
    var seqNode = newNimNode(nnkBracket)
    for arg in args:
      seqNode.add(
        quote do:
          cast[ptr pointer](addr `arg`)
      )
    result = quote do:
      @`seqNode`


