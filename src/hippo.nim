## HIP Library for Nim
import
  std/[strformat, os]

proc getHipPlatform*(): string =
  ## getHipPlatform has to be ran at compile time, and gets the target platform for hipcc
  ## NVCC and HIPCC (when building for nvidia) require we pass compiler args in -Xcompiler=""
  ## hipcc uses HIP_PLATFORM to determine amd / nvidia
  ## hipcc defaults to amd if amdclang++ or hip clang is found
  ## https://github.com/ROCm/llvm-project/blob/00fdfae9aeef14c905550601c2218a6b5962f48c/amd/hipcc/bin/hipvars.pm#L131
  let 
    clangPath = getEnv("HIP_CLANG_PATH", "") / "clang++"
    amdClangPath = "/opt/rocm/bin/amdclang++"
    defaultPlatform = if (fileExists(clangPath) or fileExists(amdClangPath)): "amd" else: "nvidia"
    hipPlatform = getEnv("HIP_PLATFORM", defaultPlatform)
  result = hipPlatform

const HipPlatform = getHipPlatform()

# HippoRuntime can be set to "HIP", "HIP_CPU", or "CUDA"
# HIP hipcc will auto detect the runtime of the build system
# HIP_CPU will use the HIP-CPU runtime header
# CUDA will use nvcc

const HippoRuntime* {.strdefine.} = "HIP"

echo &"DEBUG: Using Hippo Runtime: {HippoRuntime}"
if HippoRuntime == "HIP":
  echo &"DEBUG: Using HIP Platform: {HipPlatform}"

when HippoRuntime == "HIP_CPU":
  # Intel TBB is required for HIP-CPU
  {.passL: "-ltbb".}
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
else:
  echo "DEBUG: Using HIP runtime"
  include hip

## Hippo Templates
## nim wrappers around hip and cuda functions

template hippoMalloc*(size: int): pointer =
  ## Allocate memory on the GPU and return a pointer to it
  var p: pointer
  when HippoRuntime == "CUDA":
    handleError(cudaMalloc(addr p, size.cint))
  else:
    handleError(hipMalloc(addr p, size.cint))
  p

template hippoMemcpy*(dst: pointer, src: pointer, size: int, kind: HippoMemcpyKind) =
  ## Copy memory from `src` to `dst`. direction of device and host is determined by `kind`
  when HippoRuntime == "CUDA":
    handleError(cudaMemcpy(dst, src, size.cint, kind))
  else:
    handleError(hipMemcpy(dst, src, size.cint, kind))

template hippoFree*(p: pointer) =
  ## Free memory on the GPU
  when HippoRuntime == "CUDA":
    handleError(cudaFree(p))
  else:
    handleError(hipFree(p))


## Kernel Execution

proc launchKernel*(
  kernel: proc,
  gridDim: Dim3 = newDim3(1,1,1), # default to a grid of 1 block
  blockDim: Dim3 = newDim3(1,1,1),  # default to 1 thread per block
  sharedMemBytes: uint32 = 0, # TODO dynamic shared memory
  stream: HippoStream = nil,
  args: tuple
): HippoError =
  # launchKernel is designed to be similar to `kernel`<<<blockDim, gridDim>>>(args)

  # this function is horrible but it works
  # needs to be refactored to handle all the different runtimes and arguments better

  # having some issues between hip and hip-cpu, so defining different versions of launchKernel
  when HippoRuntime == "HIP" and HipPlatform == "amd":
    # This branch works for all args
    echo "executing HIP"
    var kernelArgs: seq[pointer]
    for key, arg in args.fieldPairs:
      kernelArgs.add(cast[pointer](addr arg))
    result = hipLaunchKernel(
      cast[pointer](kernel),
      gridDim,
      blockDim,
      cast[ptr pointer](addr kernelArgs[0]),
    )
  elif HippoRuntime == "HIP" and HipPlatform == "nvidia":
    # TODO fix args on this branch
    hipLaunchKernelGGL(
      kernel,
      gridDim,
      blockDim,
      0, # TODO
      nil, # TODO
      # TODO handle args properly
      cast[ptr[cint]](args[0]),
      cast[ptr[cint]](args[1]),
      cast[ptr[cint]](args[2])
      )
    result = hipGetLastError()
  elif HippoRuntime == "HIP_CPU":
    # TODO fix args on this branch
    echo "executing kernel on CPU"
    hipLaunchKernelGGL(
      kernel,
      gridDim,
      blockDim,
      0, # TODO
      nil, # TODO
      # TODO handle args properly
      args[0],
      args[1],
      args[2]
    )
    result = hipGetLastError()
  elif HippoRuntime == "CUDA":
    # This branch works for all args
    echo "executing CUDA"
    var kernelArgs: seq[pointer]
    for key, arg in args.fieldPairs:
      kernelArgs.add(cast[pointer](addr arg))
    result = cudaLaunchKernel(
      kernel,
      gridDim,
      blockDim,
      cast[ptr pointer](addr kernelArgs[0])
      #sharedMemBytes,
      #stream
    )
  else:
    raise newException(Exception, &"Unknown runtime: {HippoRuntime}")

template hippoLaunchKernel*(
  kernel: proc,
  gridDim: Dim3 = newDim3(1,1,1), # default to a grid of 1 block
  blockDim: Dim3 = newDim3(1,1,1),  # default to 1 thread per block
  sharedMemBytes: uint32 = 0,
  stream: HippoStream = nil,
  args: tuple
) =
  handleError(launchKernel(kernel, gridDim, blockDim, sharedMemBytes, stream, args))
