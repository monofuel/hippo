## HIP Library for Nim
import
  std/[strformat]

# HippoRuntime can be set to "HIP", "HIP_CPU", or "CUDA"
# HIP hipcc will auto detect the runtime of the build system
# HIP_CPU will use the HIP-CPU runtime header
# CUDA will use nvcc

const HippoRuntime* {.strdefine.} = "HIP"

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
  # nvcc loads the CUDA runtime automatically
  # Note: i have not actually setup any CUDA stuff yet
  echo "DEBUG: Using CUDA runtime"
  include cuda
else:
  echo "DEBUG: Using HIP runtime"
  include hip



# Kernel Execution

proc launchKernel*(
  kernel: proc,
  gridDim: Dim3 = newDim3(1,1,1), # default to a grid of 1 block
  blockDim: Dim3 = newDim3(1,1,1),  # default to 1 thread per block
  sharedMemBytes: uint32 = 0,
  stream: hipStream_t = nil,
  args: tuple
): hipError_t =
  # launchKernel is designed to be similar to `kernel`<<<blockDim, gridDim>>>(args)

  # having some issues between hip and hip-cpu, so defining different versions of launchKernel
  when HippoRuntime == "HIP":
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
  elif HippoRuntime == "HIP_CPU":
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
    result = hipDeviceSynchronize()
  else:
    raise newException(Exception, &"Unknown runtime: {HippoRuntime}")
  if result != 0:
    return result