# Hippo

## Minimal Example

```
import hippo

proc addKernel*(a, b: cint; c: ptr[cint]) {.hippoGlobal.} =
  c[] = a + b

var c: int32
var dev_c: ptr[int32]
handleError(hipMalloc(cast[ptr pointer](addr dev_c), sizeof(int32).cint))
handleError(launchKernel(addKernel,args = (2,7,dev_c)))
handleError(hipMemcpy(addr c, dev_c, sizeof(int32).cint, hipMemcpyDeviceToHost))
echo "2 + 7 = ", c
handleError(hipFree(dev_c))
```


## Notes

- very much WORK IN PROGRESS
- very basic kernels work that use block/thread indices
- no support for shared memory yet

- hippo *requires* that you use cpp. it will not work with c
  - `nim cpp -r tests/hip/vector_sum.nim`
  - The HIP Runtime is a C++ library, and so is the CUDA Runtime. As such, we can't do anything about this.
  - You can make separate projects for the C and C++ parts and link them together. (I have not tested this yet, would love examples if you do this!)

- building for GPU requires that `hipcc` is in your PATH
  - HIP supports both CUDA and ROCM, so you can use this with either
  - for CUDA, you will need to have `nvcc` in your PATH as well as `hipcc`
    - set the environment variable HIP_PLATFORM=nvidia for hipcc to build for nvidia
    - HIP nvidia compiling still needs work: https://github.com/monofuel/hippo/pull/1
      - I got a hacky fix to work, but it needs to be polished up and requires small changes in the nim compiler
- HIP-CPU is supported with the `-d:HippoRuntime=HIP_CPU` flag
  - note: you need to pull the HIP-CPU git submodule to use this feature
  - does not require hipcc. works with gcc.
  - you can write kernels and test them on cpu with breakpoints and everything!

## Motivation

- I want GPU compute (HIP / CUDA) to be a first-class citizen in Nim.
- This library is built around using HIP, which supports compiling for both AMD and NVIDIA GPUs.
  - for CUDA, hipcc is basically a wrapper around `nvcc`.

- initial example can be found at `tests/hip`
  - assuming the hipcc compiler is in your PATH, you can run the example with `nim cpp -r vector_sum.nim`
- cpu examples are at `tests/hip_cpu`

- in theory, this library could also directly support `nvcc` and `cuda_runtime.h` as well, but I have not worked on that yet.
  - pls send me a H100 if you want me to work on that

## Compiling

- by default, hipcc will build for the GPU detected in your system.
- If you need to build for various GPUs, you can use --passC:"--offload-arch=gfx1100" to specify the GPU target
  - for example, to build for a 7900 xtx, you would use --passC:"--offload-arch=gfx1100"
  - for a radeon w7500, you would use --passC:"--offload-arch=gfx1102"
  - for a geforce 1650 ti, you would use --passC:"--gpu-architecture=sm_75" 
    - You also have to set the `HIP_PLATFORM=nvidia` environment variable to build for nvidia GPUs if you don't have an nvidia GPU in your system


### Required flags

You must compile with nim cpp for c++
You must also set the compiler type to 'clang' and the exe to 'hipcc'

example cli: `nim cpp --cc:clang --clang.cpp.exe:hipcc --clang.cpp.linkerexe:hipcc filename.nim`

example config.nims:
```
--cc:clang
--clang.cpp.exe:hipcc
--clang.cpp.linkerexe:hipcc
```

nvcc handles compiler and linker arguments differently than clang, and requires changes to the nim compiler to work properly at the moment.


### Optional flags

- `-d:HippoRuntime=HIP` (default) (requires hipcc)
- `-d:HippoRuntime=HIP_CPU` for cpu-only usage (does not require hipcc)
  - you must pull the HIP-CPU submodule to use this feature
- no cuda features implemented yet, but should be technically possible.

## Pragmas

- all pragmas are prefixed with `hippo` to avoid conflicts

basic kernel example:
```
proc add(a,b: int; c: ptr[int]): {.hippoGlobal.} =
  c[] = a + b
```

## Feature todo list

- [x] support c++ attributes like `__global__`, `__device__`
- [x] support kernel execution syntax `<<<1,1>>>` somehow
- [x] figure out how to handle built-ins like block/thread indices
- [ ] setup automation for building nim types from HIP headers
  - the official hip_runtime.h headers are kinda wild
  - hip-cpu headers might be easier to c2nim?

- [ ] Ensure that every example from the book "CUDA by Example" can be run with this library

- [x] Add a compiler option to use [HIP-CPU](https://github.com/ROCm/HIP-CPU) to run HIP code on the CPU
  - will be useful for debugging and testing
  - also useful for running on systems without a GPU
- [ ] helper functions to make hip/cuda calls more nim-y
- [ ] add pictures to project
- [x] setup automated testing build for HIP
- [x] setup automated testing with hip-cpu
- [ ] setup CI runners for both nvidia & amd GPUs for testing binaries
- [ ] setup automation for building nim types from CUDA headers

## Stretch goals

- [ ] Test with [chipStar](https://github.com/CHIP-SPV/chipStar) verify it can run with openCL
- [ ] Figure out some way to get this to work with WEBGPU