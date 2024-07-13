# Hippo

<img src="tests/hip/julia.png">
A Julia Set fractal generated with HIP

## Minimal HIP Example

- IMPORTANT: requires at least nim 2.1.9 for gpu usage
  - nim 2.1.9 added `--cc:nvcc` and `--cc:hipcc`

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

## Functions

- There are 3 sets of function prefixes.
- `hippo*` prefixed functions are friendly nim interfaces for either HIP or CUDA
  - This is the recommended way to use this library, as it is the most nim-like
  - These functions check for errors and raise them as exceptions
- `hip*` prefixed functions are the raw HIP C++ functions
- `cuda*` prefixed functions are the raw CUDA C functions


## Notes

- very much WORK IN PROGRESS
- very basic kernels work that use block/thread indices
- no support for shared memory yet (almost there)

- there are examples for several different targets in the examples directory.
- building for HIP requires that `hipcc` is in your PATH
- [vector_sum_cuda.nims](examples/vector_sum_hippo.nims) example .nims for building with hipcc
- [vector_sum_hippo.nims](examples/vector_sum_cuda.nims) example .nims for building with nvcc for cuda
- [vector_sum_hip_nvidia.nims](examples/vector_sum_hip_nvidia.nims) example .nims for building with hipcc with the cuda backend
  - tell nim that you want the 'nvcc' compiler settings to get the right arguments, but swap out the compiler executable for hipcc
  - you might also need to have HIP_PLATFORM=nvidia set in your environment
- [vector_sum_cpu.nims](examples/vector_sum_cpu.nims) example .nims for building with the HIP-CPU backend (no hipcc required)

- HIP-CPU is supported with the `-d:HippoRuntime=HIP_CPU` flag
  - note: you need to pull the HIP-CPU git submodule to use this feature
  - does not require hipcc. works with gcc.
  - you can write kernels and test them on cpu with echos and breakpoints and everything!

- hippo *requires* that you use cpp. it will not work with c
  - `nim cpp -r tests/hip/vector_sum.nim`
  - The HIP Runtime is a C++ library, and so is the CUDA Runtime. As such, we can't do anything about this.
  - You can make separate projects for the C and C++ parts and link them together. (I have not tested this yet, would love examples if you do this!)

## Motivation

- I want GPU compute (HIP / CUDA) to be a first-class citizen in Nim.
- This library is built around using HIP, which supports compiling for both AMD and NVIDIA GPUs.
  - for CUDA, hipcc is basically a wrapper around `nvcc`.

- examples for each platform can be found in the [examples directory](examples/)
  - assuming the hipcc compiler is in your PATH, you can run the example with `nim cpp -r vector_sum.nim`

## Compiling

- by default, hipcc will build for the GPU detected in your system.
- If you need to build for various GPUs, you can use `--passC:"--offload-arch=gfx1100"` to specify the GPU target
  - for example, to build for a 7900 xtx, you would use `--passC:"--offload-arch=gfx1100"`
  - for a radeon w7500, you would use `--passC:"--offload-arch=gfx1102"`
  - for a geforce 1650 ti, you would use `--passC:"--gpu-architecture=sm_75"`
    - You also have to set the `HIP_PLATFORM=nvidia` environment variable to build for nvidia GPUs if you don't have an nvidia GPU in your system
    - hipcc will pick nvcc by default if you have nvcc but do not have the amd rocm stack installed

- If you want to build for CPU with `-d:"HippoRuntime:HIP_CPU"`, you must have intel tbb libraries installed. `yum install -y tbb-devel`

### Required flags for HIP

- You must compile with nim cpp for c++
- Please refer to the [examples](examples/) for the nim compiler settings needed for each target

### Optional flags

- `-d:HippoRuntime=HIP` (default) (requires hipcc)
  - If you want to build HIP for nvidia, you might need to set the environment variable `HIP_PLATFORM=nvidia` as well
- `-d:HippoRuntime=HIP_CPU` for cpu-only usage (does not require hipcc)
  - you must pull the HIP-CPU submodule to use this feature
- `-d:HippoRuntime=CUDA` (requires nvcc)

## Pragmas

- all pragmas are prefixed with `hippo`

basic kernel example:
```
proc add(a,b: int; c: ptr[int]): {.hippoGlobal.} =
  c[] = a + b
```

## Feature todo list

- [x] support c++ attributes like `__global__`, `__device__`
- [x] support kernel execution syntax `<<<1,1>>>` somehow
- [x] figure out how to handle built-ins like block/thread indices
- [x] Add a compiler option to use [HIP-CPU](https://github.com/ROCm/HIP-CPU) to run HIP code on the CPU
  - will be useful for debugging and testing
  - also useful for running on systems without a GPU
- [x] helper functions to make hip/cuda calls more nim-y
- [x] setup automated testing build for HIP
- [x] setup automated testing with hip-cpu
- [ ] Ensure that every example from the book "CUDA by Example" can be run with this library
- [ ] add pictures to project
- [ ] setup CI runners for both nvidia & amd GPUs for testing binaries
- [ ] setup automation for building nim types from HIP headers
  - the official hip_runtime.h headers are kinda wild
  - hip-cpu headers might be easier to c2nim?
- [ ] setup automation for building nim types from CUDA headers

## Stretch goals

- [ ] Test with [chipStar](https://github.com/CHIP-SPV/chipStar) verify it can run with openCL
- [ ] Figure out some way to get this to work with WEBGPU