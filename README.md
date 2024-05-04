# Hippo

- very much WORK IN PROGRESS

- requires that `hipcc` is in your PATH
  - hip supports both CUDA and HIP, so you can use this with either
  - for CUDA, you will need to have `nvcc` in your PATH

## Motivation

- I want GPU compute (HIP / CUDA) to be a first-class citizen in Nim.
- Ideally, I would like to support both CUDA and HIP, but I am starting with HIP since it can target both AMD and NVIDIA GPUs.
  - however it is understandable to want to switch out with CUDA if you only care about NVIDIA GPUs

- initial example can be found at `tests/hip`
- assuming the hipcc compiler is in your PATH, you can run the example with `im cpp --cc:clang --clang.cpp.exe=hipcc --clang.cpp.linkerexe=hipcc call_params_emit.nim`

## Feature todo list

- [ ] support c++ attributes like `__global__`, `__device__`
- [ ] support kernel execution syntax `<<<1,1>>>` somehow
- [ ] figure out how to handle built-ins like block/thread indices
- [ ] setup automation for building nim types from HIP headers

- [ ] Ensure that every example from the book "CUDA by Example" can be run with this library

- [ ] setup automated testing with hip-cpu
- [ ] setup CI runners for both nvidia & amd GPUs for testing binaries
- [ ] setup automation for building nim types from CUDA headers

## Stretch goals

- [ ] Add a compiler option to use [HIP-CPU](https://github.com/ROCm/HIP-CPU) to run HIP code on the CPU
  - will be useful for debugging and testing
  - also useful for running on systems without a GPU
- [ ] Test with [chipStar](https://github.com/CHIP-SPV/chipStar) verify it can run with openCL
- [ ] Figure out some way to get this to work with WEBGPU