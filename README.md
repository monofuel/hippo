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
- assuming the hipcc compiler is in your PATH, you can run the example with `nim cpp --cc:clang --clang.cpp.exe:hipcc --clang.cpp.linkerexe:hipcc call_params_emit.nim`

## Compiling

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

example for nvcc (untested):
```
--cc:gcc
--clang.cpp.exe:nvcc
--clang.cpp.linkerexe:nvcc
```


### Optional flags

- `-d:HippoRuntime=HIP` (default) (requires hipcc)
- `-d:HippoRuntime=HIP_CPU` for cpu-only usage (does not require hipcc)
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

- [ ] Add a compiler option to use [HIP-CPU](https://github.com/ROCm/HIP-CPU) to run HIP code on the CPU
  - will be useful for debugging and testing
  - also useful for running on systems without a GPU
- [ ] add pictures to project
- [ ] setup automated testing build for HIP
  - need a docker image with nim, hipcc, and nvcc
- [ ] setup automated testing with hip-cpu
- [ ] setup CI runners for both nvidia & amd GPUs for testing binaries
- [ ] setup automation for building nim types from CUDA headers

## Stretch goals

- [ ] Test with [chipStar](https://github.com/CHIP-SPV/chipStar) verify it can run with openCL
- [ ] Figure out some way to get this to work with WEBGPU