# Hippo backends

Hippo supports four backends for GPU and CPU execution.

- AMD (HIP/ROCm)
- NVIDIA (CUDA)
- HIP-CPU (HIP but compiling for CPU)
- Simple (nim threads, can be parallel or single threaded)

- Hippo should generally auto-detect the backend you want to use based on the compiler and target, however these settings can be overridden for special environments (eg: if you have both AMD and NVIDIA compilers installed, you can set the platform to force one or the other).
- the backend can be set with HippoRuntime.
  - `-d:HippoRuntime=HIP` (default) (requires hipcc)
  - `-d:HippoRuntime=CUDA` (requires nvcc)
  - `-d:HippoRuntime=HIP_CPU` for cpu-only usage (does not require hipcc)
  - `-d:HippoRuntime=SIMPLE` for CPU-only nim threading.
- the HIP runtime also supports two platforms, AMD and NVIDIA.
  - `-d:HippoPlatform=amd` forces AMD platform (bypasses auto-detection)
  - `-d:HippoPlatform=nvidia` forces NVIDIA platform (bypasses auto-detection)
  - If not specified, hippo auto-detects the platform based on the presence of amdclang++ (this mirrors the behavior of how hipcc detects the platform).

## AMD (HIP/ROCm)

Uses the HIP runtime to target AMD GPUs via ROCm. Requires `nim cpp` compilation.
the HIP runtime can also target NVIDIA GPUs via HIP_PLATFORM=nvidia.

**Known pitfalls:**
- Platform detection may fail if the compiler cannot be found or both AMD and NVIDIA compilers are present. Set `HIP_PLATFORM=amd` environment variable to force AMD platform detection.

## NVIDIA (CUDA)

Uses the CUDA runtime with nvcc compiler to target NVIDIA GPUs. Requires `nim cpp` compilation.

**Known issues:**
- The nvcc compiler may produce warnings about the C++ code that Nim generates. These warnings are mostly harmless.
  - TODO make better!

## HIP-CPU

Supports using Hippo (and all hip functions) on cpu. very useful for debugging and testing. works great with debuggers and breakpoints.

Please review the HIP-CPU readme for documentation and limitations. eg: warps do not run in lockstep, and inline PTX obviously does not work on cpu.

Uses the HIP-CPU runtime to execute HIP code on CPU. Requires `nim cpp` compilation and Intel TBB library (`libtbb`). 

**Known issues:**
- Requires `libtbb` to be installed, which can be picky on certain Linux distributions. (there's an included nix flake that works for me on NixOS and WSL, I think most modern distros should work fine with libtbb installed)
- Uses HIP-CPU headers from the bundled HIP-CPU subdirectory. (make sure to pull submodules!)

## Simple

Pure Nim threads backend that runs kernels on CPU using Nim's threading. Can compile with `nim js`, `nim c` or `nim cpp`. Works without threads enabled (falls back to single-threaded execution).
nim 2.0 has threads enabled by default now.
nim js only supports single threaded execution currently.

The simple backend is extremely useful for translating existing CPU code to GPU code. it works perfectly with debuggers and breakpoints.

you still should be using `hippoMalloc` to get shared memory allocations that are thread-safe. Certain garbage collectors might work fine out of the box, your mileage may vary and I hope you know what you're doing.

**Known issues:**
- Only parallelizes blocks, not threads within blocks. A single block with many threads will run all threads on one CPU thread.
  - blocks doesn't really make sense in the context of CPU, but could maybe use more thinking.
- `hippoSyncthreads()` is not currently implemented and will raise an exception.
- for nim before 2.0, requires `--threads:on` for multi-core performance. Without threads, execution is limited to a single core.
- does not support parallelism on js currently.
