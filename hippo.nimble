version     = "0.9.0"
author      = "Andrew Brower"
description = "HIP library for Nim"
license     = "MIT"

srcDir = "src"

requires "nim >= 2.1.9"


task test, "Run all tests":
  # default is currently HIP_CPU for being easy to test and accurate
  exec "set -e; for f in tests/test*.nim; do nim cpp -r $f; done"


task test_amd, "Run all tests with AMD HIP":
  exec "set -e; for f in tests/test*.nim; do nim cpp --cc:hipcc --define:'useMalloc' --define:'HippoRuntime:HIP' -r $f; done"

task test_cuda, "Run all tests with CUDA":
  exec "set -e; for f in tests/test*.nim; do nim cpp --cc:nvcc --define:'useMalloc' -r $f; done"

task test_hip_cpu, "Run all tests with HIP-CPU backend":
  # explicit hip_cpu test in case we ever change the default for our tests
  exec "set -e; for f in tests/test*.nim; do nim cpp --define:'HippoRuntime:HIP_CPU' -r $f; done"

task test_simple, "Run all tests with Simple backend (threads enabled)":
  # "simple with threads on" means we are using the Hippo backend to run kernels on CPU using Nim's threading.
  # we run many many gpu threads on top of a pool of cpu threads.
  exec "set -e; for f in tests/test*.nim; do nim cpp --threads:on --define:'HippoRuntime:SIMPLE' -r $f; done"

task test_simple_no_threads, "Run all tests with Simple backend (threads disabled)":
  # "simple with threads off" means we are using the Hippo backend to run kernels on CPU using a single cpu thread.
  # we run many many gpu threads on top of a single cpu thread.
  exec "set -e; for f in tests/test*.nim; do nim cpp --threads:off --define:'HippoRuntime:SIMPLE' -r $f; done"
