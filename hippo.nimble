version     = "0.8.0"
author      = "Andrew Brower"
description = "HIP library for Nim"
license     = "MIT"

srcDir = "src"

requires "nim >= 2.1.9"


task test, "Run all tests":
  # uses HIP_CPU
  exec "set -e; for f in tests/test*.nim; do nim cpp -r $f; done"

task test_amd, "Run all tests with AMD HIP":
  exec "set -e; for f in tests/test*.nim; do nim cpp --cc:hipcc --define:'useMalloc' --define:'HippoRuntime:HIP' -r $f; done"

task test_cuda, "Run all tests with CUDA":
  exec "set -e; for f in tests/test*.nim; do nim cpp --cc:nvcc --define:'useMalloc' -r $f; done"

task test_hip_cpu, "Run all tests with HIP-CPU backend":
  exec "set -e; for f in tests/test*.nim; do nim cpp --define:'HippoRuntime:HIP_CPU' -r $f; done"

task test_simple, "Run all tests with Simple backend (threads enabled)":
  exec "set -e; for f in tests/test*.nim; do nim cpp --threads:on --define:'HippoRuntime:SIMPLE' -r $f; done"

task test_simple_no_threads, "Run all tests with Simple backend (threads disabled)":
  exec "set -e; for f in tests/test*.nim; do nim cpp --threads:off --define:'HippoRuntime:SIMPLE' -r $f; done"
