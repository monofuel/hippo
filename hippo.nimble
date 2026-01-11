version     = "0.7.0"
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

task test_simple, "Run all tests with Simple backend (threads enabled)":
  # TODO: Get dot_product working on simple backend
  exec "set -e; nim cpp --threads:on --define:'HippoRuntime:SIMPLE' -r tests/test_vector_sum.nim"

task test_simple_no_threads, "Run all tests with Simple backend (threads disabled)":
  # TODO: Get dot_product working on simple backend
  exec "set -e; nim cpp --threads:off --define:'HippoRuntime:SIMPLE' -r tests/test_vector_sum.nim"
