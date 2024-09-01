version     = "0.5.2"
author      = "Andrew Brower"
description = "HIP library for Nim"
license     = "MIT"

srcDir = "src"

requires "nim >= 2.1.9"


task test, "Run all tests":
  exec "for f in tests/test*.nim; do nim cpp -r $f; done"

task test_amd, "Run all tests with AMD HIP":
  exec "for f in tests/test*.nim; do nim cpp --cc:hipcc --define:'useMalloc' --define:'HippoRuntime:HIP' -r $f; done"

task test_cuda, "Run all tests with CUDA":
  exec "for f in tests/test*.nim; do nim cpp --cc:nvcc --define:'useMalloc' -r $f; done"
