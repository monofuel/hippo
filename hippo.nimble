version     = "0.0.1"
author      = "Andrew Brower"
description = "HIP library for Nim"
license     = "MIT"

srcDir = "src"

requires "nim >= 2.0.0"


task test, "Run all tests":
  exec "for f in tests/test*.nim; do nim cpp -r $f; done"
