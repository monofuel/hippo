version     = "0.0.2"
author      = "Andrew Brower"
description = "HIP library for Nim"
license     = "MIT"

srcDir = "src"

requires "nim >= 2.1.9"


task test, "Run all tests":
  exec "for f in tests/test*.nim; do nim cpp -r $f; done"
