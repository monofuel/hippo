## Put your tests here.

import std/[unittest, osproc]

suite "hip tests":
    test "call_params.nim":
      # test building call_params.nim example
      let res = execCmd("nim cpp tests/hip/call_params.nim")
      assert res == 0
