import hippo

proc addKernel*(a, b: cint; c: ptr[cint]) {.hippoGlobal.} =
  c[] = a + b

proc main() =
  echo "DEBUG: init"
  var c: int32
  var dev_c: ptr[int32]
  discard hipMalloc(cast[ptr pointer](addr dev_c), sizeof(int32).cint)
  echo "DEBUG: hipMalloc"
  var arg1 = 2
  var arg2 = 7
  var args = (addr arg1, addr arg2, addr dev_c)
  discard hipLaunchKernel(
    cast[pointer](addKernel),
    newDim3(1, 1, 1),
    newDim3(1, 1, 1),
    cast[ptr pointer](addr args))
  discard hipMemcpy(addr c, dev_c, sizeof(int32).cint, hipMemcpyDeviceToHost)
  echo "DEBUG: hipMemcpy"
  echo "2 + 7 = ", c
  discard hipFree(dev_c)
  echo "DEBUG: hipFree"
  echo "DEBUG: done"

when isMainModule:
  main()