import hippo

proc addKernel*(a, b: cint; c: ptr[cint]) {.hippoGlobal.} =
  c[] = a + b


proc main() =
  echo "DEBUG: init"
  var c: int32
  var dev_c: ptr[int32]
  discard hipMalloc(cast[ptr pointer](addr dev_c), sizeof(int32).cint)
  echo "DEBUG: hipMalloc"
  discard launchKernel(
    addKernel,
    args = (2,7,dev_c)
  )

  discard hipMemcpy(addr c, dev_c, sizeof(int32).cint, hipMemcpyDeviceToHost)
  echo "DEBUG: hipMemcpy"
  echo "2 + 7 = ", c
  discard hipFree(dev_c)
  echo "DEBUG: hipFree"
  echo "DEBUG: done"

when isMainModule:
  main()