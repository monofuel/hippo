import hippo


## asmNoStackFrame is required to avoid a bunch of nim stuff
func add*(a,b: cint; c: ptr[cint]) {.exportc, asmNoStackFrame.} =
  {.push checks: off}
  c[] = a + b
  {.pop}


proc main() =
  echo "DEBUG: init"
  var c: int32
  var dev_c: ptr[int32]
  discard hipMalloc(cast[ptr pointer](addr dev_c), sizeof(int32).cint)
  echo "DEBUG: hipMalloc"
  # TODO implement <<< >>> syntax
  {.emit: """
  //add<<<1,1>>>(2,7,dev_c);
  add(2,7,dev_c);
  """.}
  discard hipMemcpy(addr c, dev_c, sizeof(int32).cint, hipMemcpyDeviceToHost)
  echo "DEBUG: hipMemcpy"
  echo "2 + 7 = ", c
  discard hipFree(dev_c)
  echo "DEBUG: hipFree"
  echo "DEBUG: done"



when isMainModule:
  main()