import hippo

proc addKernel*(a, b: cint; c: ptr[cint]) {.hippoGlobal.} =
  c[] = a + b


proc main() =
  var c: int32
  var dev_c: pointer
  handleError(hipMalloc(addr dev_c, sizeof(int32).cint))
  handleError(launchKernel(
    addKernel,
    args = (2,7,dev_c)
  ))
  handleError(hipMemcpy(addr c, dev_c, sizeof(int32).cint, hipMemcpyDeviceToHost))
  echo "2 + 7 = ", c
  handleError(hipFree(dev_c))

when isMainModule:
  main()