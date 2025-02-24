import hippo

proc addKernel*(a, b: cint; c: ptr[cint]) {.hippoGlobal.} =
  c[] = a + b

var
  c: int32
  dev_c = hippoMalloc(sizeof(int32))
hippoLaunchKernel(
  addKernel,
  args = hippoArgs(
    cast[pointer](2),
    cast[pointer](7),
    dev_c.p
  )
)
hippoMemcpy(addr c, dev_c, sizeof(int32), hipMemcpyDeviceToHost)
echo "2 + 7 = ", c