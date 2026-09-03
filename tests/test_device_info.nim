import
  std/unittest,
  hippo

suite "device info":
  test "device info reports plausible values":
    let info = hippoDeviceInfo()
    when HippoRuntime == "SIMPLE":
      check info.cuCount == 1
      check info.ldsBytes == 0
      check info.warpSize == 1
    else:
      check info.cuCount > 0
      check info.ldsBytes > 0
      check info.warpSize > 0
