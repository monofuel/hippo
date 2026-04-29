import
  hippo,
  std/[unittest, math]

when HippoRuntime == "HIP":
  proc wmmaMatmulKernel(
    aData, bData: ptr cushort,
    cData: ptr cfloat
  ) {.hippoGlobal.} =
    let tid = int(threadIdx.x)
    if tid >= 32:
      return
    let a = cast[ptr UncheckedArray[cushort]](aData)
    let b = cast[ptr UncheckedArray[cushort]](bData)
    let c = cast[ptr UncheckedArray[cfloat]](cData)

    var fragA {.noinit.}: HippoWmmaHalf16
    var fragB {.noinit.}: HippoWmmaHalf16
    var fragC = hippoWmmaZeroF32()

    for i in 0 ..< 16:
      hippoWmmaSetF16(fragA, i, a[tid * 16 + i])
    for i in 0 ..< 16:
      hippoWmmaSetF16(fragB, i, b[tid * 16 + i])

    fragC = hippoWmmaF32_16x16x16_f16(fragA, fragB, fragC)

    for i in 0 ..< 8:
      c[tid * 8 + i] = hippoWmmaGetF32(fragC, i)

  suite "wmma":
    test "wmma ones matmul":
      var hostA: array[32 * 16, cushort]
      var hostB: array[32 * 16, cushort]
      var hostC: array[32 * 8, cfloat]

      for i in 0 ..< 32 * 16:
        hostA[i] = hippoFloatToHalf(1.0'f32)
        hostB[i] = hippoFloatToHalf(1.0'f32)

      let devA = hippoMalloc(32 * 16 * sizeof(cushort))
      let devB = hippoMalloc(32 * 16 * sizeof(cushort))
      let devC = hippoMalloc(32 * 8 * sizeof(cfloat))

      hippoMemcpy(devA, addr hostA[0], cint(32 * 16 * sizeof(cushort)), HippoMemcpyHostToDevice)
      hippoMemcpy(devB, addr hostB[0], cint(32 * 16 * sizeof(cushort)), HippoMemcpyHostToDevice)

      let grid = newDim3(1'u32)
      let blk = newDim3(32'u32)
      hippoLaunchKernel(wmmaMatmulKernel, gridDim = grid, blockDim = blk,
                        args = hippoArgs(devA.p, devB.p, devC.p))
      discard hipDeviceSynchronize()

      hippoMemcpy(addr hostC[0], devC, cint(32 * 8 * sizeof(cfloat)), HippoMemcpyDeviceToHost)

      for i in 0 ..< 32 * 8:
        check hostC[i] == 16.0'f32

    test "wmma accumulate":
      var hostA: array[32 * 16, cushort]
      var hostB: array[32 * 16, cushort]
      var hostC: array[32 * 8, cfloat]

      for i in 0 ..< 32 * 16:
        hostA[i] = hippoFloatToHalf(2.0'f32)
        hostB[i] = hippoFloatToHalf(3.0'f32)

      let devA = hippoMalloc(32 * 16 * sizeof(cushort))
      let devB = hippoMalloc(32 * 16 * sizeof(cushort))
      let devC = hippoMalloc(32 * 8 * sizeof(cfloat))

      hippoMemcpy(devA, addr hostA[0], cint(32 * 16 * sizeof(cushort)), HippoMemcpyHostToDevice)
      hippoMemcpy(devB, addr hostB[0], cint(32 * 16 * sizeof(cushort)), HippoMemcpyHostToDevice)

      let grid = newDim3(1'u32)
      let blk = newDim3(32'u32)
      hippoLaunchKernel(wmmaMatmulKernel, gridDim = grid, blockDim = blk,
                        args = hippoArgs(devA.p, devB.p, devC.p))
      discard hipDeviceSynchronize()

      hippoMemcpy(addr hostC[0], devC, cint(32 * 8 * sizeof(cfloat)), HippoMemcpyDeviceToHost)

      for i in 0 ..< 32 * 8:
        check hostC[i] == 96.0'f32
