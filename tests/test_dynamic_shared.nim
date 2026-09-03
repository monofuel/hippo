import
  std/unittest,
  hippo

when HippoRuntime != "SIMPLE":
  const
    ThreadsPerBlock = 256'u32
    SharedBytes = uint32(256 * sizeof(cfloat))

  proc dynamicSharedKernel(outs: ptr cfloat) {.hippoGlobal.} =
    ## Stage values through dynamic shared memory and write them back reversed.
    let
      smem = hippoDynamicShared(cfloat)
      outArr = cast[ptr UncheckedArray[cfloat]](outs)
      tid = int(threadIdx.x)
    smem[tid] = cfloat(tid) * 2.0'f32
    hippoSyncthreads()
    outArr[tid] = smem[int(blockDim.x) - 1 - tid]

  suite "dynamic shared memory":
    test "dynamic shared memory round trips values":
      var hostOuts = newSeq[cfloat](int(ThreadsPerBlock))
      let devOuts = hippoMalloc(int(ThreadsPerBlock) * sizeof(cfloat))

      hippoLaunchKernel(
        dynamicSharedKernel,
        gridDim = newDim3(1'u32),
        blockDim = newDim3(ThreadsPerBlock),
        sharedMemBytes = SharedBytes,
        args = hippoArgs(devOuts.p)
      )
      hippoSynchronize()
      hippoMemcpy(addr hostOuts[0], devOuts, int(ThreadsPerBlock) * sizeof(cfloat),
                  HippoMemcpyDeviceToHost)

      for i in 0 ..< int(ThreadsPerBlock):
        check hostOuts[i] == cfloat(int(ThreadsPerBlock) - 1 - i) * 2.0'f32
