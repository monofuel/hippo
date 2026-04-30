import
  hippo,
  std/[unittest],
  ./utils

const
  N = 1024
  ThreadsPerBlock = 256
  NumBlocks = 4

proc twoPhaseKernel(data, scratch: ptr[cfloat], n: cint) {.hippoGlobal.} =
  let globalTid = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x)
  let stride = int(gridDim.x) * int(blockDim.x)

  # Phase 1: each thread writes its global ID as float
  var i = globalTid
  while i < n:
    cast[ptr UncheckedArray[cfloat]](data)[i] = cfloat(i)
    i += stride

  hippoGridSync()

  # Phase 2: each thread reads the NEXT element (written by another block in phase 1)
  i = globalTid
  while i < n:
    let readIdx = (i + 1) mod n
    cast[ptr UncheckedArray[cfloat]](scratch)[i] =
      cast[ptr UncheckedArray[cfloat]](data)[readIdx] * 2.0f
    i += stride

suite "cooperative launch":
  testSkipPlatforms "grid sync two-phase", "SIMPLE", "HIP_CPU":
    var dev_data = hippoMalloc(N * sizeof(cfloat))
    var dev_scratch = hippoMalloc(N * sizeof(cfloat))

    var nElems = cint(N)
    hippoLaunchCooperative(
      twoPhaseKernel,
      gridDim = newDim3(NumBlocks.uint32),
      blockDim = newDim3(ThreadsPerBlock.uint32),
      args = hippoArgs(dev_data.p, dev_scratch.p, nElems))

    hippoSynchronize()

    var result: array[N, cfloat]
    hippoMemcpy(addr result[0], dev_scratch, N * sizeof(cfloat),
                HippoMemcpyDeviceToHost)

    for i in 0 ..< N:
      let expected = cfloat(((i + 1) mod N) * 2)
      assertAlmostEqual(float64(result[i]), float64(expected), 1e-5)
