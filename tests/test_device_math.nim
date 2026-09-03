import
  std/[unittest, math],
  hippo

const
  SampleCount = 1024
  ThreadsPerBlock = 256'u32
  BlocksPerGrid = uint32(SampleCount div 256)
  FunctionCount = 6
  RelativeTolerance = 1e-6

proc deviceMathKernel(xs: ptr cfloat, ps: ptr cfloat, outs: ptr cfloat) {.hippoGlobal.} =
  ## Evaluate every wrapped device math function over the sampled inputs.
  let idx = int(threadIdx.x + blockIdx.x * blockDim.x)
  if idx < SampleCount:
    let
      xArr = cast[ptr UncheckedArray[cfloat]](xs)
      pArr = cast[ptr UncheckedArray[cfloat]](ps)
      outArr = cast[ptr UncheckedArray[cfloat]](outs)
      x = xArr[idx]
      p = pArr[idx]
    outArr[0 * SampleCount + idx] = hippoTanhf(x)
    outArr[1 * SampleCount + idx] = hippoCoshf(x)
    outArr[2 * SampleCount + idx] = hippoRsqrtf(p)
    outArr[3 * SampleCount + idx] = hippoFminf(x, p)
    outArr[4 * SampleCount + idx] = hippoFmaf(x, p, x)
    outArr[5 * SampleCount + idx] = hippoExpf(x)

proc checkRelative(actual: cfloat, expected: float64) =
  ## Assert a device result matches the host reference within the relative tolerance.
  let scale = max(abs(expected), 1.0)
  let diff = abs(float64(actual) - expected)
  check diff / scale < RelativeTolerance

suite "device math":
  test "device math wrappers match host math":
    var
      hostXs: seq[cfloat] = @[]
      hostPs: seq[cfloat] = @[]
      hostOuts = newSeq[cfloat](FunctionCount * SampleCount)
    for i in 0 ..< SampleCount:
      let x = -4.0 + 8.0 * float64(i) / float64(SampleCount - 1)
      hostXs.add(cfloat(x))
      hostPs.add(cfloat(abs(x) + 0.5))

    let
      devXs = hippoMalloc(SampleCount * sizeof(cfloat))
      devPs = hippoMalloc(SampleCount * sizeof(cfloat))
      devOuts = hippoMalloc(FunctionCount * SampleCount * sizeof(cfloat))

    hippoMemcpy(devXs, addr hostXs[0], SampleCount * sizeof(cfloat), HippoMemcpyHostToDevice)
    hippoMemcpy(devPs, addr hostPs[0], SampleCount * sizeof(cfloat), HippoMemcpyHostToDevice)
    hippoLaunchKernel(
      deviceMathKernel,
      gridDim = newDim3(BlocksPerGrid),
      blockDim = newDim3(ThreadsPerBlock),
      args = hippoArgs(devXs.p, devPs.p, devOuts.p)
    )
    hippoSynchronize()
    hippoMemcpy(addr hostOuts[0], devOuts, FunctionCount * SampleCount * sizeof(cfloat),
                HippoMemcpyDeviceToHost)

    for i in 0 ..< SampleCount:
      let
        x = float64(hostXs[i])
        p = float64(hostPs[i])
      checkRelative(hostOuts[0 * SampleCount + i], math.tanh(x))
      checkRelative(hostOuts[1 * SampleCount + i], math.cosh(x))
      checkRelative(hostOuts[2 * SampleCount + i], 1.0 / math.sqrt(p))
      checkRelative(hostOuts[3 * SampleCount + i], min(x, p))
      checkRelative(hostOuts[4 * SampleCount + i], x * p + x)
      checkRelative(hostOuts[5 * SampleCount + i], math.exp(x))
