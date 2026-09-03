import
  std/[unittest, math, random],
  hippo,
  ./utils

const
  ReduceCount = 1024
  BlockThreads = uint32(ReduceCount)
  MaxWarpsPerBlock = 64
  SubWarpWidth = 8
  RandomSeed = 1337
  RelativeTolerance = 1e-5

proc blockReduceKernel(data: ptr cfloat, res: ptr cfloat) {.hippoGlobal.} =
  ## Reduce one value per thread into a single block-wide sum.
  var blockWarpSums {.hippoShared.}: array[MaxWarpsPerBlock, cfloat]
  let
    arr = cast[ptr UncheckedArray[cfloat]](data)
    outArr = cast[ptr UncheckedArray[cfloat]](res)
    tid = int(threadIdx.x)
    total = blockReduceSum(arr[tid], addr blockWarpSums[0])
  if tid == 0:
    outArr[0] = total

proc warpReduceKernel(data: ptr cfloat, res: ptr cfloat) {.hippoGlobal.} =
  ## Reduce a single warp of values to their sum and maximum.
  let
    arr = cast[ptr UncheckedArray[cfloat]](data)
    outArr = cast[ptr UncheckedArray[cfloat]](res)
    lane = int(threadIdx.x)
    laneSum = warpReduceSum(arr[lane])
    laneMax = warpReduceMax(arr[lane])
  if lane == 0:
    outArr[0] = laneSum
    outArr[1] = laneMax

proc shuffleKernel(uintOut: ptr cuint, floatOut: ptr cfloat) {.hippoGlobal.} =
  ## Exercise the uint32, xor and width-taking shuffle wrappers.
  let
    uArr = cast[ptr UncheckedArray[cuint]](uintOut)
    fArr = cast[ptr UncheckedArray[cfloat]](floatOut)
    lane = int(threadIdx.x)
  uArr[lane] = hippoShflDown(cuint(lane), 1)
  fArr[lane] = hippoShflXor(cfloat(lane), 1)
  fArr[HippoWarpSize + lane] = hippoShflDown(cfloat(lane), 1, SubWarpWidth)

proc kahanSum(values: seq[cfloat]): float64 =
  ## Sum values on the host with Kahan compensation for an accurate reference.
  var compensation = 0.0
  for value in values:
    let
      y = float64(value) - compensation
      t = result + y
    compensation = (t - result) - y
    result = t

suite "warp reduction":
  testSkipPlatforms "block reduce sum matches a host kahan sum", "SIMPLE":
    var
      rng = initRand(RandomSeed)
      hostData: seq[cfloat] = @[]
      hostResult = @[0.0'f32]
    for _ in 0 ..< ReduceCount:
      hostData.add(cfloat(rng.rand(1.0)))
    let
      devData = hippoMalloc(ReduceCount * sizeof(cfloat))
      devResult = hippoMalloc(sizeof(cfloat))

    hippoMemcpy(devData, addr hostData[0], ReduceCount * sizeof(cfloat), HippoMemcpyHostToDevice)
    hippoLaunchKernel(
      blockReduceKernel,
      gridDim = newDim3(1'u32),
      blockDim = newDim3(BlockThreads),
      args = hippoArgs(devData.p, devResult.p)
    )
    hippoSynchronize()
    hippoMemcpy(addr hostResult[0], devResult, sizeof(cfloat), HippoMemcpyDeviceToHost)

    let expected = kahanSum(hostData)
    check abs(float64(hostResult[0]) - expected) / expected < RelativeTolerance

  testSkipPlatforms "warp reduce sum and max over one warp", "SIMPLE":
    var
      rng = initRand(RandomSeed + 1)
      hostData: seq[cfloat] = @[]
      hostResult = @[0.0'f32, 0.0'f32]
    for _ in 0 ..< HippoWarpSize:
      hostData.add(cfloat(rng.rand(1.0)))
    let
      devData = hippoMalloc(HippoWarpSize * sizeof(cfloat))
      devResult = hippoMalloc(2 * sizeof(cfloat))

    hippoMemcpy(devData, addr hostData[0], HippoWarpSize * sizeof(cfloat), HippoMemcpyHostToDevice)
    hippoLaunchKernel(
      warpReduceKernel,
      gridDim = newDim3(1'u32),
      blockDim = newDim3(uint32(HippoWarpSize)),
      args = hippoArgs(devData.p, devResult.p)
    )
    hippoSynchronize()
    hippoMemcpy(addr hostResult[0], devResult, 2 * sizeof(cfloat), HippoMemcpyDeviceToHost)

    let expected = kahanSum(hostData)
    check abs(float64(hostResult[0]) - expected) / expected < RelativeTolerance
    check hostResult[1] == max(hostData)

  testSkipPlatforms "shuffle wrappers move lane values", "SIMPLE":
    var
      hostUints = newSeq[cuint](HippoWarpSize)
      hostFloats = newSeq[cfloat](2 * HippoWarpSize)
    let
      devUints = hippoMalloc(HippoWarpSize * sizeof(cuint))
      devFloats = hippoMalloc(2 * HippoWarpSize * sizeof(cfloat))

    hippoLaunchKernel(
      shuffleKernel,
      gridDim = newDim3(1'u32),
      blockDim = newDim3(uint32(HippoWarpSize)),
      args = hippoArgs(devUints.p, devFloats.p)
    )
    hippoSynchronize()
    hippoMemcpy(addr hostUints[0], devUints, HippoWarpSize * sizeof(cuint), HippoMemcpyDeviceToHost)
    hippoMemcpy(addr hostFloats[0], devFloats, 2 * HippoWarpSize * sizeof(cfloat),
                HippoMemcpyDeviceToHost)

    for lane in 0 ..< HippoWarpSize:
      let downLane = if lane + 1 < HippoWarpSize: lane + 1 else: lane
      check hostUints[lane] == cuint(downLane)
      check hostFloats[lane] == cfloat(lane xor 1)
      let subLane = if (lane mod SubWarpWidth) + 1 < SubWarpWidth: lane + 1 else: lane
      check hostFloats[HippoWarpSize + lane] == cfloat(subLane)
