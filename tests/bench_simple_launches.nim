## Measure cold and warmed synchronous launches with `nimble bench_simple`.
## Keep this workload unchanged when comparing worker scheduler implementations.
import
  std/[monotimes, times, algorithm, strformat],
  hippo

const
  Lanes = 32
  Warmup = 50
  Samples = 1000

proc smallKernel(output: ptr cint; seed: cint) {.hippoGlobal.} =
  ## Do a small amount of independent work per lane.
  let values = cast[ptr UncheckedArray[cint]](output)
  values[int(threadIdx.x)] = seed + cint(threadIdx.x)

proc sharedKernel(output: ptr cint; seed: cint) {.hippoGlobal.} =
  ## Exchange neighboring lane values across one block barrier.
  var shared {.hippoShared.}: array[Lanes, cint]
  let
    lane = int(threadIdx.x)
    values = cast[ptr UncheckedArray[cint]](output)
  shared[lane] = seed + cint(lane)
  hippoSyncthreads()
  values[lane] = shared[(lane + 1) mod Lanes]

proc benchmark(workers: uint; shared: bool) =
  ## Report first-launch latency and warmed median/p95, checking final results.
  setThreads(workers)
  let output = hippoMalloc(Lanes * sizeof(cint))
  var
    seed: cint
    actual: array[Lanes, cint]
    elapsed: array[Samples, int64]
  template launch() =
    if shared:
      hippoLaunchKernel(sharedKernel, blockDim = newDim3(Lanes), args = hippoArgs(output.p, seed))
    else:
      hippoLaunchKernel(smallKernel, blockDim = newDim3(Lanes), args = hippoArgs(output.p, seed))
  let start = getMonoTime()
  launch()
  let cold = (getMonoTime() - start).inNanoseconds
  for index in 0 ..< Warmup:
    launch()
  for index in 0 ..< Samples:
    seed = cint(index)
    let start = getMonoTime()
    launch()
    elapsed[index] = (getMonoTime() - start).inNanoseconds
  hippoMemcpy(addr actual[0], output, sizeof(actual), HippoMemcpyDeviceToHost)
  for lane in 0 ..< Lanes:
    let expectedLane = if shared: (lane + 1) mod Lanes else: lane
    doAssert actual[lane] == seed + cint(expectedLane)
  elapsed.sort()
  let name = if shared: "shared" else: "plain"
  echo &"{name} workers={workers} cold_us={float64(cold) / 1000:.2f} median_us={float64(elapsed[Samples div 2]) / 1000:.2f} p95_us={float64(elapsed[Samples * 95 div 100]) / 1000:.2f}"

proc main() =
  ## Run identical launch workloads at several CPU worker counts.
  when HippoRuntime != "SIMPLE":
    {.error: "Use nimble bench_simple for this SIMPLE scheduler benchmark.".}
  else:
    for shared in [false, true]:
      for workers in [1'u, 2'u, 4'u, 8'u]:
        benchmark(workers, shared)

when isMainModule:
  main()
