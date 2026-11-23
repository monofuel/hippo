## Manual Chapter 11.3 and 11.4 checks; deliberately outside the test_*.nim glob.
## Requires native HIP or CUDA, host threads and two real GPUs. Missing hardware is an
## unmet prerequisite, not a successful emulation of multi-GPU behavior.
## Run from the repository root. Build for both GPUs' actual architectures.
## On high-steel, unset the existing architecture override for this process:
##
## .. code-block:: sh
##
##   env -u HSA_OVERRIDE_GFX_VERSION nim cpp --cc:hipcc -d:useMalloc -d:HippoRuntime:HIP --threads:on --passC:--offload-arch=gfx1100 --passC:--offload-arch=gfx1036 -r tests/manual_chapter11_multigpu.nim
##   nim cpp --cc:nvcc -d:useMalloc -d:HippoRuntime:CUDA -r tests/manual_chapter11_multigpu.nim
##   HIP_PLATFORM=nvidia nim cpp --cc:nvcc --nvcc.cpp.exe:hipcc --nvcc.cpp.linkerexe:hipcc -d:useMalloc -d:HippoRuntime:HIP -r tests/manual_chapter11_multigpu.nim
##
## The first two visible devices are used. Standard ROCR_VISIBLE_DEVICES or
## CUDA_VISIBLE_DEVICES can select a pair. No peer-to-peer support or speedup is required.
## Chapter 11.4 shares portable, mapped host inputs; each GPU gets its own alias.
## HIP's write-combined flag is accepted for CUDA compatibility but does not
## promise physical write-combined memory or a performance gain on AMD.

import
  std/unittest,
  hippo,
  ./chapter11_dot

when HippoRuntime notin ["HIP", "CUDA"]:
  {.error: "Chapter 11.3/11.4 require a native HIP or CUDA GPU backend.".}
when not compileOption("threads"):
  {.error: "Chapter 11.3/11.4 require --threads:on.".}

type
  MemoryMode = enum
    CopiedInputs, PortableInputs
  Worker = object
    device, actualDevice: cint
    a, b: pointer
    offset, count: int
    mode: MemoryMode
    total: float64
    error: array[1024, char]

const
  Sizes = [0, 1, 257, 4099, 65553]
  DeviceOrders = [[0'i32, 1'i32], [1'i32, 0'i32]]
  PortableFlags = [HippoHostAllocMapped or HippoHostAllocPortable,
    HippoHostAllocMapped or HippoHostAllocPortable or HippoHostAllocWriteCombined]
  Tolerance = 1e-6

proc runSelected(worker: ptr Worker) =
  ## Create and destroy all device resources while this worker's device is current.
  let stream = hippoStreamCreate()
  defer: hippoStreamDestroy(stream)
  worker.actualDevice = hippoGetDevice()
  case worker.mode
  of CopiedInputs:
    worker.total = copiedDot(floatOffset(worker.a, worker.offset),
      floatOffset(worker.b, worker.offset), worker.count, stream)
  of PortableInputs:
    worker.total = mappedDot(worker.a, worker.b, worker.offset, worker.count, stream)

proc runWorker(worker: ptr Worker) {.thread.} =
  ## Select a device per host thread and propagate failures to the test's thread.
  try:
    let original = hippoGetDevice()
    hippoSetDevice(worker.device)
    try:
      runSelected(worker)
    finally:
      hippoSetDevice(original)
  except Exception as error:
    for index in 0 ..< min(error.msg.len, worker.error.high):
      worker.error[index] = error.msg[index]

proc checkPair(a, b: pointer; count, firstCount: int;
               devices: array[2, cint]; mode: MemoryMode) =
  ## Check both per-device slices and their sum against independent host results.
  let original = hippoGetDevice()
  var
    workers = [
      Worker(device: devices[0], actualDevice: -1, a: a, b: b,
        offset: 0, count: firstCount, mode: mode),
      Worker(device: devices[1], actualDevice: -1, a: a, b: b,
        offset: firstCount, count: count - firstCount, mode: mode)
    ]
    threads: array[2, Thread[ptr Worker]]
    started = 0
  try:
    for index in 0 ..< workers.len:
      createThread(threads[index], runWorker, addr workers[index])
      inc started
  finally:
    for index in 0 ..< started:
      joinThread(threads[index])
  for index in 0 ..< workers.len:
    if workers[index].error[0] != '\0':
      raise newException(ValueError, "GPU " & $workers[index].device & ": " &
        $cast[cstring](addr workers[index].error[0]))
    doAssert workers[index].actualDevice == devices[index], "Worker selected the wrong GPU."
    let expected = referenceDot(floatOffset(a, workers[index].offset),
      floatOffset(b, workers[index].offset), workers[index].count)
    doAssert abs(workers[index].total - expected) <= Tolerance,
      "Wrong partial result on GPU " & $devices[index] & " offset=" & $workers[index].offset &
      " count=" & $workers[index].count
  doAssert workers[0].actualDevice != workers[1].actualDevice, "Two distinct GPUs are required."
  doAssert abs(workers[0].total + workers[1].total - referenceDot(a, b, count)) <= Tolerance,
    "The combined two-GPU result is incorrect."
  doAssert hippoGetDevice() == original, "Worker selection changed the caller's current device."

proc deviceLabel(device: cint): string =
  ## Include actual device names in the manual test report.
  var properties: HippoDeviceProp
  hippoGetDeviceProperties(properties, device)
  let kind = if properties.integrated != 0: "integrated GPU" else: "discrete GPU"
  $device & ": " & $cast[cstring](addr properties.name[0]) &
    " (" & kind & ")"

if hippoGetDeviceCount() < 2:
  raise newException(ValueError, "Chapter 11.3/11.4 need at least two visible GPUs.")

suite "Chapter 11 multi-GPU [" & deviceLabel(0) & " + " & deviceLabel(1) & "]":
  test "11.3: per-GPU copies with equal, uneven and empty partitions":
    for count in Sizes:
      var
        a = newSeq[cfloat](max(count, 1))
        b = newSeq[cfloat](max(count, 1))
      for seed in [0, 7]:
        fillInputs(addr a[0], addr b[0], count, seed)
        for devices in DeviceOrders:
          for firstCount in [count div 2, max(count - 1, 0)]:
            checkPair(addr a[0], addr b[0], count, firstCount, devices, CopiedInputs)

  test "11.4: portable mapped host inputs shared across both GPUs and threads":
    let original = hippoGetDevice()
    defer: hippoSetDevice(original)
    for device in [0'i32, 1'i32]:
      var properties: HippoDeviceProp
      hippoGetDeviceProperties(properties, device)
      doAssert properties.canMapHostMemory != 0,
        "GPU " & $device & " cannot map host memory for Chapter 11.4."
    hippoSetDevice(0)
    for count in Sizes:
      for flags in PortableFlags:
        let a = hippoHostAlloc(max(count, 1) * sizeof(cfloat), flags)
        defer: hippoHostFree(a)
        let b = hippoHostAlloc(max(count, 1) * sizeof(cfloat), flags)
        defer: hippoHostFree(b)
        for seed in [1, 11]:
          fillInputs(a, b, count, seed)
          for devices in DeviceOrders:
            for firstCount in [count div 2, max(count - 1, 0)]:
              checkPair(a, b, count, firstCount, devices, PortableInputs)
