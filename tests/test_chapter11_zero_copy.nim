## Chapter 11.2: single-device copied versus mapped host-memory dot products.
## Runs in the existing CI test tasks; no two-GPU setup or timing assertions.
## HIP-CPU exercises the pointer/dataflow contract using ordinary CPU memory.
## Real multi-GPU coverage for 11.3/11.4 lives in manual_chapter11_multigpu.nim.

import
  std/[unittest, math],
  hippo

when HippoRuntime == "HIP" or HippoRuntime == "HIP_CPU":
  import ./chapter11_dot

  const
    Sizes = [0, 1, 33, 257, 4099, 32785]
    Allocations = [HippoHostAllocMapped,
      HippoHostAllocMapped or HippoHostAllocPortable,
      HippoHostAllocMapped or HippoHostAllocWriteCombined,
      HippoHostAllocMapped or HippoHostAllocPortable or HippoHostAllocWriteCombined]
    Tolerance = 1e-6

  proc checkAllocation(count: int; flags: uint32) =
    ## Reuse a mapped allocation after host writes and compare with copied inputs.
    let
      capacity = max(count, 1) * sizeof(cfloat)
      a = hippoHostAlloc(capacity, flags)
    defer: hippoHostFree(a)
    let b = hippoHostAlloc(capacity, flags)
    defer: hippoHostFree(b)
    let stream = hippoStreamCreate()
    defer: hippoStreamDestroy(stream)
    for seed in [0, 7]:
      fillInputs(a, b, count, seed)
      let
        expected = referenceDot(a, b, count)
        copied = copiedDot(a, b, count, stream)
        mapped = mappedDot(a, b, 0, count, stream)
      doAssert abs(copied - expected) <= Tolerance,
        "Copied dot mismatch: count=" & $count & " flags=" & $flags
      doAssert abs(mapped - expected) <= Tolerance,
        "Mapped dot mismatch: count=" & $count & " flags=" & $flags

  proc checkMappedOutput() =
    ## Poison mapped block sums, then verify device writes are visible on the host.
    const Count = 1031
    let a = hippoHostAlloc(Count * sizeof(cfloat), HippoHostAllocMapped)
    defer: hippoHostFree(a)
    let b = hippoHostAlloc(Count * sizeof(cfloat), HippoHostAllocMapped)
    defer: hippoHostFree(b)
    let output = hippoHostAlloc(MaxBlocks * sizeof(cfloat), HippoHostAllocMapped)
    defer: hippoHostFree(output)
    let stream = hippoStreamCreate()
    defer: hippoStreamDestroy(stream)
    let
      deviceA = hippoHostGetDevicePointer(a)
      deviceB = hippoHostGetDevicePointer(b)
      deviceOutput = hippoHostGetDevicePointer(output)
      hostOutput = cast[ptr UncheckedArray[cfloat]](output)
    for seed in [1, 11]:
      fillInputs(a, b, Count, seed)
      for index in 0 ..< MaxBlocks:
        hostOutput[index] = cfloat(NaN)
      launchDot(deviceA, deviceB, deviceOutput, Count, stream)
      var actual = 0.0
      for index in 0 ..< MaxBlocks:
        actual += float64(hostOutput[index])
      doAssert abs(actual - referenceDot(a, b, Count)) <= Tolerance

  proc checkSlices() =
    ## Check the offset/combine path on one device without claiming multi-GPU coverage.
    const
      Count = 521
      Split = 73
      Slices = [(0, 0), (0, 1), (17, 257), (Count - 1, 1), (Count, 0)]
    let a = hippoHostAlloc(Count * sizeof(cfloat), HippoHostAllocMapped)
    defer: hippoHostFree(a)
    let b = hippoHostAlloc(Count * sizeof(cfloat), HippoHostAllocMapped)
    defer: hippoHostFree(b)
    let stream = hippoStreamCreate()
    defer: hippoStreamDestroy(stream)
    fillInputs(a, b, Count, 7)
    for slice in Slices:
      let expected = referenceDot(floatOffset(a, slice[0]), floatOffset(b, slice[0]), slice[1])
      doAssert abs(mappedDot(a, b, slice[0], slice[1], stream) - expected) <= Tolerance
    let combined = mappedDot(a, b, 0, Split, stream) + mappedDot(a, b, Split, Count - Split, stream)
    doAssert abs(combined - referenceDot(a, b, Count)) <= Tolerance

  suite "Chapter 11.2: zero-copy host memory on one device":
    test "device enumeration and selecting the current device work":
      let
        count = hippoGetDeviceCount()
        original = hippoGetDevice()
      check count >= 1
      check original >= 0 and original < count
      hippoSetDevice(original)
      check hippoGetDevice() == original
      when HippoRuntime == "HIP_CPU":
        check count == 1

    test "copied and mapped dots match across sizes, flags and host updates":
      var properties: HippoDeviceProp
      hippoGetDeviceProperties(properties, hippoGetDevice())
      if properties.canMapHostMemory == 0:
        skip()
      else:
        for count in Sizes:
          for flags in Allocations:
            checkAllocation(count, flags)

    test "kernel writes to mapped output become visible after stream synchronization":
      var properties: HippoDeviceProp
      hippoGetDeviceProperties(properties, hippoGetDevice())
      if properties.canMapHostMemory == 0:
        skip()
      else:
        checkMappedOutput()

    test "offsets and combining partial results work on a single device":
      var properties: HippoDeviceProp
      hippoGetDeviceProperties(properties, hippoGetDevice())
      if properties.canMapHostMemory == 0:
        skip()
      else:
        checkSlices()
else:
  suite "Chapter 11.2: zero-copy host memory":
    test "HIP and HIP-CPU only; other backends are deferred":
      skip()
