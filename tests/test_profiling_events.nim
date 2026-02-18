import
  std/unittest,
  hippo,
  ./utils

const
  KernelThreads = 256

proc writeOneKernel(output: ptr[int32]) {.hippoGlobal.} =
  ## Write a known value from one GPU thread.
  let tid = threadIdx.x + blockIdx.x * blockDim.x
  if tid == 0:
    let outputArray = cast[ptr UncheckedArray[int32]](output)
    outputArray[0] = 1

proc addOneKernel(input: ptr[int32], output: ptr[int32]) {.hippoGlobal.} =
  ## Read and transform a value from one GPU thread.
  let tid = threadIdx.x + blockIdx.x * blockDim.x
  if tid == 0:
    let inputArray = cast[ptr UncheckedArray[int32]](input)
    let outputArray = cast[ptr UncheckedArray[int32]](output)
    outputArray[0] = inputArray[0] + 1

suite "Profiling events":
  testSkipPlatforms "event timing and query", "SIMPLE":
    let stream = hippoStreamCreate()
    defer:
      hippoStreamDestroy(stream)

    let startEvent = hippoEventCreateWithFlags(HippoEventDefault)
    defer:
      hippoEventDestroy(startEvent)

    let stopEvent = hippoEventCreate()
    defer:
      hippoEventDestroy(stopEvent)

    let markerEvent = hippoEventCreateWithFlags(HippoEventDisableTiming)
    defer:
      hippoEventDestroy(markerEvent)

    var devValue = hippoMalloc(sizeof(int32))
    var hostValue = int32(0)

    hippoEventRecord(startEvent, stream)
    hippoLaunchKernel(
      writeOneKernel,
      gridDim = newDim3(1'u32),
      blockDim = newDim3(KernelThreads.uint32),
      stream = stream,
      args = hippoArgs(devValue.p)
    )
    hippoEventRecord(markerEvent, stream)
    hippoEventRecord(stopEvent, stream)

    discard hippoEventQuery(markerEvent)

    hippoEventSynchronize(stopEvent)
    check hippoEventQuery(stopEvent)

    let elapsedMs = hippoEventElapsedTime(startEvent, stopEvent)
    check elapsedMs >= 0.0

    hippoMemcpy(addr hostValue, devValue, sizeof(int32), HippoMemcpyDeviceToHost)
    check hostValue == 1

  testSkipPlatforms "stream wait event ordering", "SIMPLE":
    let producerStream = hippoStreamCreate()
    defer:
      hippoStreamDestroy(producerStream)

    let consumerStream = hippoStreamCreate()
    defer:
      hippoStreamDestroy(consumerStream)

    let readyEvent = hippoEventCreate()
    defer:
      hippoEventDestroy(readyEvent)

    var devInput = hippoMalloc(sizeof(int32))
    var devOutput = hippoMalloc(sizeof(int32))
    var hostOutput = int32(0)

    hippoLaunchKernel(
      writeOneKernel,
      gridDim = newDim3(1'u32),
      blockDim = newDim3(1'u32),
      stream = producerStream,
      args = hippoArgs(devInput.p)
    )
    hippoEventRecord(readyEvent, producerStream)

    hippoStreamWaitEvent(consumerStream, readyEvent)
    hippoLaunchKernel(
      addOneKernel,
      gridDim = newDim3(1'u32),
      blockDim = newDim3(1'u32),
      stream = consumerStream,
      args = hippoArgs(devInput.p, devOutput.p)
    )
    hippoStreamSynchronize(consumerStream)

    hippoMemcpy(addr hostOutput, devOutput, sizeof(int32), HippoMemcpyDeviceToHost)
    check hostOutput == 2
