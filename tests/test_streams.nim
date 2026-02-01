# test cuda / hip streams

import
  std/[unittest, random], hippo, ./utils

const
  N = 1024 * 1024
  FULL_DATA_SIZE = N * 20
  BlockSize = 256

proc addKernel(a, b, c: ptr[cint]) {.hippoGlobal.} =
  ## Simple kernel to average 3 values in a[] and b[].
  let tid = blockIdx.x * blockDim.x + threadIdx.x
  if tid < N.uint:
    let aArray = cast[ptr UncheckedArray[cint]](a)
    let bArray = cast[ptr UncheckedArray[cint]](b)
    let cArray = cast[ptr UncheckedArray[cint]](c)
    let idx1 = (tid + 1) mod 256
    let idx2 = (tid + 2) mod 256
    let as_val = (aArray[tid].float + aArray[idx1].float + aArray[idx2].float) / 3.0f
    let bs_val = (bArray[tid].float + bArray[idx1].float + bArray[idx2].float) / 3.0f
    cArray[tid] = ((as_val + bs_val) / 2).cint

suite "Streams":
  testSkipPlatforms "single stream test", "SIMPLE":

    var props: HippoDeviceProp
    let device = hippoGetDevice()
    hippoGetDeviceProperties(props, device)
    if props.deviceOverlap == 0:
      raise newException(Exception, "Device does not support overlap")

    # Create stream
    let stream = hippoStreamCreate()

    let startEvent = hippoEventCreate()
    let stopEvent = hippoEventCreate()

    # Allocate device memory
    var dev_a = hippoMalloc(sizeof(cint) * N)
    var dev_b = hippoMalloc(sizeof(cint) * N)
    var dev_c = hippoMalloc(sizeof(cint) * N)

    # Allocate page-locked host memory for async transfers
    let host_a_raw = hippoHostAlloc(sizeof(cint) * FULL_DATA_SIZE)
    let host_b_raw = hippoHostAlloc(sizeof(cint) * FULL_DATA_SIZE)
    let host_c_raw = hippoHostAlloc(sizeof(cint) * FULL_DATA_SIZE)
    var host_a = cast[ptr UncheckedArray[cint]](host_a_raw)
    var host_b = cast[ptr UncheckedArray[cint]](host_b_raw)
    var host_c = cast[ptr UncheckedArray[cint]](host_c_raw)

    # Initialize host data
    for i in 0..<FULL_DATA_SIZE:
      host_a[i] = rand(1000).cint
      host_b[i] = rand(1000).cint

    hippoEventRecord(startEvent, stream)

    # Process data in chunks
    for i in countup(0, FULL_DATA_SIZE - N, N):
      # Copy input data to device asynchronously
      hippoMemcpyAsync(dev_a.p, addr host_a[i], sizeof(cint) * N, HippoMemcpyHostToDevice, stream)
      hippoMemcpyAsync(dev_b.p, addr host_b[i], sizeof(cint) * N, HippoMemcpyHostToDevice, stream)

      # Launch kernel on stream
      hippoLaunchKernel(
        addKernel,
        gridDim = newDim3((N div BlockSize).uint32),
        blockDim = newDim3(BlockSize.uint32),
        stream = stream,
        args = hippoArgs(dev_a.p, dev_b.p, dev_c.p)
      )

      # Copy results back to host asynchronously
      hippoMemcpyAsync(addr host_c[i], dev_c.p, sizeof(cint) * N, HippoMemcpyDeviceToHost, stream)

    hippoEventRecord(stopEvent, stream)

    # Synchronize the stream
    hippoStreamSynchronize(stream)
    hippoEventSynchronize(stopEvent)

    let elapsedTime = hippoEventElapsedTime(startEvent, stopEvent)
    check elapsedTime >= 0.0

    # Verify results (check a few samples)
    var correct = 0
    for i in 0..<min(10, FULL_DATA_SIZE):
      let idx1 = (i + 1) mod 256
      let idx2 = (i + 2) mod 256
      let as_val = (host_a[i].float + host_a[idx1].float + host_a[idx2].float) / 3.0f
      let bs_val = (host_b[i].float + host_b[idx1].float + host_b[idx2].float) / 3.0f
      let expected = ((as_val + bs_val) / 2.0f).cint
      if host_c[i] == expected:
        correct += 1

    check correct == min(10, FULL_DATA_SIZE)

    # TODO we should implement nim ref helpers to automatically clean stuff like this up
    # Cleanup
    hippoHostFree(host_a_raw)
    hippoHostFree(host_b_raw)
    hippoHostFree(host_c_raw)
    hippoEventDestroy(startEvent)
    hippoEventDestroy(stopEvent)
    hippoStreamDestroy(stream)

  testSkipPlatforms "multi stream test", "SIMPLE":

    var props: HippoDeviceProp
    let device = hippoGetDevice()
    hippoGetDeviceProperties(props, device)
    if props.deviceOverlap == 0:
      raise newException(Exception, "Device does not support overlap")

    let stream0 = hippoStreamCreate()
    let stream1 = hippoStreamCreate()

    let startEvent = hippoEventCreate()
    let stopEvent = hippoEventCreate()

    var dev_a0 = hippoMalloc(sizeof(cint) * N)
    var dev_b0 = hippoMalloc(sizeof(cint) * N)
    var dev_c0 = hippoMalloc(sizeof(cint) * N)
    var dev_a1 = hippoMalloc(sizeof(cint) * N)
    var dev_b1 = hippoMalloc(sizeof(cint) * N)
    var dev_c1 = hippoMalloc(sizeof(cint) * N)

    let host_a_raw = hippoHostAlloc(sizeof(cint) * FULL_DATA_SIZE)
    let host_b_raw = hippoHostAlloc(sizeof(cint) * FULL_DATA_SIZE)
    let host_c_raw = hippoHostAlloc(sizeof(cint) * FULL_DATA_SIZE)
    var host_a = cast[ptr UncheckedArray[cint]](host_a_raw)
    var host_b = cast[ptr UncheckedArray[cint]](host_b_raw)
    var host_c = cast[ptr UncheckedArray[cint]](host_c_raw)

    for i in 0..<FULL_DATA_SIZE:
      host_a[i] = rand(1000).cint
      host_b[i] = rand(1000).cint

    hippoEventRecord(startEvent)

    for i in countup(0, FULL_DATA_SIZE - (N * 2), N * 2):
      hippoMemcpyAsync(dev_a0.p, addr host_a[i], sizeof(cint) * N, HippoMemcpyHostToDevice, stream0)
      hippoMemcpyAsync(dev_a1.p, addr host_a[i + N], sizeof(cint) * N, HippoMemcpyHostToDevice, stream1)
      hippoMemcpyAsync(dev_b0.p, addr host_b[i], sizeof(cint) * N, HippoMemcpyHostToDevice, stream0)
      hippoMemcpyAsync(dev_b1.p, addr host_b[i + N], sizeof(cint) * N, HippoMemcpyHostToDevice, stream1)

      hippoLaunchKernel(
        addKernel,
        gridDim = newDim3((N div BlockSize).uint32),
        blockDim = newDim3(BlockSize.uint32),
        stream = stream0,
        args = hippoArgs(dev_a0.p, dev_b0.p, dev_c0.p)
      )
      hippoLaunchKernel(
        addKernel,
        gridDim = newDim3((N div BlockSize).uint32),
        blockDim = newDim3(BlockSize.uint32),
        stream = stream1,
        args = hippoArgs(dev_a1.p, dev_b1.p, dev_c1.p)
      )

      hippoMemcpyAsync(addr host_c[i], dev_c0.p, sizeof(cint) * N, HippoMemcpyDeviceToHost, stream0)
      hippoMemcpyAsync(addr host_c[i + N], dev_c1.p, sizeof(cint) * N, HippoMemcpyDeviceToHost, stream1)

    hippoEventRecord(stopEvent)

    hippoStreamSynchronize(stream0)
    hippoStreamSynchronize(stream1)
    hippoEventSynchronize(stopEvent)

    let elapsedTime = hippoEventElapsedTime(startEvent, stopEvent)
    check elapsedTime >= 0.0

    var correct = 0
    for i in 0..<min(10, FULL_DATA_SIZE):
      let idx1 = (i + 1) mod 256
      let idx2 = (i + 2) mod 256
      let as_val = (host_a[i].float + host_a[idx1].float + host_a[idx2].float) / 3.0f
      let bs_val = (host_b[i].float + host_b[idx1].float + host_b[idx2].float) / 3.0f
      let expected = ((as_val + bs_val) / 2.0f).cint
      if host_c[i] == expected:
        correct += 1

    check correct == min(10, FULL_DATA_SIZE)

    hippoHostFree(host_a_raw)
    hippoHostFree(host_b_raw)
    hippoHostFree(host_c_raw)
    hippoEventDestroy(startEvent)
    hippoEventDestroy(stopEvent)
    hippoStreamDestroy(stream0)
    hippoStreamDestroy(stream1)
