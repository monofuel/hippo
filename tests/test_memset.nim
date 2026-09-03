import
  std/unittest,
  hippo

const
  BufferBytes = 64 * 1024 * 1024
  FillByte = 0xAB'u8
  AsyncValue = 0x7F

suite "memset":
  test "memset zeroes a large device buffer":
    var hostBuffer = newSeq[uint8](BufferBytes)
    for i in 0 ..< BufferBytes:
      hostBuffer[i] = FillByte
    let devBuffer = hippoMalloc(BufferBytes)

    hippoMemcpy(devBuffer, addr hostBuffer[0], BufferBytes, HippoMemcpyHostToDevice)
    hippoMemset(devBuffer, 0, BufferBytes)
    hippoMemcpy(addr hostBuffer[0], devBuffer, BufferBytes, HippoMemcpyDeviceToHost)

    var nonZero = 0
    for i in 0 ..< BufferBytes:
      if hostBuffer[i] != 0'u8:
        nonZero.inc
    check nonZero == 0

  test "memset async fills a device buffer on a stream":
    var hostBuffer = newSeq[uint8](BufferBytes)
    let
      devBuffer = hippoMalloc(BufferBytes)
      stream = hippoStreamCreate()

    hippoMemcpy(devBuffer, addr hostBuffer[0], BufferBytes, HippoMemcpyHostToDevice)
    hippoMemsetAsync(devBuffer.p, AsyncValue.cint, BufferBytes, stream)
    hippoStreamSynchronize(stream)
    hippoMemcpy(addr hostBuffer[0], devBuffer, BufferBytes, HippoMemcpyDeviceToHost)
    hippoStreamDestroy(stream)

    var wrong = 0
    for i in 0 ..< BufferBytes:
      if hostBuffer[i] != uint8(AsyncValue):
        wrong.inc
    check wrong == 0
