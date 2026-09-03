import
  hippo,
  std/unittest,
  ./utils

const N = 3 * 1024 * 1024 * 1024 div 4

suite "allocations above 2 GiB":
  testSkipPlatforms "3 GiB malloc and memcpy round trip", "SIMPLE", "HIP_CPU":
    var host = newSeq[float32](N)
    host[0] = 1.5
    host[N - 1] = 2.5
    let dev = hippoMalloc(N * 4)
    hippoMemcpy(dev, addr host[0], N * 4, HippoMemcpyHostToDevice)
    var back = newSeq[float32](N)
    hippoMemcpy(addr back[0], dev, N * 4, HippoMemcpyDeviceToHost)
    check back[0] == 1.5
    check back[N - 1] == 2.5
    check back[N div 2] == 0.0
