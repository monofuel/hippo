import std/[unittest, strformat], hippo

const N: int32 = 10

proc addKernelBlock(a, b, c: ptr[cint]){.hippoGlobal.} =
  let tid = blockIdx.x  # handle data at this index as an integer
  if tid < N.uint:  # guard for out of bounds
    let aArray = cast[ptr UncheckedArray[cint]](a)
    let bArray = cast[ptr UncheckedArray[cint]](b)
    let cArray = cast[ptr UncheckedArray[cint]](c)
    cArray[tid] = aArray[tid] + bArray[tid]

proc addKernelThread(a, b, c: ptr[cint]){.hippoGlobal.} =
  let tid = threadIdx.x  # handle data at this index as an integer
  if tid < N.uint:  # guard for out of bounds
    let aArray = cast[ptr UncheckedArray[cint]](a)
    let bArray = cast[ptr UncheckedArray[cint]](b)
    let cArray = cast[ptr UncheckedArray[cint]](c)
    cArray[tid] = aArray[tid] + bArray[tid]

suite "vector_sum":
  test "blocks":
    var a,b,c: array[N, int32]

    # allocate gpu memory
    var dev_a = hippoMalloc(sizeof(int32)*N)
    var dev_b = hippoMalloc(sizeof(int32)*N)
    var dev_c = hippoMalloc(sizeof(int32)*N)

    # fill in arrays a and b on the host
    for i in 0..<N:
      a[i] = -i
      b[i] = i * i

    # copy data to device
    hippoMemcpy(dev_a, addr a[0], sizeof(int32)*N, hipMemcpyHostToDevice)
    hippoMemcpy(dev_b, addr b[0], sizeof(int32)*N, hipMemcpyHostToDevice)

    # launch kernel
    hippoLaunchKernel(
      addKernelBlock,
      gridDim = newDim3(N.uint32),
      args = (dev_a, dev_b, dev_c)
    )

    # copy result back to host
    hippoMemcpy(addr c[0], dev_c, sizeof(int32)*N, hipMemcpyDeviceToHost)

    # validate results on cpu
    for i in 0..<N:
      let expected = a[i] + b[i]
      assert(c[i] == expected, &"{a[i]} + {b[i]} = {c[i]} != {expected}")

    # free gpu memory
    hippoFree(dev_a)
    hippoFree(dev_b)
    hippoFree(dev_c)

  test "threads":
    var a,b,c: array[N, int32]

    # allocate gpu memory
    var dev_a = hippoMalloc(sizeof(int32)*N)
    var dev_b = hippoMalloc(sizeof(int32)*N)
    var dev_c = hippoMalloc(sizeof(int32)*N)

    # fill in arrays a and b on the host
    for i in 0..<N:
      a[i] = -i
      b[i] = i * i

    # copy data to device
    hippoMemcpy(dev_a, addr a[0], sizeof(int32)*N, hipMemcpyHostToDevice)
    hippoMemcpy(dev_b, addr b[0], sizeof(int32)*N, hipMemcpyHostToDevice)

    # launch kernel
    hippoLaunchKernel(
      addKernelThread,
      blockDim = newDim3(N.uint32),
      args = (dev_a, dev_b, dev_c)
    )

    # copy result back to host
    hippoMemcpy(addr c[0], dev_c, sizeof(int32)*N, hipMemcpyDeviceToHost)

    # validate results on cpu
    for i in 0..<N:
      let expected = a[i] + b[i]
      assert(c[i] == expected, &"{a[i]} + {b[i]} = {c[i]} != {expected}")

    # free gpu memory
    hippoFree(dev_a)
    hippoFree(dev_b)
    hippoFree(dev_c)
