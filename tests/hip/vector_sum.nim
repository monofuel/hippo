import hippo

const N: int32 = 10

proc addKernel(a, b, c: ptr[cint]){.hippoGlobal.} =
  let tid = blockIdx.x  # handle data at this index as an integer
  if tid < N.uint:  # guard for out of bounds
    let aArray = cast[ptr UncheckedArray[cint]](a)
    let bArray = cast[ptr UncheckedArray[cint]](b)
    let cArray = cast[ptr UncheckedArray[cint]](c)
    cArray[tid] = aArray[tid] + bArray[tid]


proc main() =
  var a,b,c: array[N, int32]
  var dev_a, dev_b, dev_c: ptr[int32]

  # allocate gpu memory
  handleError(hipMalloc(cast[ptr pointer](addr dev_a), sizeof(int32)*N))
  handleError(hipMalloc(cast[ptr pointer](addr dev_b), sizeof(int32)*N))
  handleError(hipMalloc(cast[ptr pointer](addr dev_c), sizeof(int32)*N))

  # fill in arrays a and b on the host
  for i in 0..<N:
    a[i] = -i
    b[i] = i * i

  # copy data to device
  handleError(hipMemcpy(dev_a, cast[ptr pointer](addr a), sizeof(int32)*N, hipMemcpyHostToDevice))
  handleError(hipMemcpy(dev_b, cast[ptr pointer](addr b), sizeof(int32)*N, hipMemcpyHostToDevice))

  # launch kernel
  handleError(launchKernel(
    addkernel,
    blockDim = newDim3(N.uint32,1,1),
    gridDim = newDim3(1,1,1),
    args = (dev_a, dev_b, dev_c)
  ))

  # copy result back to host
  handleError(hipMemcpy(cast[ptr pointer](addr c), dev_c, sizeof(int32)*N, hipMemcpyDeviceToHost))

  # display the results
  for i in 0..<N:
    echo a[i], " + ", b[i], " = ", c[i]

  # free gpu memory
  handleError(hipFree(dev_a))
  handleError(hipFree(dev_b))
  handleError(hipFree(dev_c))




when isMainModule:
  main()