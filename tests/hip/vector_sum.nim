import hippo

const N = 10



proc addKernel(a,b,c: ptr[cint]){.hippoGlobal.} =
  var blockIdx {.importcpp: "blockIdx".}: BlockIdx
  let tid = blockIdx.x # handle data at this index
  if tid < N: # guard for out of bounds
    c[] = a[] + b[]

proc main() =
  var a,b,c: array[N, int]
  var dev_a, dev_b, dev_c: ptr[int]

  # allocate gpu memory
  handleError(hipMalloc(cast[ptr pointer](addr dev_a), sizeof(int).cint))
  handleError(hipMalloc(cast[ptr pointer](addr dev_b), sizeof(int).cint))
  handleError(hipMalloc(cast[ptr pointer](addr dev_c), sizeof(int).cint))

  # fill in arrays a and b on the host
  for i in 0..<N:
    a[i] = -i
    b[i] = i * i

  # copy data to device
  handleError(hipMemcpy(dev_a, cast[ptr pointer](addr a), sizeof(int)*N, hipMemcpyHostToDevice))
  handleError(hipMemcpy(dev_b, cast[ptr pointer](addr b), sizeof(int)*N, hipMemcpyHostToDevice))

  # launch kernel
  handleError(launchKernel(
    addkernel,
    blockDim = newDim3(N),
    args = (dev_a, dev_b, dev_c)
  ))

  # copy result back to host
  handleError(hipMemcpy(cast[ptr pointer](addr c), dev_c, sizeof(int)*N, hipMemcpyDeviceToHost))

  # display the results
  for i in 0..<N:
    echo a[i], " + ", b[i], " = ", c[i]

  # free gpu memory
  handleError(hipFree(dev_a))
  handleError(hipFree(dev_b))
  handleError(hipFree(dev_c))




when isMainModule:
  main()