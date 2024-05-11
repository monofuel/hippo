import std/[unittest], hippo

const N: int32 = 10

proc addKernel(a, b, c: ptr[cint]){.hippoGlobal.} =
  let tid = blockIdx.x  # handle data at this index as an integer
  if tid < N.uint:  # guard for out of bounds
    let aArray = cast[ptr UncheckedArray[cint]](a)
    let bArray = cast[ptr UncheckedArray[cint]](b)
    let cArray = cast[ptr UncheckedArray[cint]](c)
    cArray[tid] = aArray[tid] + bArray[tid]

suite "vector_sum":
    test "run":
      var a,b,c: array[N, int32]
      var dev_a, dev_b, dev_c: pointer

      # allocate gpu memory
      handleError(hippoMalloc(dev_a, sizeof(int32)*N))
      handleError(hippoMalloc(dev_b, sizeof(int32)*N))
      handleError(hippoMalloc(dev_c, sizeof(int32)*N))

      # fill in arrays a and b on the host
      for i in 0..<N:
        a[i] = -i
        b[i] = i * i

      # copy data to device
      handleError(hippoMemcpy(dev_a, addr a[0], sizeof(int32)*N, hipMemcpyHostToDevice))
      handleError(hippoMemcpy(dev_b, addr b[0], sizeof(int32)*N, hipMemcpyHostToDevice))

      # launch kernel
      handleError(launchKernel(
        addkernel,
        gridDim = newDim3(N.uint32),
        args = (dev_a, dev_b, dev_c)
      ))

      # copy result back to host
      handleError(hippoMemcpy(addr c[0], dev_c, sizeof(int32)*N, hipMemcpyDeviceToHost))

      # display the results
      for i in 0..<N:
        echo a[i], " + ", b[i], " = ", c[i]

      # free gpu memory
      handleError(hippoFree(dev_a))
      handleError(hippoFree(dev_b))
      handleError(hippoFree(dev_c))
