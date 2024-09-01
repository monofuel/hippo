# vector_sums_hippo.nims is setup for hipcc to build for amd
# but it should work with either --cc:hipcc or --cc:nvcc
# hippo functions will translate to cuda or hip automatically depending on which is compiled for

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
  var a,b,c: array[N, int32] # host-side arrays

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
    addkernel,
    gridDim = newDim3(N.uint32),
    args = (dev_a, dev_b, dev_c)
  )

  # copy result back to host
  hippoMemcpy(addr c[0], dev_c, sizeof(int32)*N, hipMemcpyDeviceToHost)

  # display the results
  for i in 0..<N:
    echo a[i], " + ", b[i], " = ", c[i]

  # Hippo automatically frees gpu memory when it goes out of scope

when isMainModule:
  main()