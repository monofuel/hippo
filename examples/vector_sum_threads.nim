# vector_sums_threads.nims is setup to use the SIMPLE pure nim backend with --threads:on
# does not need hipcc or nvcc, or require compiling with cpp
# --define:"HippoRuntime:SIMPLE"
# only hippo functions are available, no hip or cuda functions. will try to emulate gpu behavior.

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
  hippoMemcpy(dev_a, addr a[0], sizeof(int32)*N, HippoMemcpyHostToDevice)
  hippoMemcpy(dev_b, addr b[0], sizeof(int32)*N, HippoMemcpyHostToDevice)

  # launch kernel
  hippoLaunchKernel(
    addkernel,
    gridDim = newDim3(N.uint32),
    args = hippoArgs(dev_a.p, dev_b.p, dev_c.p)
  )

  # copy result back to host
  hippoMemcpy(addr c[0], dev_c, sizeof(int32)*N, HippoMemcpyDeviceToHost)

  # display the results
  for i in 0..<N:
    echo a[i], " + ", b[i], " = ", c[i]

  # Hippo automatically frees gpu memory when it goes out of scope

when isMainModule:
  main()