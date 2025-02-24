# vector_sums_hippo_host_device.nim showcases how to use
# __device__ and __host__ functions in the same program.

import hippo

const N: int32 = 10

proc addFunc(a, b: int32): int32 {.hippoHostDevice.} =
  ## Host or Device usable function
  return a + b

proc addKernel(a, b, c: ptr[cint]){.hippoGlobal.} =
  let tid = blockIdx.x  # handle data at this index as an integer
  if tid < N.uint:  # guard for out of bounds
    let aArray = cast[ptr UncheckedArray[cint]](a)
    let bArray = cast[ptr UncheckedArray[cint]](b)
    let cArray = cast[ptr UncheckedArray[cint]](c)
    cArray[tid] = addFunc(aArray[tid], bArray[tid])

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
    args = (dev_a.p, dev_b.p, dev_c.p)
  )

  # copy result back to host
  hippoMemcpy(addr c[0], dev_c, sizeof(int32)*N, hipMemcpyDeviceToHost)

  # verify results using CPU
  for i in 0..<N:
    let cpuResult = addFunc(a[i], b[i])
    assert c[i] == cpuResult, "GPU and CPU results don't match at index " & $i & 
      ". GPU: " & $c[i] & ", CPU: " & $cpuResult

  # display the results
  for i in 0..<N:
    echo a[i], " + ", b[i], " = ", c[i]

  # Hippo automatically frees gpu memory when it goes out of scope

when isMainModule:
  main()