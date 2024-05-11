# vector_sum_cuda.nims is setup with nvcc to build for GPU
# requires https://github.com/monofuel/Nim/tree/cudacc-nvcc at the moment
# nim cpp -r examples/vector_sum_cuda.nim

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
  var dev_a, dev_b, dev_c: pointer

  # allocate gpu memory
  handleError(cudaMalloc(addr dev_a, sizeof(int32)*N))
  handleError(cudaMalloc(addr dev_b, sizeof(int32)*N))
  handleError(cudaMalloc(addr dev_c, sizeof(int32)*N))

  # fill in arrays a and b on the host
  for i in 0..<N:
    a[i] = -i
    b[i] = i * i

  # copy data to device
  handleError(cudaMemcpy(dev_a, addr a[0], sizeof(int32)*N, cudaMemcpyHostToDevice))
  handleError(cudaMemcpy(dev_b, addr b[0], sizeof(int32)*N, cudaMemcpyHostToDevice))

  # launch kernel
  handleError(launchKernel(
    addkernel,
    gridDim = newDim3(N.uint32),
    args = (dev_a, dev_b, dev_c)
  ))

  # copy result back to host
  handleError(cudaMemcpy(addr c[0], dev_c, sizeof(int32)*N, cudaMemcpyDeviceToHost))

  # display the results
  for i in 0..<N:
    echo a[i], " + ", b[i], " = ", c[i]

  # free gpu memory
  handleError(cudaFree(dev_a))
  handleError(cudaFree(dev_b))
  handleError(cudaFree(dev_c))

when isMainModule:
  main()