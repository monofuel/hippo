# nim cpp --cc:nvcc --define:"useMalloc" hello_kernel.nim
{.emit: """
__global__ void add(int a, int b, int *c) {
    *c = a + b;
}
""".}

type
  cudaMemcpyKind* {.size: sizeof(cint), importcpp: "cudaMemcpyKind".} = enum
    cudaMemcpyHostToHost = 0,    ## < Host-to-Host Copy
    cudaMemcpyHostToDevice = 1,  ## < Host-to-Device Copy
    cudaMemcpyDeviceToHost = 2,  ## < Device-to-Host Copy
    cudaMemcpyDeviceToDevice = 3, ## < Device-to-Device Copy
    cudaMemcpyDefault = 4        ## < Runtime will automatically determine copy-kind based on virtual addresses.

proc cudaMalloc*(`ptr`: ptr pointer; size: cint): cint {.importcpp: "cudaMalloc(@)".}
proc cudaMemcpy*(dst: pointer; src: pointer; size: cint; kind: cudaMemcpyKind): cint {.importcpp: "cudaMemcpy(@)".}
proc cudaFree*(`ptr`: pointer): cint {.importcpp: "cudaFree(@)".}

proc main() =
  var c: int32
  var dev_c: ptr[int32]
  discard cudaMalloc(cast[ptr pointer](addr dev_c), sizeof(int32).cint)
  {.emit: """
  add<<<1,1>>>(2,7,dev_c);
  """.}
  discard cudaMemcpy(addr c, dev_c, sizeof(int32).cint, cudaMemcpyDeviceToHost)
  echo "2 + 7 = ", c
  discard cudaFree(dev_c)

when isMainModule:
  main()