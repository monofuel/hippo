# nim cpp --cc:nvcc --define:"useMalloc" hello_kernel.nim
# minimal cuda example for nim without using hippo
{.emit: """
__global__ void add(int a, int b) {
  int c;
  c = a + b;
}
""".}

{.emit: """
add<<<1,1>>>(2,7);
""".}
