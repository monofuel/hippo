# https://gpuopen.com/learn/wmma_on_rdna3/
# Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic
# Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

import hippo, std/strformat




{.emit:"""

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Use half16 as an alias of the internal clang vector type of 16 fp16 values
typedef _Float16 half16 __attribute__((ext_vector_type(16)));

__global__ void wmma_matmul(__half* a, __half* b, __half* c)
{
    const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lIdx = threadIdx.x;

    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
    // a_frag will store one column of the 16x16 matrix A tile
    // b_frag will store one row of the 16x16 matrix B tile
    half16 a_frag;
    half16 b_frag;
    // initialize c fragment to 0
    half16 c_frag = {};

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    const int lane = lIdx % 16;

    for (int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16*ele + lane];
    }

    for (int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * lane + ele];
    }

    // call the WMMA intrinsic with OPSEL set to "false"
    c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

    for (int ele = 0; ele < 8; ++ele)
    {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_frag output
        c[16 * r + lane] = c_frag[ele*2];
        // if OPSEL was set to "true", the line above would instead be
        // c[16 * r + lane] = c_frag[ele*2 + 1];
    }

}
"""
.}

proc wmmaMatmul(a: ptr half, b: ptr half, c: ptr half) {.importcpp: "wmma_matmul".}


proc main() =

  # {.emit: """
  # __half a[16 * 16] = {};
  # __half b[16 * 16] = {};
  # __half c[16 * 16] = {};
  # __half *a_gpu, *b_gpu, *c_gpu;
  # hipMalloc(&a_gpu, 16*16 * sizeof(__half));
  # hipMalloc(&b_gpu, 16*16 * sizeof(__half));
  # hipMalloc(&c_gpu, 16*16 * sizeof(__half));

  # // fill in some data into matrices A and B
  # for (int i = 0; i < 16; ++i)
  # {
  #     for (int j = 0; j < 16; ++j)
  #     {
  #         a[i * 16 + j] = (__half)1.f;
  #         b[i * 16 + j] = (__half)1.f;
  #     }
  # }

  # hipMemcpy(a_gpu, a, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
  # hipMemcpy(b_gpu, b, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
  # hipMemcpy(c_gpu, c, (16*16) * sizeof(__half), hipMemcpyHostToDevice);

  # wmma_matmul<<<dim3(1), dim3(32, 1, 1), 0, 0>>>(a_gpu, b_gpu, c_gpu);

  # hipMemcpy(c, c_gpu, (16 * 16) * sizeof(__half), hipMemcpyDeviceToHost);

  # hipFree(a_gpu);
  # hipFree(b_gpu);
  # hipFree(c_gpu);

  # for (int i = 0; i < 16; ++i)
  # {
  #     for (int j = 0; j < 16; ++j)
  #     {
  #         printf("%f ", (float)c[i * 16 + j]);
  #     }
  #     printf("\\n");
  # }
  # """.}


  var
    a: array[16*16, half]
    b: array[16*16, half]
    c: array[16*16, half]

  for i in 0..<16:
    for j in 0..<16:
      a[i*16 + j] = toHalf(1.0)
      b[i*16 + j] = toHalf(1.0)

  var dev_a = hippoMalloc(sizeof(half)*16*16)
  var dev_b = hippoMalloc(sizeof(half)*16*16)
  var dev_c = hippoMalloc(sizeof(half)*16*16)

  hippoMemcpy(dev_a, addr a[0], sizeof(half)*16*16, hipMemcpyHostToDevice)
  hippoMemcpy(dev_b, addr b[0], sizeof(half)*16*16, hipMemcpyHostToDevice)

  hippoLaunchKernel(
    wmmaMatmul,
    gridDim = newDim3(1),
    blockDim = newDim3(32, 1, 1),
    args = hippoArgs(dev_a.p, dev_b.p, dev_c.p)
  )

  hippoMemcpy(addr c[0], dev_c, sizeof(half)*16*16, hipMemcpyDeviceToHost)

  for i in 0..<16:
    for j in 0..<16:
      stdout.write fmt"{c[i*16 + j]} "
    stdout.write "\n"

  # Hippo automatically frees gpu memory when it goes out of scope

when isMainModule:
  main()