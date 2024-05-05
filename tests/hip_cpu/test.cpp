#include <hip/hip_runtime.h>
#include <iostream>

// g++ --std=c++17 -I../../HIP-CPU/include -ltbb test.cpp
// clang -I../../HIP-CPU/include -ltbb -lstdc++ test.cpp

__global__
void kernel(const int* pA, const int* pB, int* pC) {
    const auto gidx = blockIdx.x * blockDim.x + threadIdx.x;

    pC[gidx] = pA[gidx] + pB[gidx];
}

int main() {
    int a[]{1, 2, 3, 4, 5};
    int b[]{6, 7, 8, 9, 10};
    int c[sizeof(a) / sizeof(a[0])];

    int* pA{nullptr}; hipMalloc((void**)&pA, sizeof(a));
    int* pB{nullptr}; hipMalloc((void**)&pB, sizeof(b));
    int* pC{nullptr}; hipMalloc((void**)&pC, sizeof(c));

    hipMemcpy(pA, a, sizeof(a), hipMemcpyHostToDevice);
    hipMemcpy(pB, b, sizeof(b), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(
        kernel,
        dim3(1),
        dim3(sizeof(a) / sizeof(a[0])),
        0,
        nullptr,
        pA,
        pB,
        pC);

    hipMemcpy(c, pC, sizeof(c), hipMemcpyDeviceToHost);

    for (auto i = 0u; i != sizeof(a) / sizeof(a[0]); ++i) {
      if (c[i] != a[i] + b[i]) throw;
      std::cout << "a[" << i << "] = " << a[i] << ", ";
      std::cout << "b[" << i << "] = " << b[i] << ", ";
      std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    return 0;
}