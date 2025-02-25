import hippo


{.emit: """
#include <cuda_runtime.h>
#include <mma.h>        // For WMMA operations
#include <cuda_fp16.h>  // For half-precision data type

using namespace nvcuda;

// Kernel using WMMA for 16x16 matrix multiplication
__global__ void wmmaKernel(__half* d_a, __half* d_b, float* d_c, int M, int N, int K) {
    // Declare WMMA fragments for 16x16x16 tile
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC;

    // Leading dimension (assuming contiguous storage for 16x16 matrices)
    unsigned ldm = 16;

    // Load matrix A from global memory
    wmma::load_matrix_sync(fragA, d_a, ldm);

    // Load matrix B from global memory
    wmma::load_matrix_sync(fragB, d_b, ldm);

    // Initialize the accumulator fragment to zero
    wmma::fill_fragment(fragC, 0.0f);

    // Perform the matrix multiply-accumulate operation
    wmma::mma_sync(fragC, fragA, fragB, fragC);

    // Store the result to global memory (row-major)
    wmma::store_matrix_sync(d_c, fragC, ldm, wmma::mem_row_major);
}
"""
.}


# Import the half-precision type from CUDA
type
  half* {.importcpp: "__half", header: "cuda_fp16.h".} = object

#   # Define WMMA fragment types for matrix A, B, and the accumulator C
#   WmmaFragmentA* {.importcpp: "wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>", header: "mma.h".} = object
#   WmmaFragmentB* {.importcpp: "wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>", header: "mma.h".} = object
#   WmmaFragmentC* {.importcpp: "wmma::fragment<wmma::accumulator, 16, 16, 16, float>", header: "mma.h".} = object

# # Utility to convert float to half
proc float2half*(f: cfloat): half {.importcpp: "__float2half(@)", header: "cuda_fp16.h".}

proc wmmaKernel*(A: ptr half, B: ptr half, C: ptr cfloat, M: cint, N: cint, K: cint) {.importcpp: "wmmaKernel".}
# # Define memory layout constants
# const
#   WmmaMemRowMajor* = 0.cuint  # wmma::mem_row_major
#   WmmaMemColMajor* = 1.cuint  # wmma::mem_col_major

# # Import WMMA operations
# proc loadMatrixSync*(frag: var WmmaFragmentA, mem: ptr half, ldm: cuint, layout: cuint) 
#   {.importcpp: "wmma::load_matrix_sync(@, #, #, #)", header: "mma.h".}
# proc loadMatrixSync*(frag: var WmmaFragmentB, mem: ptr half, ldm: cuint, layout: cuint) 
#   {.importcpp: "wmma::load_matrix_sync(@, #, #, #)", header: "mma.h".}
# proc fillFragment*(frag: var WmmaFragmentC, value: cfloat) 
#   {.importcpp: "wmma::fill_fragment(@, #)", header: "mma.h".}
# proc mmaSync*(acc: var WmmaFragmentC, a: WmmaFragmentA, b: WmmaFragmentB, c: WmmaFragmentC) 
#   {.importcpp: "wmma::mma_sync(@, #, #, #)", header: "mma.h".}
# proc storeMatrixSync*(mem: ptr cfloat, frag: WmmaFragmentC, ldm: cuint, layout: cuint) 
#   {.importcpp: "wmma::store_matrix_sync(@, #, #, #)", header: "mma.h".}

# proc wmmaKernel(A: ptr half, B: ptr half, C: ptr cfloat, M: cint, N: cint, K: cint) {.hippoGlobal.} =
#   var fragA: WmmaFragmentA
#   var fragB: WmmaFragmentB
#   var fragC: WmmaFragmentC

#   let ldm = 16.cuint  # Leading dimension, assuming M=N=K=16

#   # Load matrices A and B from global memory (assuming row-major storage)
#   loadMatrixSync(fragA, A, ldm, WmmaMemRowMajor)
#   loadMatrixSync(fragB, B, ldm, WmmaMemRowMajor)

#   # Initialize accumulator to zero
#   fillFragment(fragC, 0.0f)

#   # Perform matrix multiply-accumulate
#   mmaSync(fragC, fragA, fragB, fragC)

#   # Store result to global memory
#   storeMatrixSync(C, fragC, ldm, WmmaMemRowMajor)


# Matrix dimensions (fixed to 16×16 for this example)
let M = 16
let K = 16
let N = 16

# Allocate and initialize host memory
var h_A = newSeq[half](M * K)
var h_B = newSeq[half](K * N)
var h_C = newSeq[cfloat](M * N)

for i in 0 ..< M * K:
  h_A[i] = float2half(i.float32 / (M * K).float32)  # Sample data
for i in 0 ..< K * N:
  h_B[i] = float2half(i.float32 / (K * N).float32)  # Sample data

# Allocate device memory
var d_A = hippoMalloc(M * K * sizeof(half))
var d_B = hippoMalloc(K * N * sizeof(half))
var d_C = hippoMalloc(M * N * sizeof(cfloat))

# Copy data to device
hippoMemcpy(d_A, addr h_A[0], M * K * sizeof(half), HippoMemcpyHostToDevice)
hippoMemcpy(d_B, addr h_B[0], K * N * sizeof(half), HippoMemcpyHostToDevice)

# Launch the kernel (1 block of 32 threads = 1 warp)
let gridDim = newDim3(1, 1, 1)
let blockDim = newDim3(32, 1, 1)  # At least one warp
hippoLaunchKernel(
  wmmaKernel,
  gridDim,
  blockDim,
  args = @[addr d_A.p, addr d_B.p, addr d_C.p, cast[ptr pointer](M), cast[ptr pointer](N), cast[ptr pointer](K)]
)

# Synchronize and copy result back
hippoSynchronize()
hippoMemcpy(addr h_C[0], d_C, M * N * sizeof(cfloat), HippoMemcpyDeviceToHost)

# h_C now contains the result (C = A × B)