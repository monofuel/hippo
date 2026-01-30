import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test hippoConstant memory - read-only cached constants
#
# Device constant memory must be backed by a device symbol. Using `const` here
# makes Nim emit a host-only compile-time constant, which fails on HIP.
const HostTestConstant = 42'i32
const HostTestArray: array[4, int32] = [10'i32, 20'i32, 30'i32, 40'i32]

var TestConstant {.hippoConstant.}: int32 = HostTestConstant
var TestArray {.hippoConstant.}: array[4, int32] = HostTestArray

proc useConstant(output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that reads from constant memory
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  let tid = threadIdx.x + blockIdx.x * blockDim.x

  # Use the constant value in computation
  outputArray[tid] = TestConstant + int32(tid)

proc useConstantArray(output: ptr[int32]){.hippoGlobal.} =
  ## Kernel that reads from constant array
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  let tid = threadIdx.x + blockIdx.x * blockDim.x

  # Access constant array element
  let arrayValue = TestArray[tid mod 4]
  outputArray[tid] = arrayValue * 2

suite "constant memory":
  test "constant_scalar":
    const NumThreads = 8
    var output = newSeq[int32](NumThreads)

    # Allocate GPU memory
    var dev_output = hippoMalloc(sizeof(int32) * NumThreads)

    # Launch kernel that uses constant scalar
    hippoLaunchKernel(
      useConstant,
      gridDim = newDim3(1.uint32),
      blockDim = newDim3(NumThreads.uint32),
      args = hippoArgs(dev_output.p)
    )

    # Copy results back
    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * NumThreads, HippoMemcpyDeviceToHost)

    # Verify results: each thread should have TestConstant + tid
    for i in 0..<NumThreads:
      let expected = int32(HostTestConstant + i)
      assert(output[i] == expected, fmt"Thread {i}: expected {expected}, got {output[i]}")

  test "constant_array":
    const NumThreads = 8
    var output = newSeq[int32](NumThreads)

    # Allocate GPU memory
    var dev_output = hippoMalloc(sizeof(int32) * NumThreads)

    # Ensure device constant memory is populated for array reads.
    hippoMemcpyToSymbol(TestArray, addr HostTestArray[0], sizeof(HostTestArray))

    # Launch kernel that uses constant array
    hippoLaunchKernel(
      useConstantArray,
      gridDim = newDim3(1.uint32),
      blockDim = newDim3(NumThreads.uint32),
      args = hippoArgs(dev_output.p)
    )

    # Copy results back
    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * NumThreads, HippoMemcpyDeviceToHost)

    # Verify results: each thread should have TestArray[tid mod 4] * 2
    for i in 0..<NumThreads:
      let arrayValue = HostTestArray[i mod 4]
      let expected = int32(arrayValue * 2)
      assert(output[i] == expected, fmt"Thread {i}: expected {expected} (from array[{i mod 4}]={arrayValue}), got {output[i]}")
