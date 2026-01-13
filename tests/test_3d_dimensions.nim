import
  hippo,
  std/[unittest, strformat],
  ./utils

# Test 3D grid and block dimensions (y, z axes)

const
  GridX = 2.uint32
  GridY = 3.uint32
  GridZ = 2.uint32
  BlockX = 4.uint32
  BlockY = 2.uint32
  BlockZ = 3.uint32

proc test3DCoords(output: ptr[int32]){.hippoGlobal.} =
  # Calculate total threads per block and total blocks
  let totalThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z
  let totalBlocks = gridDim.x * gridDim.y * gridDim.z

  # Calculate linear thread index within block
  let threadLinear = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y

  # Calculate linear block index within grid
  let blockLinear = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y

  # Calculate global linear index
  let globalIndex = blockLinear * totalThreadsPerBlock + threadLinear

  # Write coordinates as: x*10000 + y*100 + z
  let coords = int32(blockIdx.x * 10000 + blockIdx.y * 100 + blockIdx.z)
  let outputArray = cast[ptr UncheckedArray[int32]](output)
  outputArray[globalIndex] = coords

suite "3d dimensions":
  testSkipPlatforms "3d_coords", "SIMPLE":
    # Calculate expected dimensions
    let totalBlocks = int(GridX * GridY * GridZ)
    let totalThreadsPerBlock = int(BlockX * BlockY * BlockZ)
    let totalElements = totalBlocks * totalThreadsPerBlock

    var output = newSeq[int32](totalElements)

    # Allocate GPU memory
    var dev_output = hippoMalloc(sizeof(int32) * totalElements)

    # Launch kernel with 3D grid and block dimensions
    hippoLaunchKernel(
      test3DCoords,
      gridDim = newDim3(GridX, GridY, GridZ),
      blockDim = newDim3(BlockX, BlockY, BlockZ),
      args = hippoArgs(dev_output.p)
    )

    # Copy results back
    hippoMemcpy(addr output[0], dev_output, sizeof(int32) * totalElements, HippoMemcpyDeviceToHost)

    # Verify coordinates
    var index = 0
    for gz in 0..<GridZ:
      for gy in 0..<GridY:
        for gx in 0..<GridX:
          for bz in 0..<BlockZ:
            for by in 0..<BlockY:
              for bx in 0..<BlockX:
                let expectedCoords = int32(gx * 10000 + gy * 100 + gz)
                assert(output[index] == expectedCoords, fmt"Index {index}: expected {expectedCoords}, got {output[index]}")
                index += 1
