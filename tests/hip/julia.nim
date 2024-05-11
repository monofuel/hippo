import hippo

const DIM = 1000

proc juliaKernel(p: pointer) {.hippoGlobal.} =
  let x = threadIdx.x
  let y = threadIdx.y
  # let c = complex(-0.8, 0.156)
  # let z = complex(2.0 * x / DIM - 1.0, 2.0 * y / DIM - 1.0)
  # let i = 0
  # while i < 256 and z.abs < 2.0:
  #   z = z * z + c
  #   inc i
  # p[x + y * DIM] = i

proc main() =
  # TODO init bitmap
  var res: array[DIM * DIM, int32]

  let devBitmap = hippoMalloc(DIM * DIM * sizeof(int32))
  let grid = newDim3(DIM,DIM)
  hippoLaunchKernel(
    juliaKernel,
    gridDim=grid,
    args = (devBitmap,)
  )
  hippoMemcpy(addr res[0], devBitmap, DIM * DIM * sizeof(int32), HippoMemcpyDeviceToHost)
  # TODO display and exit
  hippoFree(devBitmap)


when isMainModule:
  main()