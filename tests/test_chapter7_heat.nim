import
  std/[unittest, math],
  hippo

when HippoRuntime in ["HIP", "HIP_CPU", "CUDA"]:
  import ./heat_diffusion

  type
    HeatAccess = enum
      GlobalMemory, Texture1D, Texture2D
    GridBuffer = object
      memory: GpuRef
      pitchBytes: int

  const
    BlockSide = 8'u32
    Speed = 0.25'f32
    Tolerance = 2e-5
    PaddingElements = 3
    GridSizes = [(1, 1), (1, 7), (9, 1), (19, 11), (32, 24)]
    StepCounts = [0, 1, 2, 3, 90]
    ProbeWidth = 5
    ProbeHeight = 3
    ProbeCount = 8
    ProbeX: array[ProbeCount, cfloat] = [0.5'f32, 4.5, 0.5, 4.5, -2.5, 9.5, 1.25, 1.75]
    ProbeY: array[ProbeCount, cfloat] = [0.5'f32, 0.5, 2.5, 2.5, 1.5, 9.5, 1.25, 1.75]
    ProbeIndices = [0, 4, 10, 14, 5, 14, 6, 6]

  proc writeHeaters(dst, heaters: ptr cfloat; width, height, pitch: cint) {.hippoGlobal.} =
    ## Chapter 7.3: restore heater cells in the current input before diffusion.
    let
      x = int(blockIdx.x * blockDim.x + threadIdx.x)
      y = int(blockIdx.y * blockDim.y + threadIdx.y)
    if x < int(width) and y < int(height):
      let
        index = y * int(pitch) + x
        value = cast[ptr UncheckedArray[cfloat]](heaters)[index]
      if value != 0.0'f32:
        cast[ptr UncheckedArray[cfloat]](dst)[index] = value

  proc writeHeaters1D(dst: ptr cfloat; heaters: HippoFloatTexture;
                      width, height, pitch: cint) {.hippoGlobal.} =
    ## Chapter 7.3.4: read heater cells through a linear texture object.
    let
      x = int(blockIdx.x * blockDim.x + threadIdx.x)
      y = int(blockIdx.y * blockDim.y + threadIdx.y)
    if x < int(width) and y < int(height):
      let value = hippoTex1DFetch(heaters, cint(y * int(width) + x))
      if value != 0.0'f32:
        cast[ptr UncheckedArray[cfloat]](dst)[y * int(pitch) + x] = value

  proc diffuseGlobal(dst, src: ptr cfloat; width, height, pitch: cint) {.hippoGlobal.} =
    ## Chapter 7.3: diffuse into a separate buffer, explicitly clamping neighbors.
    let
      x = int(blockIdx.x * blockDim.x + threadIdx.x)
      y = int(blockIdx.y * blockDim.y + threadIdx.y)
    if x < int(width) and y < int(height):
      let
        values = cast[ptr UncheckedArray[cfloat]](src)
        row = y * int(pitch)
        center = values[row + x]
        left = values[row + max(x - 1, 0)]
        right = values[row + min(x + 1, int(width) - 1)]
        top = values[max(y - 1, 0) * int(pitch) + x]
        bottom = values[min(y + 1, int(height) - 1) * int(pitch) + x]
      cast[ptr UncheckedArray[cfloat]](dst)[row + x] =
        center + Speed * (top + bottom + left + right - 4.0'f32 * center)

  proc diffuse1D(dst: ptr cfloat; src: HippoFloatTexture;
                 width, height, pitch: cint) {.hippoGlobal.} =
    ## Chapter 7.3.4: fetch the same stencil from a 1D texture over a 2D grid.
    let
      x = int(blockIdx.x * blockDim.x + threadIdx.x)
      y = int(blockIdx.y * blockDim.y + threadIdx.y)
    if x < int(width) and y < int(height):
      let
        row = y * int(width)
        center = hippoTex1DFetch(src, cint(row + x))
        left = hippoTex1DFetch(src, cint(row + max(x - 1, 0)))
        right = hippoTex1DFetch(src, cint(row + min(x + 1, int(width) - 1)))
        top = hippoTex1DFetch(src, cint(max(y - 1, 0) * int(width) + x))
        bottom = hippoTex1DFetch(src, cint(min(y + 1, int(height) - 1) * int(width) + x))
      cast[ptr UncheckedArray[cfloat]](dst)[y * int(pitch) + x] =
        center + Speed * (top + bottom + left + right - 4.0'f32 * center)

  proc probe1D(texture: HippoFloatTexture; output: ptr cfloat; count: cint) {.hippoGlobal.} =
    ## Fetch each linear element so its signed float32 format can be checked.
    let index = cint(threadIdx.x)
    if index < count:
      cast[ptr UncheckedArray[cfloat]](output)[index] = hippoTex1DFetch(texture, index)

  proc probe2D(texture: HippoFloatTexture; xs, ys, output: ptr cfloat) {.hippoGlobal.} =
    ## Probe corners, clamped coordinates and fractional nearest-neighbor reads.
    let index = int(threadIdx.x)
    if index < ProbeCount:
      cast[ptr UncheckedArray[cfloat]](output)[index] = hippoTex2D(texture,
        cast[ptr UncheckedArray[cfloat]](xs)[index],
        cast[ptr UncheckedArray[cfloat]](ys)[index])

  proc allocateGrid(width, height: int; access: HeatAccess): GridBuffer =
    ## Give 2D textures real row padding, including on HIP-CPU's tight allocator.
    if access == Texture2D:
      let allocation = hippoMallocPitch((width + PaddingElements) * sizeof(cfloat), height)
      result = GridBuffer(memory: allocation.memory, pitchBytes: allocation.pitchBytes)
    else:
      result = GridBuffer(memory: hippoMalloc(width * height * sizeof(cfloat)),
        pitchBytes: width * sizeof(cfloat))

  proc upload(grid: GridBuffer; values: var seq[cfloat]; width, height: int) =
    ## Copy logical rows without copying padding from the host reference buffer.
    hippoMemcpy2D(grid.memory.p, grid.pitchBytes, addr values[0],
      width * sizeof(cfloat), width * sizeof(cfloat), height, HippoMemcpyHostToDevice)

  proc makeTexture(grid: GridBuffer; width, height: int;
                   access: HeatAccess): HippoFloatTexture =
    ## Create a borrowed texture view once, before any simulation steps run.
    if access == Texture2D:
      result = hippoCreateFloatTexture2D(grid.memory, width, height, grid.pitchBytes)
    else:
      result = hippoCreateFloatTexture1D(grid.memory, width * height)

  proc referenceHeat(initial, heaters: seq[cfloat]; width, height, steps: int): seq[float64] =
    ## Independently compute the quarter-speed stencil as a double-precision mean.
    result = newSeq[float64](initial.len)
    var next = newSeq[float64](initial.len)
    for index in 0 ..< initial.len:
      result[index] = float64(initial[index])
    for step in 0 ..< steps:
      for index in 0 ..< heaters.len:
        if heaters[index] != 0.0'f32:
          result[index] = float64(heaters[index])
      for y in 0 ..< height:
        for x in 0 ..< width:
          var neighbors = 0.0
          for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            let
              column = min(max(x + delta[0], 0), width - 1)
              row = min(max(y + delta[1], 0), height - 1)
            neighbors += result[row * width + column]
          next[y * width + x] = neighbors / 4.0
      swap(result, next)

  proc checkHeat(access: HeatAccess; width, height, steps: int) =
    ## Compare every cell, including edges and heaters, after odd/even swaps.
    var
      initial = newSeq[cfloat](width * height)
      heaters = newSeq[cfloat](width * height)
      actual = newSeq[cfloat](width * height)
      input = allocateGrid(width, height, access)
      output = allocateGrid(width, height, access)
      sources = allocateGrid(width, height, access)
      inputTexture, outputTexture, sourceTexture: HippoFloatTexture
      widthArg = cint(width)
      heightArg = cint(height)
      pitchArg = cint(input.pitchBytes div sizeof(cfloat))
    for index in 0 ..< initial.len:
      initial[index] = cfloat(index mod 11) / 10.0'f32
    heaters[0] = 1.0'f32
    heaters[heaters.high] = 0.1'f32
    heaters[(height div 2) * width + width div 2] = 0.6'f32
    for index in 0 ..< initial.len:
      if heaters[index] != 0.0'f32:
        initial[index] = heaters[index]

    upload(input, initial, width, height)
    upload(sources, heaters, width, height)
    # Poison output storage to catch missing writes in partial blocks or pitch gaps.
    hippoMemset(output.memory, 0xFF, output.pitchBytes * height)
    if access != GlobalMemory:
      inputTexture = makeTexture(input, width, height, access)
      outputTexture = makeTexture(output, width, height, access)
      sourceTexture = makeTexture(sources, width, height, access)
    defer:
      if access != GlobalMemory:
        hippoDestroyFloatTexture(inputTexture)
        hippoDestroyFloatTexture(outputTexture)
        hippoDestroyFloatTexture(sourceTexture)

    # HIP-CPU can drain its default stream twice and overlap queued batches.
    # An explicit stream preserves the ordering between these dependent kernels.
    let
      stream = hippoStreamCreate()
      grid = newDim3((uint32(width) + BlockSide - 1) div BlockSide,
        (uint32(height) + BlockSide - 1) div BlockSide)
      blockSize = newDim3(BlockSide, BlockSide)
      expected = referenceHeat(initial, heaters, width, height, steps)
    defer: hippoStreamDestroy(stream)
    for step in 0 ..< steps:
      case access
      of GlobalMemory:
        hippoLaunchKernel(writeHeaters, gridDim = grid, blockDim = blockSize, stream = stream,
          args = hippoArgs(input.memory.p, sources.memory.p, widthArg, heightArg, pitchArg))
        hippoLaunchKernel(diffuseGlobal, gridDim = grid, blockDim = blockSize, stream = stream,
          args = hippoArgs(output.memory.p, input.memory.p, widthArg, heightArg, pitchArg))
      of Texture1D:
        hippoLaunchKernel(writeHeaters1D, gridDim = grid, blockDim = blockSize, stream = stream,
          args = hippoArgs(input.memory.p, sourceTexture, widthArg, heightArg, pitchArg))
        hippoLaunchKernel(diffuse1D, gridDim = grid, blockDim = blockSize, stream = stream,
          args = hippoArgs(output.memory.p, inputTexture, widthArg, heightArg, pitchArg))
      of Texture2D:
        hippoLaunchKernel(writeHeaters2D, gridDim = grid, blockDim = blockSize, stream = stream,
          args = hippoArgs(input.memory.p, sourceTexture, widthArg, heightArg, pitchArg))
        hippoLaunchKernel(diffuse2D, gridDim = grid, blockDim = blockSize, stream = stream,
          args = hippoArgs(output.memory.p, inputTexture, widthArg, heightArg, pitchArg))
      swap(input, output)
      swap(inputTexture, outputTexture)
    hippoStreamSynchronize(stream)
    hippoMemcpy2D(addr actual[0], width * sizeof(cfloat), input.memory.p,
      input.pitchBytes, width * sizeof(cfloat), height, HippoMemcpyDeviceToHost)
    checkpoint $access & " " & $width & "x" & $height & " steps=" & $steps
    for index in 0 ..< actual.len:
      doAssert abs(float64(actual[index]) - expected[index]) <= Tolerance,
        $access & " " & $width & "x" & $height & " steps=" & $steps &
        " cell=" & $index & " actual=" & $actual[index] & " expected=" & $expected[index]

  suite "Chapter 7.3: heat diffusion in global memory":
    test "matches the host stencil with heaters, edges and buffer swaps":
      for size in GridSizes:
        for steps in StepCounts:
          checkHeat(GlobalMemory, size[0], size[1], steps)

  suite "Chapter 7.3.4: heat diffusion through 1D textures":
    test "matches the host stencil with heaters, edges and buffer swaps":
      for size in GridSizes:
        for steps in StepCounts:
          checkHeat(Texture1D, size[0], size[1], steps)

    test "linear texture reads signed floats and sees backing-memory updates":
      var values: seq[cfloat] = @[-1.25'f32, 0.0'f32, 2.5'f32, -3.75'f32, 4.25'f32]
      let
        memory = hippoMalloc(values.len * sizeof(cfloat))
        texture = hippoCreateFloatTexture1D(memory, values.len)
        output = hippoMalloc(values.len * sizeof(cfloat))
      defer: hippoDestroyFloatTexture(texture)
      var
        actual = newSeq[cfloat](values.len)
        countArg = cint(values.len)
      for update in 0 .. 1:
        hippoMemcpy(memory, addr values[0], values.len * sizeof(cfloat), HippoMemcpyHostToDevice)
        hippoLaunchKernel(probe1D, blockDim = newDim3(BlockSide),
          args = hippoArgs(texture, output.p, countArg))
        hippoSynchronize()
        hippoMemcpy(addr actual[0], output, values.len * sizeof(cfloat), HippoMemcpyDeviceToHost)
        check actual == values
        for value in values.mitems:
          value += 0.5'f32

  suite "Chapter 7: heat diffusion through 2D textures":
    test "matches the host stencil with pitched rows and sampler-clamped edges":
      for size in GridSizes:
        for steps in StepCounts:
          checkHeat(Texture2D, size[0], size[1], steps)

    test "point sampling respects row pitch, clamps edges and sees updated data":
      var
        values = newSeq[cfloat](ProbeWidth * ProbeHeight)
        actual = newSeq[cfloat](ProbeCount)
        xs = ProbeX
        ys = ProbeY
      let
        grid = allocateGrid(ProbeWidth, ProbeHeight, Texture2D)
        texture = makeTexture(grid, ProbeWidth, ProbeHeight, Texture2D)
        devXs = hippoMalloc(sizeof(ProbeX))
        devYs = hippoMalloc(sizeof(ProbeY))
        output = hippoMalloc(ProbeCount * sizeof(cfloat))
      defer: hippoDestroyFloatTexture(texture)
      check grid.pitchBytes > ProbeWidth * sizeof(cfloat)
      hippoMemcpy(devXs, addr xs[0], sizeof(ProbeX), HippoMemcpyHostToDevice)
      hippoMemcpy(devYs, addr ys[0], sizeof(ProbeY), HippoMemcpyHostToDevice)
      hippoMemset(grid.memory, 0xFF, grid.pitchBytes * ProbeHeight)
      for index in 0 ..< values.len:
        values[index] = cfloat(index - 7) * 0.25'f32
      for update in 0 .. 1:
        upload(grid, values, ProbeWidth, ProbeHeight)
        hippoLaunchKernel(probe2D, blockDim = newDim3(BlockSide),
          args = hippoArgs(texture, devXs.p, devYs.p, output.p))
        hippoSynchronize()
        hippoMemcpy(addr actual[0], output, actual.len * sizeof(cfloat), HippoMemcpyDeviceToHost)
        for index in 0 ..< ProbeCount:
          check actual[index] == values[ProbeIndices[index]]
        for value in values.mitems:
          value += 0.5'f32

    test "rejects invalid float texture extents and pitches":
      let memory = hippoMalloc(ProbeWidth * ProbeHeight * sizeof(cfloat))
      expect ValueError:
        discard hippoCreateFloatTexture1D(memory, 0)
      expect ValueError:
        discard hippoCreateFloatTexture2D(memory, 0, ProbeHeight, ProbeWidth * sizeof(cfloat))
      expect ValueError:
        discard hippoCreateFloatTexture2D(memory, ProbeWidth, ProbeHeight, sizeof(cfloat))
else:
  suite "Chapter 7: heat diffusion":
    test "SIMPLE texture support is deferred":
      skip()
