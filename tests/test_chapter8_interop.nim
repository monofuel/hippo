import
  std/[unittest, math, strutils],
  hippo

when defined(linux) and (HippoRuntime == "HIP" or HippoRuntime == "HIP_CPU"):
  import ./[gl_interop, chapter8_pixels]

  const
    Sizes = [(1, 1), (19, 11), (32, 24)]
    Ticks = [0'i32, 7, 83]

  proc checkPixels(api: GLApi; width, height: int; stream: HippoStream) =
    ## Check GL-visible bytes and a texture upload after repeated map/unmap cycles.
    var
      buffer = newInteropBuffer(api, width * height * sizeof(Pixel))
      texture: cuint
      actual = newSeq[Pixel](width * height)
      texels = newSeq[Pixel](width * height)
      temperatures = newSeq[cfloat](width * height)
      widthArg = cint(width)
      heightArg = cint(height)
      pitchArg = cint(width)
    let memory = hippoMalloc(temperatures.len * sizeof(cfloat))
    defer: buffer.close(stream)
    api.genTextures(1, addr texture)
    defer: api.deleteTextures(1, addr texture)
    api.bindTexture(Texture2D, texture)
    api.texImage2D(Texture2D, 0, Rgba8, widthArg, heightArg, 0, Rgba, UnsignedByte, nil)
    for index in 0 ..< temperatures.len:
      temperatures[index] = cfloat(index mod 17) / 16.0'f32
    hippoMemcpy(memory, addr temperatures[0], temperatures.len * sizeof(cfloat), HippoMemcpyHostToDevice)
    for mode in BasicPattern .. HeatMap:
      for tick in Ticks:
        var
          modeArg = mode
          tickArg = tick
          mapped = buffer.map(stream)
        try:
          hippoLaunchKernel(renderPixels,
            gridDim = newDim3((uint32(width) + BlockSide - 1) div BlockSide,
              (uint32(height) + BlockSide - 1) div BlockSide),
            blockDim = newDim3(BlockSide, BlockSide), stream = stream,
            args = hippoArgs(mapped, widthArg, heightArg, tickArg, modeArg, memory.p, pitchArg))
        finally:
          buffer.unmap(stream)
        api.bindBuffer(PixelUnpackBuffer, buffer.name)
        api.getBufferSubData(PixelUnpackBuffer, 0, buffer.sizeBytes, addr actual[0])
        api.bindBuffer(PixelUnpackBuffer, 0)
        buffer.uploadTexture(texture, width, height)
        api.bindTexture(Texture2D, texture)
        api.getTexImage(Texture2D, 0, Rgba, UnsignedByte, addr texels[0])
        doAssert api.getError() == 0, "OpenGL rejected buffer readback or texture upload."
        doAssert actual == texels, "The PBO and displayed texture disagree."
        for y in 0 ..< height:
          for x in 0 ..< width:
            let index = y * width + x
            var expected: Pixel
            case mode
            of BasicPattern:
              let phase = abs((float64(x) / float64(width) - 0.5) * 100.0) -
                abs((float64(y) / float64(height) - 0.5) * 100.0)
              expected = Pixel(g: uint8(128.0 + 127.0 * math.sin(phase)), a: 255)
            of Ripples:
              let
                dx = float64(x - width div 2)
                dy = float64(y - height div 2)
                distance = math.sqrt(dx * dx + dy * dy) / 10.0
                grey = uint8(128.0 + 127.0 * math.cos(distance - float64(tick) / 7.0) / (distance + 1.0))
              expected = Pixel(r: grey, g: grey, b: grey, a: 255)
            else:
              # Known ramp stops and independent channel-wise triangle functions.
              let t = float64(temperatures[index])
              expected = Pixel(
                r: uint8(clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0) * 255.0),
                g: uint8(clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0) * 255.0),
                b: uint8(clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0) * 255.0), a: 255)
            doAssert abs(int(actual[index].r) - int(expected.r)) <= 1 and
              abs(int(actual[index].g) - int(expected.g)) <= 1 and
              abs(int(actual[index].b) - int(expected.b)) <= 1 and actual[index].a == 255,
              "Wrong Chapter 8 pixel: mode=" & $mode & " tick=" & $tick &
                " size=" & $width & "x" & $height & " index=" & $index

  proc checkReadWrite(api: GLApi; stream: HippoStream) =
    ## Verify GL-to-compute visibility and preservation of untouched boundary pixels.
    const Count = 37
    var
      buffer: InteropBuffer
      values, actual: array[Count, Pixel]
      countArg = cint(Count)
    for index in 0 ..< Count:
      values[index] = Pixel(r: uint8(index), g: uint8(255 - index), b: 73, a: 255)
    buffer = newInteropBuffer(api, sizeof(values), writeDiscard = false, initial = addr values[0])
    defer: buffer.close(stream)
    let mapped = buffer.map(stream)
    try:
      hippoLaunchKernel(modifyPixels, gridDim = newDim3(5), blockDim = newDim3(BlockSide),
        stream = stream, args = hippoArgs(mapped, countArg))
    finally:
      buffer.unmap(stream)
    api.bindBuffer(PixelUnpackBuffer, buffer.name)
    api.getBufferSubData(PixelUnpackBuffer, 0, sizeof(actual), addr actual[0])
    api.bindBuffer(PixelUnpackBuffer, 0)
    for index in 1 ..< Count - 1:
      values[index].r = values[index].r xor 0x5A'u8
      values[index].g = values[index].g xor 0xA5'u8
    doAssert actual == values, "Read/write mapping lost old data or changed boundary pixels."
    doAssert api.getError() == 0

  suite "Chapter 8: OpenGL pixel-buffer interoperability":
    test "basic pattern, animated ripples and heat colors survive compute/GL handoffs":
      var context: GLTestContext
      try:
        context = newTestContext()
      except OpenGLUnavailableError:
        checkpoint getCurrentExceptionMsg()
      defer: context.close()
      var canRun = context != nil
      when HippoRuntime == "HIP":
        if canRun:
          let renderer = ($context.api.getString(Renderer)).toLowerAscii()
          if "llvmpipe" in renderer or "softpipe" in renderer or "software" in renderer:
            checkpoint "Native HIP sharing requires hardware OpenGL, not " & renderer
            canRun = false
      if not canRun:
        skip()
      else:
        when HippoRuntime == "HIP":
          discard hippoUseGLDevice()
        let stream = hippoStreamCreate()
        defer: hippoStreamDestroy(stream)
        for size in Sizes:
          checkPixels(context.api, size[0], size[1], stream)
        checkReadWrite(context.api, stream)
else:
  suite "Chapter 8: OpenGL pixel-buffer interoperability":
    test "Linux HIP and HIP-CPU only; other platforms are deferred":
      skip()
