## Manual Chapter 7 heat-map demo; excluded from automated test tasks.
## Dependencies: `nimble install windy boxy pixie opengl`.
## Build and run from the repository root:
##
## .. code-block:: sh
##
##   nim cpp --skipParentCfg -d:release -r tests/gui_heatmap.nim
##   nim cpp --skipParentCfg --cc:hipcc -d:release -d:useMalloc -d:HippoRuntime:HIP -r tests/gui_heatmap.nim
##
## The first command uses HIP-CPU; the second uses AMD HIP.
## On NixOS, `nix develop .#basic` supplies the system GUI libraries.
## The compiled backend is independent of the Nix shell's name.
## Use --help for controls, frame limits and PNG output options.

# Keep the GUI dependencies' CPU SIMD intrinsics out of HIP device compilation.
{.define: pixieNoSimd.}
{.define: zippyNoSimd.}
{.define: crunchyNoSimd.}

import
  std/[os, parseopt, strutils],
  boxy, pixie, opengl, windy,
  hippo,
  ./heat_diffusion

type Options = object
  frames: int
  screenshot: string

const
  GridWidth = 128
  GridHeight = 128
  WindowSide = 768
  BlockSide = 8'u32
  StepsPerFrame = 4
  FrameDelayMs = 16
  RowBytes = GridWidth * sizeof(cfloat)
  DefaultScreenshot = "heatmap.png"
  Palette = [
    [12, 18, 48], [26, 70, 145], [23, 173, 190],
    [246, 218, 87], [241, 99, 42], [180, 20, 43]
  ]
  Help = """Chapter 7 heat diffusion (HIP / HIP-CPU).
Space: pause/resume. R: reset. S: save heatmap.png. Esc: close.
--frames=N         Close after N rendered frames (default: run until closed).
--screenshot=PATH  Save the final heat map as a PNG when closing.
--help             Show these controls.
"""

proc readOptions(): Options =
  ## Parse optional frame limits and PNG output for a reproducible demo capture.
  for kind, key, value in getopt():
    case kind
    of cmdLongOption:
      case key
      of "frames":
        result.frames = parseInt(value)
        if result.frames <= 0:
          raise newException(ValueError, "--frames must be positive.")
      of "screenshot":
        if value.len == 0:
          raise newException(ValueError, "--screenshot needs a PNG path.")
        result.screenshot = value
      of "help":
        echo Help
        quit(0)
      else:
        raise newException(ValueError, "Unknown option: --" & key)
    else:
      raise newException(ValueError, "Use --help for the demo's options.")

proc heatColor(temperature: cfloat): ColorRGBX =
  ## Map the fixed temperature range [0, 1] from deep blue through yellow to red.
  let
    position = clamp(float(temperature), 0.0, 1.0) * float(Palette.high)
    segment = min(int(position), Palette.high - 1)
    fraction = position - float(segment)
  var channels: array[3, uint8]
  for channel in 0 .. 2:
    channels[channel] = uint8(float(Palette[segment][channel]) * (1.0 - fraction) +
      float(Palette[segment + 1][channel]) * fraction)
  rgbx(channels[0], channels[1], channels[2], 255)

proc main() =
  ## Animate the tested Chapter 7 2D texture kernels in a resizable Windy window.
  let
    options = readOptions()
    first = hippoMallocPitch(RowBytes, GridHeight)
    second = hippoMallocPitch(RowBytes, GridHeight)
    sources = hippoMallocPitch(RowBytes, GridHeight)
    sourceTexture = hippoCreateFloatTexture2D(sources.memory, GridWidth,
      GridHeight, sources.pitchBytes)
    stream = hippoStreamCreate()
    image = newImage(GridWidth, GridHeight)
    window = newWindow("Hippo — Chapter 7 heat diffusion", ivec2(WindowSide, WindowSide))
  defer: window.close()
  defer: hippoStreamDestroy(stream)
  defer: hippoDestroyFloatTexture(sourceTexture)
  var
    input = first.memory
    output = second.memory
    inputTexture = hippoCreateFloatTexture2D(input, GridWidth, GridHeight, first.pitchBytes)
    outputTexture = hippoCreateFloatTexture2D(output, GridWidth, GridHeight, second.pitchBytes)
    inputPitch = first.pitchBytes
    outputPitch = second.pitchBytes
    widthArg = cint(GridWidth)
    heightArg = cint(GridHeight)
    initial = newSeq[cfloat](GridWidth * GridHeight)
    heaters = newSeq[cfloat](GridWidth * GridHeight)
    values = newSeq[cfloat](GridWidth * GridHeight)
    paused, resetRequested, saveRequested: bool
    frames, steps: int
  defer:
    hippoDestroyFloatTexture(inputTexture)
    hippoDestroyFloatTexture(outputTexture)

  for y in 0 ..< GridHeight:
    for x in 0 ..< GridWidth:
      let index = y * GridWidth + x
      initial[index] = 0.1'f32 + 0.15'f32 * cfloat(x) / cfloat(GridWidth)
      if x in 18 .. 43 and y in 17 .. 43:
        heaters[index] = 1.0'f32
      elif (x - 90) * (x - 90) + (y - 36) * (y - 36) < 16 * 16:
        heaters[index] = 0.05'f32
      elif x in 65 .. 105 and y in 86 .. 101:
        heaters[index] = 0.8'f32
      if heaters[index] != 0.0'f32:
        initial[index] = heaters[index]

  proc reset() =
    ## Restore initial temperatures while retaining the allocated texture views.
    hippoMemcpy2D(input.p, inputPitch, addr initial[0], RowBytes,
      RowBytes, GridHeight, HippoMemcpyHostToDevice)
    hippoMemcpy2D(output.p, outputPitch, addr initial[0], RowBytes,
      RowBytes, GridHeight, HippoMemcpyHostToDevice)
    hippoMemcpy2D(sources.memory.p, sources.pitchBytes, addr heaters[0], RowBytes,
      RowBytes, GridHeight, HippoMemcpyHostToDevice)
    steps = 0

  reset()
  window.makeContextCurrent()
  loadExtensions()
  let
    renderer = newBoxy()
    grid = newDim3((uint32(GridWidth) + BlockSide - 1) div BlockSide,
      (uint32(GridHeight) + BlockSide - 1) div BlockSide)
    blockSize = newDim3(BlockSide, BlockSide)
  window.onButtonPress = proc(button: Button) =
    ## Keep the simulation interactive without requiring a separate UI framework.
    case button
    of KeySpace: paused = not paused
    of KeyR: resetRequested = true
    of KeyS: saveRequested = true
    of KeyEscape: window.closeRequested = true
    else: discard

  while not window.closeRequested:
    pollEvents()
    if window.closeRequested:
      break
    if resetRequested:
      reset()
      resetRequested = false
    if not paused:
      for iteration in 0 ..< StepsPerFrame:
        var pitchArg = cint(inputPitch div sizeof(cfloat))
        hippoLaunchKernel(writeHeaters2D, gridDim = grid, blockDim = blockSize,
          stream = stream,
          args = hippoArgs(input.p, sourceTexture, widthArg, heightArg, pitchArg))
        pitchArg = cint(outputPitch div sizeof(cfloat))
        hippoLaunchKernel(diffuse2D, gridDim = grid, blockDim = blockSize,
          stream = stream,
          args = hippoArgs(output.p, inputTexture, widthArg, heightArg, pitchArg))
        swap(input, output)
        swap(inputTexture, outputTexture)
        swap(inputPitch, outputPitch)
        inc steps
      hippoStreamSynchronize(stream)
    hippoMemcpy2D(addr values[0], RowBytes, input.p, inputPitch,
      RowBytes, GridHeight, HippoMemcpyDeviceToHost)
    for index, temperature in values:
      image.data[index] = heatColor(temperature)
    if saveRequested:
      image.writeFile(DefaultScreenshot)
      saveRequested = false
    renderer.addImage("heat", image)
    renderer.beginFrame(window.size)
    renderer.drawImage("heat", rect = rect(vec2(0, 0), window.size.vec2))
    renderer.endFrame()
    window.swapBuffers()
    window.title = "Hippo Chapter 7 | " & HippoRuntime & " | step " & $steps &
      (if paused: " | PAUSED" else: "") & " | Space: pause · R: reset · S: PNG · Esc: close"
    inc frames
    if options.frames > 0 and frames >= options.frames:
      break
    sleep(FrameDelayMs)
  if options.screenshot.len > 0:
    image.writeFile(options.screenshot)

when isMainModule:
  main()
