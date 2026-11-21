## Manual Chapter 8 OpenGL interop demo; excluded from automated test tasks.
## Dependencies: `nimble install windy pixie opengl`.
## Build and run from the repository root:
##
## .. code-block:: sh
##
##   nim cpp --skipParentCfg --cc:hipcc -d:release -d:useMalloc -d:HippoRuntime:HIP -r tests/gui_interop.nim
##   nim cpp --skipParentCfg -d:release -r tests/gui_interop.nim
##
## The first command uses native AMD HIP/GL sharing on Linux; the second uses
## HIP-CPU and a host-mapped GL buffer, labelled in the window title.
## Normal HIP frames share the GL buffer; explicit PNG captures read it back.
## On NixOS, `nix develop .#basic` supplies the system GUI libraries.
## The compiled backend is independent of the Nix shell's name.
## Use --help for view selection, controls, frame limits and PNG output options.

# Keep CPU-only SIMD intrinsics out of HIP device compilation.
{.define: pixieNoSimd.}
{.define: zippyNoSimd.}
{.define: crunchyNoSimd.}

import
  std/[os, parseopt, strutils],
  windy, opengl, pixie,
  hippo,
  ./[gl_interop, chapter8_pixels, heat_diffusion]

type Options = object
  frames: int
  mode: cint
  screenshot: string

const
  Width = 128
  Height = 128
  WindowSide = 768
  HeatStepsPerFrame = 4
  FrameDelayMs = 16
  RowBytes = Width * sizeof(cfloat)
  DefaultScreenshot = "interop.png"
  ModeNames = ["Basic pattern", "Ripples", "Heat diffusion"]
  Help = """Chapter 8 OpenGL pixel-buffer interoperability (Linux HIP / HIP-CPU).
1: basic pattern. 2: animated ripples. 3: heat diffusion.
Space: pause. R: reset. S: save interop.png. Esc: close.
--mode=basic|ripple|heat  Select the initial view (default: heat).
--frames=N              Close after N frames (default: run until closed).
--screenshot=PATH       Save the final rendered window to PNG.
--help                  Show these controls.
"""
  VertexShader = """#version 330 core
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
  FragmentShader = """#version 330 core
in vec2 uv;
uniform sampler2D pixels;
out vec4 color;
void main() {
    color = texture(pixels, vec2(uv.x, 1.0 - uv.y));
}
"""

proc readOptions(): Options =
  ## Read ordinary application arguments without changing the test configuration.
  result.mode = HeatMap
  for kind, key, value in getopt():
    if kind != cmdLongOption:
      raise newException(ValueError, "Use --help for the demo's options.")
    case key
    of "frames":
      result.frames = parseInt(value)
      if result.frames <= 0:
        raise newException(ValueError, "--frames must be positive.")
    of "mode":
      case value
      of "basic": result.mode = BasicPattern
      of "ripple": result.mode = Ripples
      of "heat": result.mode = HeatMap
      else: raise newException(ValueError, "--mode must be basic, ripple or heat.")
    of "screenshot":
      if value.len == 0:
        raise newException(ValueError, "--screenshot needs a PNG path.")
      result.screenshot = value
    of "help":
      echo Help
      quit(0)
    else:
      raise newException(ValueError, "Unknown option: --" & key)

proc compileShader(kind: GLenum; source: string): GLuint =
  ## Compile the fullscreen display shader and retain the driver's error message.
  result = glCreateShader(kind)
  var
    text = source.cstring
    status: GLint
  glShaderSource(result, 1, cast[cstringArray](addr text), nil)
  glCompileShader(result)
  glGetShaderiv(result, GL_COMPILE_STATUS, addr status)
  if status == 0:
    var message = newString(4096)
    glGetShaderInfoLog(result, GLsizei(message.len), nil, message.cstring)
    glDeleteShader(result)
    raise newException(ValueError, "Display shader failed: " & message)

proc newDisplayProgram(): GLuint =
  ## Link a triangle that samples the texture filled from our shared pixel buffer.
  let vertex = compileShader(GL_VERTEX_SHADER, VertexShader)
  defer: glDeleteShader(vertex)
  let fragment = compileShader(GL_FRAGMENT_SHADER, FragmentShader)
  defer: glDeleteShader(fragment)
  result = glCreateProgram()
  glAttachShader(result, vertex)
  glAttachShader(result, fragment)
  glLinkProgram(result)
  var status: GLint
  glGetProgramiv(result, GL_LINK_STATUS, addr status)
  if status == 0:
    var message = newString(4096)
    glGetProgramInfoLog(result, GLsizei(message.len), nil, message.cstring)
    glDeleteProgram(result)
    raise newException(ValueError, "Display program failed: " & message)

proc saveFrame(path: string; width, height: int) =
  ## Read back only on an explicit PNG request, including the displayed scaling.
  let image = newImage(width, height)
  glReadPixels(0, 0, GLsizei(width), GLsizei(height), GL_RGBA, GL_UNSIGNED_BYTE, addr image.data[0])
  image.flipVertical()
  image.writeFile(path)

proc main() =
  ## Run the book's three Chapter 8 views with explicit compute/graphics ownership.
  let
    options = readOptions()
    window = newWindow("Hippo — Chapter 8 OpenGL interop", ivec2(WindowSide, WindowSide))
  defer: window.close()
  window.makeContextCurrent()
  loadExtensions()
  let api = loadGL()
  when HippoRuntime == "HIP":
    discard hippoUseGLDevice()
  let
    stream = hippoStreamCreate()
    program = newDisplayProgram()
    first = hippoMallocPitch(RowBytes, Height)
    second = hippoMallocPitch(RowBytes, Height)
    sources = hippoMallocPitch(RowBytes, Height)
    sourceTexture = hippoCreateFloatTexture2D(sources.memory, Width, Height, sources.pitchBytes)
  defer: hippoStreamDestroy(stream)
  defer: glDeleteProgram(program)
  defer: hippoDestroyFloatTexture(sourceTexture)
  var
    buffer = newInteropBuffer(api, Width * Height * sizeof(Pixel))
    texture, vertexArray: GLuint
    input = first.memory
    output = second.memory
    inputPitch = first.pitchBytes
    outputPitch = second.pitchBytes
    inputTexture = hippoCreateFloatTexture2D(input, Width, Height, inputPitch)
    outputTexture = hippoCreateFloatTexture2D(output, Width, Height, outputPitch)
    initial = newSeq[cfloat](Width * Height)
    heaters = newSeq[cfloat](Width * Height)
    widthArg = cint(Width)
    heightArg = cint(Height)
    mode = options.mode
    tick, frames: int
    paused, resetRequested, saveRequested, capturedFinal: bool
  defer: buffer.close(stream)
  defer:
    hippoStreamSynchronize(stream)
    hippoDestroyFloatTexture(inputTexture)
    hippoDestroyFloatTexture(outputTexture)
  glGenTextures(1, addr texture)
  defer: glDeleteTextures(1, addr texture)
  glBindTexture(GL_TEXTURE_2D, texture)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8.GLint, Width, Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nil)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE.GLint)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE.GLint)
  glGenVertexArrays(1, addr vertexArray)
  defer: glDeleteVertexArrays(1, addr vertexArray)
  glBindVertexArray(vertexArray)
  glUseProgram(program)
  glUniform1i(glGetUniformLocation(program, "pixels"), 0)

  for y in 0 ..< Height:
    for x in 0 ..< Width:
      let index = y * Width + x
      initial[index] = 0.1'f32 + 0.15'f32 * cfloat(x) / cfloat(Width)
      if x in 18 .. 43 and y in 17 .. 43:
        heaters[index] = 1.0'f32
      elif (x - 90) * (x - 90) + (y - 36) * (y - 36) < 16 * 16:
        heaters[index] = 0.05'f32
      elif x in 65 .. 105 and y in 86 .. 101:
        heaters[index] = 0.8'f32
      if heaters[index] != 0.0'f32:
        initial[index] = heaters[index]

  proc reset() =
    ## Restore heat sources and the animation clock without reallocating GL storage.
    hippoMemcpy2D(input.p, inputPitch, addr initial[0], RowBytes,
      RowBytes, Height, HippoMemcpyHostToDevice)
    hippoMemcpy2D(output.p, outputPitch, addr initial[0], RowBytes,
      RowBytes, Height, HippoMemcpyHostToDevice)
    hippoMemcpy2D(sources.memory.p, sources.pitchBytes, addr heaters[0], RowBytes,
      RowBytes, Height, HippoMemcpyHostToDevice)
    tick = 0

  reset()
  let
    grid = newDim3((uint32(Width) + BlockSide - 1) div BlockSide,
      (uint32(Height) + BlockSide - 1) div BlockSide)
    blockSize = newDim3(BlockSide, BlockSide)
    backend = when HippoRuntime == "HIP": "HIP / native GL sharing"
              else: "HIP-CPU / host-mapped GL buffer"
  window.onButtonPress = proc(button: Button) =
    ## Switch among the book's examples and keep the manual demo easy to operate.
    case button
    of Key1: mode = BasicPattern
    of Key2: mode = Ripples
    of Key3: mode = HeatMap
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
      if mode == HeatMap:
        for iteration in 0 ..< HeatStepsPerFrame:
          var pitchArg = cint(inputPitch div sizeof(cfloat))
          hippoLaunchKernel(writeHeaters2D, gridDim = grid, blockDim = blockSize, stream = stream,
            args = hippoArgs(input.p, sourceTexture, widthArg, heightArg, pitchArg))
          pitchArg = cint(outputPitch div sizeof(cfloat))
          hippoLaunchKernel(diffuse2D, gridDim = grid, blockDim = blockSize, stream = stream,
            args = hippoArgs(output.p, inputTexture, widthArg, heightArg, pitchArg))
          swap(input, output)
          swap(inputTexture, outputTexture)
          swap(inputPitch, outputPitch)
      inc tick
    var
      tickArg = cint(tick)
      pitchArg = cint(inputPitch div sizeof(cfloat))
      mapped = buffer.map(stream)
    try:
      hippoLaunchKernel(renderPixels, gridDim = grid, blockDim = blockSize, stream = stream,
        args = hippoArgs(mapped, widthArg, heightArg, tickArg, mode, input.p, pitchArg))
    finally:
      buffer.unmap(stream)
    buffer.uploadTexture(texture, Width, Height)
    glViewport(0, 0, GLsizei(window.size.x), GLsizei(window.size.y))
    glDrawArrays(GL_TRIANGLES, 0, 3)
    if glGetError() != GL_NO_ERROR:
      raise newException(ValueError, "OpenGL rejected the shared-buffer display.")
    inc frames
    if saveRequested:
      saveFrame(DefaultScreenshot, int(window.size.x), int(window.size.y))
      saveRequested = false
    if options.frames > 0 and frames >= options.frames:
      if options.screenshot.len > 0:
        saveFrame(options.screenshot, int(window.size.x), int(window.size.y))
        capturedFinal = true
      break
    window.swapBuffers()
    window.title = "Hippo Chapter 8 | " & backend & " | " & ModeNames[int(mode)] &
      (if paused: " | PAUSED" else: "") & " | 1/2/3: view · Space: pause · R: reset · S: PNG · Esc: close"
    sleep(FrameDelayMs)
  if options.screenshot.len > 0 and not capturedFinal:
    glReadBuffer(GL_FRONT)
    saveFrame(options.screenshot, int(window.size.x), int(window.size.y))

when isMainModule:
  main()
