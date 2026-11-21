import
  std/dynlib,
  hippo

type
  OpenGLUnavailableError* = object of CatchableError
  GLApi* = ref object
    library: LibHandle
    genBuffers*: proc(count: cint; buffers: ptr cuint) {.cdecl.}
    deleteBuffers*: proc(count: cint; buffers: ptr cuint) {.cdecl.}
    bindBuffer*: proc(target, buffer: cuint) {.cdecl.}
    bufferData*: proc(target: cuint; size: int; data: pointer; usage: cuint) {.cdecl.}
    getBufferSubData*: proc(target: cuint; offset, size: int; data: pointer) {.cdecl.}
    mapBufferRange*: proc(target: cuint; offset, size: int; access: cuint): pointer {.cdecl.}
    unmapBuffer*: proc(target: cuint): uint8 {.cdecl.}
    genTextures*: proc(count: cint; textures: ptr cuint) {.cdecl.}
    deleteTextures*: proc(count: cint; textures: ptr cuint) {.cdecl.}
    bindTexture*: proc(target, texture: cuint) {.cdecl.}
    texImage2D*: proc(target: cuint; level, internalFormat, width, height, border: cint;
      format, dataType: cuint; pixels: pointer) {.cdecl.}
    texSubImage2D*: proc(target: cuint; level, x, y, width, height: cint;
      format, dataType: cuint; pixels: pointer) {.cdecl.}
    getTexImage*: proc(target: cuint; level: cint; format, dataType: cuint;
      pixels: pointer) {.cdecl.}
    getString*: proc(name: cuint): cstring {.cdecl.}
    getError*: proc(): cuint {.cdecl.}
    finish*: proc() {.cdecl.}
  GLTestContext* = ref object
    api*: GLApi
    xlib: LibHandle
    display, context: pointer
    drawable: culong
  InteropBuffer* = object
    api: GLApi
    name*: cuint
    sizeBytes*: int
    mapped: bool
    writeDiscard: bool
    when HippoRuntime == "HIP":
      resource: HippoGraphicsResource

const
  PixelUnpackBuffer* = 0x88EC'u32
  Texture2D* = 0x0DE1'u32
  Rgba* = 0x1908'u32
  UnsignedByte* = 0x1401'u32
  Rgba8* = 0x8058'i32
  Renderer* = 0x1F01'u32
  DynamicDraw = 0x88E8'u32
  MapWriteBit = 0x0002'u32
  MapReadBit = 0x0001'u32
  MapInvalidateBufferBit = 0x0008'u32
  GlxRgbaType = 0x8014'i32
  GlxAttributes = [0x8012'i32, 1, 0x8010, 4, 0x8011, 1,
    8, 8, 9, 8, 10, 8, 11, 8, 0]
  PbufferAttributes = [0x8041'i32, 32, 0x8040, 32, 0]

proc loadGL*(): GLApi =
  ## Load only the OpenGL functions shared by the Chapter 8 test and demo.
  ## Tests need system libraries, without adding GUI packages to ordinary builds.
  result = GLApi(library: loadLib("libGL.so.1"))
  if result.library == nil:
    raise newException(OpenGLUnavailableError, "OpenGL's libGL.so.1 is unavailable.")
  template load(field, symbol: untyped) =
    result.field = cast[typeof(result.field)](symAddr(result.library, symbol))
    if result.field == nil:
      raise newException(OpenGLUnavailableError, "OpenGL function unavailable: " & symbol)
  load(genBuffers, "glGenBuffers")
  load(deleteBuffers, "glDeleteBuffers")
  load(bindBuffer, "glBindBuffer")
  load(bufferData, "glBufferData")
  load(getBufferSubData, "glGetBufferSubData")
  load(mapBufferRange, "glMapBufferRange")
  load(unmapBuffer, "glUnmapBuffer")
  load(genTextures, "glGenTextures")
  load(deleteTextures, "glDeleteTextures")
  load(bindTexture, "glBindTexture")
  load(texImage2D, "glTexImage2D")
  load(texSubImage2D, "glTexSubImage2D")
  load(getTexImage, "glGetTexImage")
  load(getString, "glGetString")
  load(getError, "glGetError")
  load(finish, "glFinish")

proc close*(context: GLTestContext) =
  ## Destroy the offscreen GLX drawable and context without touching desktop windows.
  if context != nil:
    if context.context != nil:
      let
        makeCurrent = cast[proc(d: pointer; draw, read: culong; ctx: pointer): cint {.cdecl.}](
          symAddr(context.api.library, "glXMakeContextCurrent"))
        destroyContext = cast[proc(d, ctx: pointer) {.cdecl.}](
          symAddr(context.api.library, "glXDestroyContext"))
      discard makeCurrent(context.display, 0, 0, nil)
      destroyContext(context.display, context.context)
      context.context = nil
    if context.drawable != 0:
      let destroyPbuffer = cast[proc(d: pointer; draw: culong) {.cdecl.}](
        symAddr(context.api.library, "glXDestroyPbuffer"))
      destroyPbuffer(context.display, context.drawable)
      context.drawable = 0
    if context.display != nil:
      let closeDisplay = cast[proc(d: pointer): cint {.cdecl.}](symAddr(context.xlib, "XCloseDisplay"))
      discard closeDisplay(context.display)
      context.display = nil

proc newTestContext*(): GLTestContext =
  ## Create an invisible GLX pbuffer on the current DISPLAY, using its actual GPU.
  ## No Windy dependency or visible window is needed for automated tests.
  result = GLTestContext(api: loadGL(), xlib: loadLib("libX11.so.6"))
  if result.xlib == nil:
    raise newException(OpenGLUnavailableError, "X11 is unavailable.")
  let
    openDisplay = cast[proc(name: cstring): pointer {.cdecl.}](symAddr(result.xlib, "XOpenDisplay"))
    defaultScreen = cast[proc(d: pointer): cint {.cdecl.}](symAddr(result.xlib, "XDefaultScreen"))
    freeX = cast[proc(data: pointer): cint {.cdecl.}](symAddr(result.xlib, "XFree"))
    chooseConfig = cast[proc(d: pointer; screen: cint; attrs: ptr cint;
      count: ptr cint): ptr UncheckedArray[pointer] {.cdecl.}](symAddr(result.api.library, "glXChooseFBConfig"))
    createContext = cast[proc(d, config: pointer; renderType: cint;
      share: pointer; direct: cint): pointer {.cdecl.}](symAddr(result.api.library, "glXCreateNewContext"))
    createPbuffer = cast[proc(d, config: pointer; attrs: ptr cint): culong {.cdecl.}](
      symAddr(result.api.library, "glXCreatePbuffer"))
    makeCurrent = cast[proc(d: pointer; draw, read: culong; ctx: pointer): cint {.cdecl.}](
      symAddr(result.api.library, "glXMakeContextCurrent"))
  result.display = openDisplay(nil)
  if result.display == nil:
    raise newException(OpenGLUnavailableError, "No accessible X11 DISPLAY for Chapter 8 interop.")
  try:
    var
      count: cint
      attributes = GlxAttributes
      pbufferAttributes = PbufferAttributes
    let configs = chooseConfig(result.display, defaultScreen(result.display), addr attributes[0], addr count)
    if configs == nil or count == 0:
      raise newException(OpenGLUnavailableError, "No GLX RGBA pbuffer configuration.")
    let config = configs[0]
    result.context = createContext(result.display, config, GlxRgbaType, nil, 1)
    result.drawable = createPbuffer(result.display, config, addr pbufferAttributes[0])
    discard freeX(configs)
    if result.context == nil or result.drawable == 0 or
        makeCurrent(result.display, result.drawable, result.drawable, result.context) == 0:
      raise newException(OpenGLUnavailableError, "Unable to make the GLX pbuffer current.")
  except:
    result.close()
    raise

proc newInteropBuffer*(api: GLApi; sizeBytes: int; writeDiscard = true;
                       initial: pointer = nil): InteropBuffer =
  ## Allocate a GL pixel buffer, then register it with HIP when using a GPU.
  if sizeBytes <= 0:
    raise newException(ValueError, "Pixel buffer size must be positive.")
  result = InteropBuffer(api: api, sizeBytes: sizeBytes, writeDiscard: writeDiscard)
  api.genBuffers(1, addr result.name)
  api.bindBuffer(PixelUnpackBuffer, result.name)
  api.bufferData(PixelUnpackBuffer, sizeBytes, initial, DynamicDraw)
  api.bindBuffer(PixelUnpackBuffer, 0)
  when HippoRuntime == "HIP":
    try:
      result.resource = hippoGraphicsGLRegisterBuffer(result.name, writeDiscard)
    except:
      api.deleteBuffers(1, addr result.name)
      raise

proc map*(buffer: var InteropBuffer; stream: HippoStream): pointer =
  ## Borrow the buffer until unmap; HIP-CPU uses a host-writable GL mapping.
  if buffer.mapped:
    raise newException(ValueError, "Pixel buffer is already mapped.")
  when HippoRuntime == "HIP":
    hippoGraphicsMap(buffer.resource, stream)
    buffer.mapped = true
    let view = hippoGraphicsMappedPointer(buffer.resource)
    if view.sizeBytes < buffer.sizeBytes:
      raise newException(ValueError, "Mapped graphics allocation is smaller than the pixel buffer.")
    result = view.data
  else:
    buffer.api.bindBuffer(PixelUnpackBuffer, buffer.name)
    let access = MapWriteBit or (if buffer.writeDiscard: MapInvalidateBufferBit else: MapReadBit)
    result = buffer.api.mapBufferRange(PixelUnpackBuffer, 0, buffer.sizeBytes, access)
    buffer.api.bindBuffer(PixelUnpackBuffer, 0)
    if result == nil:
      raise newException(ValueError, "OpenGL refused the host buffer mapping.")
    buffer.mapped = true

proc unmap*(buffer: var InteropBuffer; stream: HippoStream) =
  ## Complete compute work and return the buffer to OpenGL before drawing.
  if not buffer.mapped:
    raise newException(ValueError, "Pixel buffer is not mapped.")
  when HippoRuntime == "HIP":
    hippoGraphicsUnmap(buffer.resource, stream)
    hippoStreamSynchronize(stream)
  else:
    hippoStreamSynchronize(stream)
    buffer.api.bindBuffer(PixelUnpackBuffer, buffer.name)
    let valid = buffer.api.unmapBuffer(PixelUnpackBuffer)
    buffer.api.bindBuffer(PixelUnpackBuffer, 0)
    if valid == 0:
      raise newException(ValueError, "OpenGL reported corrupt mapped storage.")
  buffer.mapped = false

proc close*(buffer: var InteropBuffer; stream: HippoStream) =
  ## Unmap and unregister before deleting the GL buffer, with its context current.
  if buffer.name != 0:
    if buffer.mapped:
      buffer.unmap(stream)
    when HippoRuntime == "HIP":
      hippoGraphicsUnregister(buffer.resource)
    buffer.api.deleteBuffers(1, addr buffer.name)
    buffer.name = 0

proc uploadTexture*(buffer: InteropBuffer; texture: cuint; width, height: int) =
  ## Upload from the GL pixel buffer by byte offset, without a host image copy.
  if buffer.mapped:
    raise newException(ValueError, "OpenGL cannot draw a buffer while compute owns it.")
  buffer.api.bindTexture(Texture2D, texture)
  buffer.api.bindBuffer(PixelUnpackBuffer, buffer.name)
  buffer.api.texSubImage2D(Texture2D, 0, 0, 0, cint(width), cint(height), Rgba, UnsignedByte, nil)
  buffer.api.bindBuffer(PixelUnpackBuffer, 0)
