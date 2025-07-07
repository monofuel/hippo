import hippo, boxy, pixie, opengl, windy

const DIM = 1000


# Pretend that CuComplex is from another library and we are not allowed to change it
# exportcpp is not needed, but makes the transpiled code more readable
type CuComplex {.exportcpp.}= object
  r: cfloat
  i: cfloat

proc newCuComplex(a,b: float): CuComplex = # no pragmas
  return CuComplex(r: a, i: b)

proc magnitude2(this: CuComplex): cfloat = # no pragmas
  return this.r * this.r + this.i * this.i

proc multiply(a,b: CuComplex): CuComplex = # no pragmas
  return newCuComplex(a.r*b.r - a.i*b.i, a.i*b.r + a.r*b.i)

proc add(a,b: CuComplex): CuComplex = # no pragmas
  return newCuComplex(a.r+b.r, a.i+b.i)

# proc `*`(a,b: CuComplex): CuComplex =
#   return newCuComplex(a.r*b.r - a.i*b.i, a.i*b.r + a.r*b.i)

# proc `+`(a,b: CuComplex): CuComplex =
#   return newCuComplex(a.r+b.r, a.i+b.i)

proc julia(x,y: int): int = # no pragmas
  const scale: float = 1.5
  let jx: float = scale * (DIM/2.float - x.float) / (DIM/2.float)
  let jy: float = scale * (DIM/2.float - y.float) / (DIM/2.float)
  let c = newCuComplex(-0.8, 0.156)
  var a = newCuComplex(jx, jy)
  for i in 0..<200:
    # for 200 iterations, test if the function of a=a*a+c diverges
    #a = a * a + c
    a = multiply(a, a)
    a = add(a, c)
    if (a.magnitude2() > 1000):
      return 0
  # otherwise if the series converges, return 1 to say it is in the set
  return 1

proc juliaKernel(p: pointer) {.autoDeviceKernel, hippoGlobal.} =
  let x = blockIdx.x.int
  let y = blockIdx.y.int
  let offset = x + y * gridDim.x.int
  # autoDeviceKernel should automatically annotate julia and functions it calls as __host__ __device__
  let juliaValue = julia(x,y)

  let res = cast[ptr UncheckedArray[uint8]](p)
  res[offset*4 + 0] = 255 * juliaValue.uint8
  res[offset*4 + 1] = 0
  res[offset*4 + 2] = 0
  res[offset*4 + 3] = 255



proc displayUntilExit(image: Image) =
  let windowSize = ivec2(DIM, DIM)

  let window = newWindow("Julia", windowSize)
  makeContextCurrent(window)
  loadExtensions()
  let bxy = newBoxy()

  # Load our image
  bxy.addImage("bg", image)

  var frame: int
  proc display() =
    bxy.beginFrame(windowSize)
    bxy.drawImage("bg", rect = rect(vec2(0, 0), windowSize.vec2))
    bxy.endFrame()
    window.swapBuffers()
    inc frame

  while not window.closeRequested:
    display()
    pollEvents()

proc getSize*(image: Image): int =
  return image.width * image.height * 4 * sizeof(uint8)

proc main() =
  var image = newImage(DIM,DIM)

  let devBitmap = hippoMalloc(image.getSize())
  let grid = newDim3(DIM,DIM)
  hippoLaunchKernel(
    juliaKernel,
    gridDim=grid,
    args = (devBitmap,)
  )
  hippoMemcpy(addr image.data[0], devBitmap, image.getSize(), HippoMemcpyDeviceToHost)

  image.writeFile("julia.png")

  displayUntilExit(image)


when isMainModule:
  main()