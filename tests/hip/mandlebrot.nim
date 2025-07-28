import hippo, boxy, pixie, opengl, windy

## Manual Mandelbrot set demo similar to tests/hip/julia.nim.
##
## This file mirrors the original CUDA/HIP Julia sample but instead renders
## the Mandelbrot set. Each pixel maps to a point `c` on the complex plane
## and we iterate the classic equation `z_{n+1} = z_n^2 + c`, starting with
## `z_0 = 0`. Pixels that remain bounded after the maximum iteration count
## are considered part of the Mandelbrot set and rendered in white. All other
## pixels diverge and are coloured black. The resulting image is displayed in
## a window and saved to `mandlebrot.png`.

const DIM = 1000

type CuComplex {.exportcpp.} = object
  r: cfloat
  i: cfloat

# Complex helpers -----------------------------------------------------------

proc newCuComplex(a, b: float): CuComplex {.hippoDevice.} =
  CuComplex(r: a, i: b)

proc magnitude2(this: CuComplex): cfloat {.hippoDevice.} =
  this.r * this.r + this.i * this.i

proc multiply(a, b: CuComplex): CuComplex {.hippoDevice.} =
  newCuComplex(a.r * b.r - a.i * b.i, a.i * b.r + a.r * b.i)

proc add(a, b: CuComplex): CuComplex {.hippoDevice.} =
  newCuComplex(a.r + b.r, a.i + b.i)

# Mandelbrot evaluation ------------------------------------------------------

proc mandelbrot(cx, cy: float): uint8 {.hippoDevice.} =
  ## Evaluate whether the point (cx, cy) is in the Mandelbrot set.
  var z = newCuComplex(0, 0)
  let c = newCuComplex(cx, cy)

  for n in 0 ..< 200:
    z = multiply(z, z)
    z = add(z, c)
    if z.magnitude2() > 1000:      # Diverged.
      return 0u8
  1u8                                # In the set.

proc mandelbrotPixel(x, y: int): uint8 {.hippoDevice.} =
  ## Map pixel coordinate to complex plane and evaluate membership.
  const scale: float = 1.5
  let jx: float = scale * (DIM / 2.0 - x.float) / (DIM / 2.0)
  let jy: float = scale * (DIM / 2.0 - y.float) / (DIM / 2.0)
  mandelbrot(jx, jy)

# Kernel --------------------------------------------------------------------

proc mandelbrotKernel(p: pointer) {.hippoGlobal.} =
  let x = blockIdx.x.int
  let y = blockIdx.y.int
  let offset = x + y * gridDim.x.int

  let value = mandelbrotPixel(x, y)

  let res = cast[ptr UncheckedArray[uint8]](p)
  res[offset * 4 + 0] = 255u8 * value
  res[offset * 4 + 1] = 0u8
  res[offset * 4 + 2] = 0u8
  res[offset * 4 + 3] = 255u8

# Utility -------------------------------------------------------------------

proc displayUntilExit(image: Image) =
  let windowSize = ivec2(DIM, DIM)
  let window = newWindow("Mandelbrot", windowSize)
  makeContextCurrent(window)
  loadExtensions()
  let bxy = newBoxy()

  bxy.addImage("bg", image)

  proc display() =
    bxy.beginFrame(windowSize)
    bxy.drawImage("bg", rect = rect(vec2(0, 0), windowSize.vec2))
    bxy.endFrame()
    window.swapBuffers()

  while not window.closeRequested:
    display()
    pollEvents()

proc getSize*(image: Image): int =
  image.width * image.height * 4 * sizeof(uint8)

# Main ----------------------------------------------------------------------

proc main() =
  var image = newImage(DIM, DIM)

  let devBitmap = hippoMalloc(image.getSize())
  let grid = newDim3(DIM, DIM)

  hippoLaunchKernel(
    mandelbrotKernel,
    gridDim = grid,
    args = hippoArgs(devBitmap)
  )

  hippoMemcpy(addr image.data[0], devBitmap, image.getSize(), HippoMemcpyDeviceToHost)

  image.writeFile("mandlebrot.png")

  displayUntilExit(image)

when isMainModule:
  main()

