import hippo

type Pixel* {.bycopy.} = object
  r*, g*, b*, a*: uint8

const
  BasicPattern* = 0'i32
  Ripples* = 1'i32
  HeatMap* = 2'i32
  BlockSide* = 8'u32

proc heatChannel(temperature, center: cfloat): uint8 {.hippoDeviceInline.} =
  ## Map one channel of the fixed temperature range using device math intrinsics.
  let
    t = hippoFminf(hippoFmaxf(temperature, 0.0'f32), 1.0'f32)
    intensity = hippoFminf(hippoFmaxf(1.5'f32 - hippoFabsf(4.0'f32 * t - center),
      0.0'f32), 1.0'f32)
  uint8(intensity * 255.0'f32)

proc renderPixels*(pixels: ptr Pixel; width, height, tick, mode: cint;
                   temperatures: ptr cfloat; pitch: cint) {.hippoGlobal.} =
  ## Chapter 8: write RGBA pixels directly into the currently mapped GL buffer.
  let
    x = int(blockIdx.x * blockDim.x + threadIdx.x)
    y = int(blockIdx.y * blockDim.y + threadIdx.y)
  if x < int(width) and y < int(height):
    let
      values = cast[ptr UncheckedArray[Pixel]](pixels)
      index = y * int(width) + x
    values[index].a = 255
    if mode == BasicPattern:
      let
        fx = cfloat(x) / cfloat(width) - 0.5'f32
        fy = cfloat(y) / cfloat(height) - 0.5'f32
        green = 128.0'f32 + 127.0'f32 * hippoSin(hippoFabsf(fx * 100.0'f32) - hippoFabsf(fy * 100.0'f32))
      values[index].r = 0
      values[index].g = uint8(green)
      values[index].b = 0
    elif mode == Ripples:
      let
        dx = cfloat(x - int(width) div 2)
        dy = cfloat(y - int(height) div 2)
        distance = hippoSqrt(dx * dx + dy * dy) / 10.0'f32
        phase = distance - cfloat(tick) / 7.0'f32
        grey = uint8(128.0'f32 + 127.0'f32 * hippoCos(phase) / (distance + 1.0'f32))
      values[index].r = grey
      values[index].g = grey
      values[index].b = grey
    else:
      let temperature = cast[ptr UncheckedArray[cfloat]](temperatures)[y * int(pitch) + x]
      values[index].r = heatChannel(temperature, 3.0'f32)
      values[index].g = heatChannel(temperature, 2.0'f32)
      values[index].b = heatChannel(temperature, 1.0'f32)

proc modifyPixels*(pixels: ptr Pixel; count: cint) {.hippoGlobal.} =
  ## Exercise read/write registration by changing old values and retaining endpoints.
  let index = int(blockIdx.x * blockDim.x + threadIdx.x)
  if index > 0 and index < int(count) - 1:
    let values = cast[ptr UncheckedArray[Pixel]](pixels)
    values[index].r = values[index].r xor 0x5A'u8
    values[index].g = values[index].g xor 0xA5'u8
