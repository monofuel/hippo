import hippo

when HippoRuntime notin ["HIP", "HIP_CPU", "CUDA"]:
  {.error: "Chapter 7 heat diffusion requires HIP, CUDA or HIP-CPU.".}

const HeatSpeed = 0.25'f32

proc writeHeaters2D*(dst: ptr cfloat; heaters: HippoFloatTexture;
                     width, height, pitch: cint) {.hippoGlobal.} =
  ## Chapter 7's 2D variant: restore heater cells sampled at pixel centers.
  let
    x = int(blockIdx.x * blockDim.x + threadIdx.x)
    y = int(blockIdx.y * blockDim.y + threadIdx.y)
  if x < int(width) and y < int(height):
    let value = hippoTex2D(heaters, cfloat(x) + 0.5'f32, cfloat(y) + 0.5'f32)
    if value != 0.0'f32:
      cast[ptr UncheckedArray[cfloat]](dst)[y * int(pitch) + x] = value

proc diffuse2D*(dst: ptr cfloat; src: HippoFloatTexture;
                width, height, pitch: cint) {.hippoGlobal.} =
  ## Diffuse into a separate buffer and let the 2D sampler clamp edge coordinates.
  let
    x = cfloat(blockIdx.x * blockDim.x + threadIdx.x) + 0.5'f32
    y = cfloat(blockIdx.y * blockDim.y + threadIdx.y) + 0.5'f32
  if x < cfloat(width) and y < cfloat(height):
    let
      center = hippoTex2D(src, x, y)
      left = hippoTex2D(src, x - 1.0'f32, y)
      right = hippoTex2D(src, x + 1.0'f32, y)
      top = hippoTex2D(src, x, y - 1.0'f32)
      bottom = hippoTex2D(src, x, y + 1.0'f32)
    cast[ptr UncheckedArray[cfloat]](dst)[int(y) * int(pitch) + int(x)] =
      center + HeatSpeed * (top + bottom + left + right - 4.0'f32 * center)
