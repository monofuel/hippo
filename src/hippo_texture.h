#ifndef HIPPO_TEXTURE_H
#define HIPPO_TEXTURE_H

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/hip_runtime.h>
#define HIPPO_RUNTIME(name) hip##name
#else
#include <cuda_runtime.h>
#define HIPPO_RUNTIME(name) cuda##name
#endif

namespace hippo_detail {

// Keep runtime resource-descriptor unions and channel formats on the C++ side.
inline HIPPO_RUNTIME(Error_t) create_float_texture_1d(HIPPO_RUNTIME(TextureObject_t)* texture,
                                         void* data, size_t count) {
    HIPPO_RUNTIME(ResourceDesc) resource{};
    resource.resType = HIPPO_RUNTIME(ResourceTypeLinear);
    resource.res.linear.devPtr = data;
    resource.res.linear.desc = HIPPO_RUNTIME(CreateChannelDesc)<float>();
    resource.res.linear.sizeInBytes = count * sizeof(float);

    HIPPO_RUNTIME(TextureDesc) sampler{};
    sampler.readMode = HIPPO_RUNTIME(ReadModeElementType);
    return HIPPO_RUNTIME(CreateTextureObject)(texture, &resource, &sampler, nullptr);
}

inline HIPPO_RUNTIME(Error_t) create_float_texture_2d(HIPPO_RUNTIME(TextureObject_t)* texture,
                                         void* data, size_t width,
                                         size_t height, size_t pitch_bytes) {
    HIPPO_RUNTIME(ResourceDesc) resource{};
    resource.resType = HIPPO_RUNTIME(ResourceTypePitch2D);
    resource.res.pitch2D.devPtr = data;
    resource.res.pitch2D.desc = HIPPO_RUNTIME(CreateChannelDesc)<float>();
    resource.res.pitch2D.width = width;
    resource.res.pitch2D.height = height;
    resource.res.pitch2D.pitchInBytes = pitch_bytes;

    HIPPO_RUNTIME(TextureDesc) sampler{};
    sampler.addressMode[0] = HIPPO_RUNTIME(AddressModeClamp);
    sampler.addressMode[1] = HIPPO_RUNTIME(AddressModeClamp);
    sampler.filterMode = HIPPO_RUNTIME(FilterModePoint);
    sampler.readMode = HIPPO_RUNTIME(ReadModeElementType);
    sampler.normalizedCoords = 0;
    return HIPPO_RUNTIME(CreateTextureObject)(texture, &resource, &sampler, nullptr);
}

} // namespace hippo_detail

#undef HIPPO_RUNTIME

#endif
