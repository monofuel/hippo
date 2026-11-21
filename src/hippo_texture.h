#ifndef HIPPO_TEXTURE_H
#define HIPPO_TEXTURE_H

#include <hip/hip_runtime.h>

namespace hippo_detail {

// Keep HIP's resource-descriptor union and channel format on the C++ side.
inline hipError_t create_float_texture_1d(hipTextureObject_t* texture,
                                         void* data, size_t count) {
    hipResourceDesc resource{};
    resource.resType = hipResourceTypeLinear;
    resource.res.linear.devPtr = data;
    resource.res.linear.desc = hipCreateChannelDesc<float>();
    resource.res.linear.sizeInBytes = count * sizeof(float);

    hipTextureDesc sampler{};
    sampler.readMode = hipReadModeElementType;
    return hipCreateTextureObject(texture, &resource, &sampler, nullptr);
}

inline hipError_t create_float_texture_2d(hipTextureObject_t* texture,
                                         void* data, size_t width,
                                         size_t height, size_t pitch_bytes) {
    hipResourceDesc resource{};
    resource.resType = hipResourceTypePitch2D;
    resource.res.pitch2D.devPtr = data;
    resource.res.pitch2D.desc = hipCreateChannelDesc<float>();
    resource.res.pitch2D.width = width;
    resource.res.pitch2D.height = height;
    resource.res.pitch2D.pitchInBytes = pitch_bytes;

    hipTextureDesc sampler{};
    sampler.addressMode[0] = hipAddressModeClamp;
    sampler.addressMode[1] = hipAddressModeClamp;
    sampler.filterMode = hipFilterModePoint;
    sampler.readMode = hipReadModeElementType;
    sampler.normalizedCoords = 0;
    return hipCreateTextureObject(texture, &resource, &sampler, nullptr);
}

} // namespace hippo_detail

#endif
