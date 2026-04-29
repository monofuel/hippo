#ifndef HIPPO_WMMA_H
#define HIPPO_WMMA_H

#ifdef __HIPCC__
typedef _Float16 wmma_half16 __attribute__((ext_vector_type(16)));
typedef float wmma_float8 __attribute__((ext_vector_type(8)));
#endif

#endif
