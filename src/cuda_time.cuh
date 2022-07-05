#ifndef CUDA_TIME_CUH
#define CUDA_TIME_CUH

#include <assert.h>
#include <stdio.h>

#include "cuda_try.cuh"
// macro for timing gpu operations

#define CUDA_TIME_FORCE_ENABLED(ce_start, ce_stop, stream, time, ...)                                                                                \
    do {                                                                                                                                             \
        CUDA_TRY(cudaStreamSynchronize((stream)));                                                                                                   \
        CUDA_TRY(cudaEventRecord((ce_start)));                                                                                                       \
        {                                                                                                                                            \
            __VA_ARGS__;                                                                                                                             \
        }                                                                                                                                            \
        CUDA_TRY(cudaEventRecord((ce_stop)));                                                                                                        \
        CUDA_TRY(cudaEventSynchronize((ce_stop)));                                                                                                   \
        CUDA_TRY(cudaEventElapsedTime((time), (ce_start), (ce_stop)));                                                                               \
    } while (0)

#ifdef DISABLE_CUDA_TIME

#define CUDA_TIME(ce_start, ce_stop, stream, time, ...)                                                                                              \
    do {                                                                                                                                             \
        CUDA_TRY(cudaStreamSynchronize((stream)));                                                                                                   \
        {                                                                                                                                            \
            __VA_ARGS__;                                                                                                                             \
        }                                                                                                                                            \
        *(time) = 0;                                                                                                                                 \
        UNUSED(ce_start);                                                                                                                            \
        UNUSED(ce_stop);                                                                                                                             \
    } while (0)

#else

#define CUDA_TIME(ce_start, ce_stop, stream, time, ...) CUDA_TIME_FORCE_ENABLED(ce_start, ce_stop, stream, time, __VA_ARGS__)

#endif

#endif
