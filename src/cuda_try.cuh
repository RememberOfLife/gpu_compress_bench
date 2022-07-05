#ifndef CUDA_TRY_CUH
#define CUDA_TRY_CUH

#include <assert.h>
#include <stdio.h>

// macro for cuda return value checking
#define CUDA_TRY(expr)                                                                                                                               \
    do {                                                                                                                                             \
        cudaError_t _err_ = (expr);                                                                                                                  \
        if (_err_ != cudaSuccess) {                                                                                                                  \
            report_cuda_error(_err_, #expr, __FILE__, __LINE__, true);                                                                               \
        }                                                                                                                                            \
    } while (0)

static void report_cuda_error(cudaError_t err, const char* cmd, const char* file, int line, bool die)
{
    printf("CUDA Error at %s:%i%s%s\n%s\n", file, line, cmd ? ": " : "", cmd ? cmd : "", cudaGetErrorString(err));
    assert(false);
    if (die) exit(EXIT_FAILURE);
}

#endif
