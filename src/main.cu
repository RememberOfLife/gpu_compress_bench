#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>

#define THREAD_PER_WARP 32

template <size_t ELEMENT_COUNT> __device__ void decompress_sub_blocks(size_t data_size, uint64_t* data)
{
    uint64_t* data_r = data + ELEMENT_COUNT;
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        data_r[-i] = data[i];
    }
}

// data points to start of sub block, data is stored in reverse
template <size_t ELEMENT_COUNT> __device__ size_t compress_sub_blocks(uint64_t* data_in, uint64_t* data_out)
{
    uint64_t* data_r = data + ELEMENT_COUNT;
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        data[i] = data_r[-i];
    }
    return sizeof(uint64_t) * ELEMENT_COUNT;
}

int main()
{
}
