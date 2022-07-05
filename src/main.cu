#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>

#define THREADS_PER_WARP 32

template <size_t ELEMENT_COUNT, size_t WORST_SIZE> __device__ void decompress_sub_blocks(size_t data_size, uint32_t* data_c, uint32_t* data_d)
{
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        data_d[i] = data_c[i];
    }
}

template <size_t ELEMENT_COUNT, size_t WORST_SIZE> __device__ size_t compress_sub_blocks(uint32_t* data_d, uint32_t* data_c)
{
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        data_c[i] = data_d[i];
    }
    return sizeof(uint32_t) * ELEMENT_COUNT;
}

template <int WARP_COUNT, size_t ELEMENT_COUNT, size_t WORST_SIZE> __global__ void kernel_initial_compress(size_t N, uint32_t* data)
{
    int warpidx = threadIdx.x / 32;
    int warpoffset = threadIdx.x % 32;
    int warpbase = THREADS_PER_WARP * warpidx;
    __shared__ uint16_t size_table[THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint8_t s_mem_d[sizeof(uint32_t) * ELEMENT_COUNT * THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint8_t s_mem_c[WORST_SIZE * THREADS_PER_WARP * WARP_COUNT];
    size_t block_idx = blockIdx.x * WARP_COUNT + warpidx;
    size_t block_gridstride = gridDim.x * WARP_COUNT;
    for (; block_idx < N; block_idx += block_gridstride) {
        char* table_position = (char*)data + ((WORST_SIZE + sizeof(uint16_t)) * THREADS_PER_WARP * block_idx);
        size_table[warpbase + warpoffset] = ((uint16_t*)table_position)[warpoffset];
        char* data_position = (char*)table_position + sizeof(uint16_t) * THREADS_PER_WARP;
        __syncwarp();
        for (size_t sub_idx = 0; sub_idx < THREADS_PER_WARP; sub_idx++) {
            size_t readers_per_subblock = THREADS_PER_WARP / ELEMENT_COUNT;
            for (size_t i = 0; i < ELEMENT_COUNT; i++) {
                ((uint32_t*)s_mem_d)[i * THREADS_PER_WARP + warpidx]
            }

            for (size_t i = (warpbase + sub_idx) * ELEMENT_COUNT; sub_pos < (size_table[warpbase + sub_idx] / sizeof(uint32_t)) + 1;
                 sub_pos += THREADS_PER_WARP) {
                ((uint32_t*)s_mem_d)[+sub_pos] = ((uint32_t*)data_position)[(warpbase + sub_idx) * ELEMENT_COUNT + sub_pos];
            }
        }
    }
}

int main()
{
}
