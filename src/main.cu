#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>

#define THREADS_PER_WARP 32

/*
        __syncwarp();
        size_table[i] += size_table[threadIdx.x];
        __syncwarp();

 for (int i = THREADS_PER_WARP / 2; i >= 0; i /= 2) {
            if (warp_offset <= i) {
                size_table[i] += size_table[threadIdx.x + i];
            }
            __syncwarp();
        }

        */

template <size_t ELEMENT_COUNT, size_t WORST_SIZE> __device__ size_t compress_sub_blocks(uint32_t* data_d, uint16_t* size_table, uint32_t* data_c)
{
    uint32_t* begin = &data_c[threadIdx.x * WORST_SIZE];
    uint32_t* end = &data_c[size_table[threadIdx.x]];
    size_t pos_c = 0;
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        data_c[pos_c++] = data_d[i];
    }
    return sizeof(uint32_t) * pos_c;
}

template <size_t ELEMENT_COUNT, size_t WORST_SIZE> __device__ void decompress_sub_blocks(uint8_t* data_c, uint16_t* size_table, uint32_t* data_d)
{
    uint32_t* begin = &data_d[threadIdx.x * WORST_SIZE];
    uint32_t* end = &data_d[size_table[threadIdx.x]];
    size_t pos_d = 0;
    for (uint32_t* c = begin; c != end; c++) {
        data_d[pos_d++] = *c;
    }
}

template <size_t ELEMENT_COUNT, size_t WORST_SIZE> __device__ void build_size_table(uint16_t* size_table, uint16_t my_sb_size)
{
    int warp_base = THREADS_PER_WARP * (threadIdx.x / THREADS_PER_WARP);
    int warp_offset = threadIdx.x % THREADS_PER_WARP;
    size_table[threadIdx.x] = my_sb_size;
    __syncwarp();

    for (int i = THREADS_PER_WARP / 2; i >= 0; i /= 2) {
        if (warp_offset <= i) {
            size_table[warp_base + i] += size_table[threadIdx.x + i];
        }
        __syncwarp();
    }
}

template <size_t ELEMENT_COUNT, size_t WORST_SIZE, uint32_t (*OPERATION)(uint32_t)>
__device__ void apply_operation(uint32_t* data_d, uint16_t* size_table, size_t block_idx, size_t total_data_size)
{
    int warp_idx = threadIdx.x / THREADS_PER_WARP;
    int warp_offset = threadIdx.x % THREADS_PER_WARP;
    int warp_base = THREADS_PER_WARP * warp_idx;
    size_t block_size = WORST_SIZE * THREADS_PER_WARP;
    size_t final_block_size = total_data_size - (total_data_size / block_size * block_size);

    size_t my_block_size = block_size;
    if (block_idx * block_size > total_data_size) {
        my_block_size = final_block_size;
    }
    size_t sb_d_offset = threadIdx.x * ELEMENT_COUNT * sizeof(uint32_t);
    uint32_t* sb_d_begin = (uint32_t*)&data_d[sb_d_offset];
    uint32_t* sb_d_end = (uint32_t*)&data_d[sb_d_offset + ELEMENT_COUNT * sizeof(uint32_t)];

    if (size_table[threadIdx.x] > my_block_size) {
        sb_d_end = &data_d[my_block_size];
    }
    for (auto i = sb_d_begin; i != sb_d_end; i++) {
        *i = OPERATION(*i);
    }
}

template <size_t ELEMENT_COUNT, size_t WORST_SIZE>
__device__ void write_out_compressed_data(uint32_t* data_c, uint16_t* size_table, char* table_begin)
{
    int warp_idx = threadIdx.x / THREADS_PER_WARP;
    int warp_offset = threadIdx.x % THREADS_PER_WARP;
    int warp_base = THREADS_PER_WARP * warp_idx;
    size_t table_size = sizeof(uint16_t*) * THREADS_PER_WARP;

    size_table[threadIdx.x] = ((uint16_t*)table_begin)[warp_offset];
    __syncwarp();

    uint32_t* data_tgt = (uint32_t*)(table_begin + table_size);
    size_t data_offset = warp_offset * sizeof(uint32_t);
    int subblock_idx = 0;
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        while (size_table[warp_base + subblock_idx] > data_offset) {
            subblock_idx++;
        }
        size_t sb_offset = data_offset - size_table[warp_base + subblock_idx];
        size_t data_c_offset = subblock_idx * WORST_SIZE + sb_offset;
        data_tgt[warp_idx + i * ELEMENT_COUNT] = data_c[data_c_offset / sizeof(uint32_t)];
        data_offset += THREADS_PER_WARP * sizeof(uint32_t);
    }
}

template <size_t ELEMENT_COUNT, size_t WORST_SIZE> __device__ void load_in_compressed_data(char* table_begin, uint16_t* size_table, uint32_t* data_c)
{
    int warp_idx = threadIdx.x / THREADS_PER_WARP;
    int warp_offset = threadIdx.x % THREADS_PER_WARP;
    int warp_base = THREADS_PER_WARP * warp_idx;
    size_t table_size = sizeof(uint16_t*) * THREADS_PER_WARP;

    size_table[threadIdx.x] = ((uint16_t*)table_begin)[warp_offset];
    __syncwarp();
    size_t data_offset = warp_offset * sizeof(uint32_t);
    int subblock_idx = 0;
    uint32_t* data_src = (uint32_t*)(table_begin + table_size);
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        while (size_table[warp_base + subblock_idx] > data_offset) {
            subblock_idx++;
        }
        size_t sb_offset = data_offset - size_table[warp_base + subblock_idx];
        size_t data_c_offset = subblock_idx * WORST_SIZE + sb_offset;
        data_c[data_c_offset / sizeof(uint32_t)] = data_src[warp_idx + i * ELEMENT_COUNT];
        data_offset += THREADS_PER_WARP * sizeof(uint32_t);
    }
}

template <size_t ELEMENT_COUNT, size_t WORST_SIZE> __device__ void load_in_uncompressed_data(uint32_t* data, uint32_t* data_d)
{
    int warp_idx = threadIdx.x / THREADS_PER_WARP;
    int warp_offset = threadIdx.x % THREADS_PER_WARP;
    int warp_base = THREADS_PER_WARP * warp_idx;

    for (int i = warp_offset; i < ELEMENT_COUNT * THREADS_PER_WARP; i += THREADS_PER_WARP) {
        data_d[warp_base + i] = data[i];
    }
}

template <int WARP_COUNT, size_t ELEMENT_COUNT, size_t WORST_SIZE, uint32_t (*OPERATION)(uint32_t)>
__global__ void kernel_apply_unary_op_on_compressed(size_t data_size, uint32_t* data)
{
    __shared__ uint16_t size_table[THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_d[ELEMENT_COUNT * THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_c[WORST_SIZE * THREADS_PER_WARP * WARP_COUNT / sizeof(uint32_t)];

    int warp_idx = threadIdx.x / THREADS_PER_WARP;

    size_t table_size = THREADS_PER_WARP * sizeof(uint16_t);

    size_t block_gridstride = gridDim.x * WARP_COUNT;
    size_t block_size = WORST_SIZE * THREADS_PER_WARP;
    size_t block_count = (data_size + block_size - 1) / block_size;

    size_t block_idx = blockIdx.x * WARP_COUNT + warp_idx;

    for (; block_idx < block_count; block_idx += block_gridstride) {
        char* table_position = (char*)data + block_idx * (table_size + block_size);

        load_in_compressed_data<ELEMENT_COUNT, WORST_SIZE>(table_position, size_table, &s_mem_c);

        decompress_sub_blocks<ELEMENT_COUNT, WORST_SIZE>(&s_mem_c, &s_mem_d);

        apply_operation<ELEMENT_COUNT, WORST_SIZE, OPERATION>(&s_mem_d, size_table, data_size);

        size_t sb_size = compress_sub_blocks<ELEMENT_COUNT, WORST_SIZE>(&s_mem_d, &s_mem_c);

        build_size_table<ELEMENT_COUNT, WORST_SIZE>(size_table, sb_size);

        write_out_compressed_data<ELEMENT_COUNT, WORST_SIZE>(&s_mem_c, size_table, table_position);
    }
}

template <int WARP_COUNT, size_t ELEMENT_COUNT, size_t WORST_SIZE, uint32_t (*OPERATION)(uint32_t)>
__global__ void kernel_inital_compress(size_t data_size, uint32_t* initial_data, uint32_t* data_c)
{
    __shared__ uint16_t size_table[THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_d[ELEMENT_COUNT * THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_c[WORST_SIZE * THREADS_PER_WARP * WARP_COUNT / sizeof(uint32_t)];

    int warp_idx = threadIdx.x / THREADS_PER_WARP;
    int warp_offset = threadIdx.x % THREADS_PER_WARP;

    size_t table_size = THREADS_PER_WARP * sizeof(uint16_t);

    size_t block_gridstride = gridDim.x * WARP_COUNT;
    size_t block_size = WORST_SIZE * THREADS_PER_WARP;
    size_t block_count = (data_size + block_size - 1) / block_size;

    size_t block_idx = blockIdx.x * WARP_COUNT + warp_idx;

    for (; block_idx < block_count; block_idx += block_gridstride) {
        char* initial_data_pos = (char*)data_c + block_idx * (ELEMENT_COUNT * sizeof(uint32_t));
        load_in_uncompressed_data<ELEMENT_COUNT, WORST_SIZE>((uint32_t*)initial_data_pos, s_mem_d);

        size_t sb_size = compress_sub_blocks<ELEMENT_COUNT, WORST_SIZE>(s_mem_d, size_table, s_mem_c);

        build_size_table<ELEMENT_COUNT, WORST_SIZE>(size_table, sb_size);

        char* data_c_pos = (char*)data_c + block_idx * (table_size + block_size);
        write_out_compressed_data<ELEMENT_COUNT, WORST_SIZE>(s_mem_c, size_table, data_c_pos);
    }
}

uint32_t inc(uint32_t x)
{
    return x + 1;
}

int main()
{
    kernel_inital_compress<1, 4, 16, &inc><<<1, 1>>>(1, NULL, NULL);
}
