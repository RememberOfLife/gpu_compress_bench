#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>
#include "fast_prng.cuh"

#define THREADS_PER_WARP 32
#define TABLE_SIZE ((size_t)THREADS_PER_WARP * sizeof(uint16_t))

struct warp_stats {
    int idx;
    int offset;
    int base;
};

struct block_stats {
    size_t idx;
    size_t size_c;
    size_t size_d;
    size_t count;
    size_t gridstride;
    size_t final_block_elem_count;
};

__device__ warp_stats get_warp_stats()
{
    warp_stats w;
    w.idx = threadIdx.x / THREADS_PER_WARP;
    w.offset = threadIdx.x % THREADS_PER_WARP;
    w.base = THREADS_PER_WARP * w.idx;
    return w;
}

constexpr size_t sb_max_elems_c(size_t sb_elem_count)
{
    return sb_elem_count + (sb_elem_count + 7) / 8;
}

__device__ block_stats get_block_stats(size_t uncompressed_data_size, size_t elem_count)
{
    block_stats b;
    int warp_count = (blockDim.x / THREADS_PER_WARP);
    int warp_idx = (threadIdx.x / THREADS_PER_WARP);
    b.size_c = sb_max_elems_c(elem_count) * THREADS_PER_WARP + TABLE_SIZE * sizeof(uint32_t);
    b.size_d = elem_count * THREADS_PER_WARP * sizeof(uint32_t);
    b.count = (uncompressed_data_size + b.size_d - 1) / b.size_d;
    b.gridstride = gridDim.x * warp_count;
    b.idx = blockIdx.x * (blockDim.x / THREADS_PER_WARP) + warp_idx;
    b.final_block_elem_count = (uncompressed_data_size - (uncompressed_data_size / b.size_d * b.size_d)) / sizeof(uint32_t);
    return b;
}

template <typename T> void bit_print(T data, bool spacing = true)
{
    size_t typewidth_m1 = sizeof(T) * 8 - 1;
    for (int i = typewidth_m1; i >= 0; i--) {
        printf("%c", (data >> i) & 0b1 ? '1' : '0');
        if (spacing && i < typewidth_m1 && i > 0 && i % 8 == 0) {
            printf(" ");
        }
    }
}

// TODO integrate with compress_sub_blocks
template <size_t SB_ELEM_COUNT> __host__ __device__ size_t compress_block(uint32_t* data_d, uint8_t* data_c)
{
    size_t in_count = SB_ELEM_COUNT;
    size_t in_idx = 0;
    size_t out_byte_idx = 0;
    while (in_idx < in_count) {
        uint64_t working = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                // splice both u32 together bytewise
                working |= (((uint64_t)data_d[in_idx] >> (j * 8)) & 0xFF) << ((j * 16) + (i * 8));
            }
            in_idx++;
        }
        // printf("WC:");
        // bit_print(working);
        // printf("\n");
        for (int i = 0; i < 9; i++) {
            // get rightmost 7 bits of working, mark as continued if working is non zero
            uint8_t acc;
            if (i == 8) {
                // if we need all 8 bytes, dont encode the continuation bit in the last one
                acc = (working & 0b11111111);
                working >>= 8;
            }
            else {
                acc = (working & 0b01111111);
                working >>= 7;
            }
            if (working) {
                acc |= 0b10000000;
            }
            data_c[out_byte_idx++] = acc;
            if (!working) {
                break;
            }
        }
    }
    return out_byte_idx;
}

// TODO integrate with decompress_sub_blocks
template <size_t SB_ELEM_COUNT> __host__ __device__ void decompress_block(size_t data_size, uint8_t* data_c, uint32_t* data_d)
{
    size_t in_byte_idx = 0;
    size_t out_idx = 0;
    while (in_byte_idx < data_size) {
        uint64_t acc = 0;
        // accumulate bytes until one is marked as uncontinued
        uint8_t shift = 0;
        while (true) {
            assert(shift < 9);
            uint64_t working;
            if (shift == 8) {
                working = data_c[in_byte_idx++];
                acc |= (working << (shift++ * 7));
            }
            else {
                working = data_c[in_byte_idx++];
                acc |= ((working & 0b01111111) << (shift++ * 7));
            }
            if (!(working & 0b10000000) || shift == 9) {
                break;
            }
        }
        // printf("WD:");
        // bit_print(acc);
        // printf("\n");
        for (int i = 0; i < 2; i++) {
            uint32_t out_el = 0;
            for (int j = 0; j < 4; j++) {
                // unsplice into two u32 elements
                out_el |= ((acc >> ((j * 16) + (i * 8))) & 0xFF) << (j * 8);
            }
            data_d[out_idx++] = out_el;
        }
    }
}

template <size_t SB_ELEM_COUNT> __device__ void decompress_sub_blocks(uint32_t* data_c, uint16_t* size_table, uint32_t* data_d)
{
    uint8_t* begin_c = (uint8_t*)&data_c[threadIdx.x * sb_max_elems_c(SB_ELEM_COUNT)];
    uint8_t* end_c = ((uint8_t*)data_c) + size_table[threadIdx.x];
    uint32_t* begin_d = &data_d[threadIdx.x * sb_max_elems_c(SB_ELEM_COUNT)];
    decompress_block<SB_ELEM_COUNT>(end_c - begin_c, begin_c, begin_d);
}

template <size_t SB_ELEM_COUNT> __device__ size_t compress_sub_blocks(uint32_t* data_d, uint16_t* size_table, uint32_t* data_c)
{
    uint8_t* begin_c = (uint8_t*)&data_c[threadIdx.x * SB_ELEM_COUNT];
    uint32_t* begin_d = &data_d[threadIdx.x * sb_max_elems_c(SB_ELEM_COUNT)];
    return compress_block<SB_ELEM_COUNT>(begin_d, begin_c);
}

template <size_t SB_ELEM_COUNT> __device__ void build_size_table(uint16_t* size_table, uint16_t my_sb_size)
{
    warp_stats w = get_warp_stats();
    size_table[threadIdx.x] = my_sb_size;
    __syncwarp();

    for (int i = THREADS_PER_WARP / 2; i >= 0; i /= 2) {
        if (w.offset <= i) {
            size_table[w.base + i] += size_table[threadIdx.x + i];
        }
        __syncwarp();
    }
}

template <size_t SB_ELEM_COUNT, uint32_t (*OPERATION)(uint32_t)>
__device__ void apply_operation(block_stats* b, uint32_t* data_d, uint16_t* size_table)
{
    size_t sb_elem_count = SB_ELEM_COUNT;
    size_t sb_offset = threadIdx.x * sb_elem_count;
    if (b->idx + 1 == b->count) {
        if (sb_offset > b->final_block_elem_count) {
            sb_elem_count = 0;
        }
        else if (sb_offset + sb_elem_count > b->final_block_elem_count) {
            sb_elem_count = b->final_block_elem_count - sb_offset;
        }
    }

    uint32_t* sb_d_begin = (uint32_t*)&data_d[sb_offset];
    uint32_t* sb_d_end = (uint32_t*)&data_d[sb_offset + sb_elem_count];

    for (auto i = sb_d_begin; i != sb_d_end; i++) {
        *i = OPERATION(*i);
    }
}

template <size_t SB_ELEM_COUNT> __device__ void write_out_compressed_data(uint32_t* data_c, uint16_t* size_table, char* table_begin)
{
    warp_stats w = get_warp_stats();

    size_table[threadIdx.x] = ((uint16_t*)table_begin)[w.offset];
    __syncwarp();

    uint32_t* data_tgt = (uint32_t*)(table_begin + TABLE_SIZE);
    size_t data_offset = w.offset * sizeof(uint32_t);
    int subblock_idx = 0;
    for (int i = 0; i < SB_ELEM_COUNT; i++) {
        while (size_table[w.base + subblock_idx] > data_offset) {
            subblock_idx++;
        }
        size_t sb_offset = data_offset - size_table[w.base + subblock_idx];
        size_t data_c_offset = subblock_idx * sb_max_elems_c(SB_ELEM_COUNT) + sb_offset;
        data_tgt[w.idx + i * SB_ELEM_COUNT] = data_c[data_c_offset / sizeof(uint32_t)];
        data_offset += THREADS_PER_WARP * sizeof(uint32_t);
    }
}

template <size_t SB_ELEM_COUNT> __device__ void load_in_compressed_data(char* table_begin, uint16_t* size_table, uint32_t* data_c)
{
    warp_stats w = get_warp_stats();

    size_table[threadIdx.x] = ((uint16_t*)table_begin)[w.offset];
    __syncwarp();
    size_t data_offset = w.offset * sizeof(uint32_t);
    int subblock_idx = 0;
    uint32_t* data_src = (uint32_t*)(table_begin + TABLE_SIZE);
    for (int i = 0; i < SB_ELEM_COUNT; i++) {
        while (size_table[w.base + subblock_idx] > data_offset) {
            subblock_idx++;
        }
        size_t sb_offset = data_offset - size_table[w.base + subblock_idx];
        size_t data_c_offset = subblock_idx * sb_max_elems_c(SB_ELEM_COUNT) + sb_offset;
        data_c[data_c_offset / sizeof(uint32_t)] = data_src[w.idx + i * SB_ELEM_COUNT];
        data_offset += THREADS_PER_WARP * sizeof(uint32_t);
    }
}

template <size_t SB_ELEM_COUNT> __device__ void load_in_uncompressed_data(uint32_t* data_src, uint32_t* data_d)
{
    warp_stats w = get_warp_stats();

    for (int i = w.offset; i < SB_ELEM_COUNT * THREADS_PER_WARP; i += THREADS_PER_WARP) {
        data_d[w.base + i] = data_src[i];
    }
}

template <size_t SB_ELEM_COUNT> __device__ void write_out_uncompressed_data(block_stats* b, uint32_t* data_d, uint32_t* data_tgt)
{
    warp_stats w = get_warp_stats();
    size_t elem_count = SB_ELEM_COUNT * THREADS_PER_WARP;
    if (b->idx + 1 == b->count) {
        elem_count = b->final_block_elem_count;
    }

    for (int i = w.offset; i < elem_count; i += THREADS_PER_WARP) {
        data_tgt[i] = data_d[w.base + i];
    }
}

template <int WARP_COUNT, size_t SB_ELEM_COUNT, uint32_t (*OPERATION)(uint32_t)>
__global__ void kernel_apply_unary_op_on_compressed(size_t uncompressed_data_size, uint32_t* data)
{
    __shared__ uint16_t size_table[THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_d[SB_ELEM_COUNT * THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_c[sb_max_elems_c(SB_ELEM_COUNT) * THREADS_PER_WARP * WARP_COUNT];

    block_stats b = get_block_stats(uncompressed_data_size, SB_ELEM_COUNT);

    for (; b.idx < b.count; b.idx += b.gridstride) {
        char* table_position = (char*)data + b.idx * b.size_c;

        load_in_compressed_data<SB_ELEM_COUNT>(table_position, size_table, &s_mem_c);

        decompress_sub_blocks<SB_ELEM_COUNT>(&s_mem_c, &s_mem_d);

        apply_operation<SB_ELEM_COUNT, OPERATION>(&b, &s_mem_d, size_table);

        size_t sb_size = compress_sub_blocks<SB_ELEM_COUNT>(&s_mem_d, &s_mem_c);

        build_size_table<SB_ELEM_COUNT>(size_table, sb_size);

        write_out_compressed_data<SB_ELEM_COUNT>(&s_mem_c, size_table, table_position);
    }
}

template <int WARP_COUNT, size_t SB_ELEM_COUNT, uint32_t (*OPERATION)(uint32_t)>
__global__ void kernel_inital_compress(size_t uncompressed_data_size, uint32_t* initial_data, uint32_t* data_c)
{
    __shared__ uint16_t size_table[THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_d[SB_ELEM_COUNT * THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_c[sb_max_elems_c(SB_ELEM_COUNT) * THREADS_PER_WARP * WARP_COUNT];

    block_stats b = get_block_stats(uncompressed_data_size, SB_ELEM_COUNT);

    for (; b.idx < b.count; b.idx += b.gridstride) {
        char* initial_data_pos = (char*)data_c + b.idx * b.size_d;
        load_in_uncompressed_data<SB_ELEM_COUNT>((uint32_t*)initial_data_pos, s_mem_d);

        size_t sb_size = compress_sub_blocks<SB_ELEM_COUNT>(s_mem_d, size_table, s_mem_c);
        build_size_table<SB_ELEM_COUNT>(size_table, sb_size);

        char* data_c_pos = (char*)data_c + b.idx * b.size_c;
        write_out_compressed_data<SB_ELEM_COUNT>(s_mem_c, size_table, data_c_pos);
    }
}

template <int WARP_COUNT, size_t SB_ELEM_COUNT, uint32_t (*OPERATION)(uint32_t)>
__global__ void kernel_final_decompress(size_t uncompressed_data_size, uint32_t* data_c, uint32_t* data_d)
{
    __shared__ uint16_t size_table[THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_d[SB_ELEM_COUNT * THREADS_PER_WARP * WARP_COUNT];
    __shared__ uint32_t s_mem_c[sb_max_elems_c(SB_ELEM_COUNT) * THREADS_PER_WARP * WARP_COUNT];

    block_stats b = get_block_stats(uncompressed_data_size, SB_ELEM_COUNT);

    for (; b.idx < b.count; b.idx += b.gridstride) {
        char* table_position = (char*)data_c + b.idx * b.size_c;
        load_in_compressed_data<SB_ELEM_COUNT>(table_position, size_table, &s_mem_c);

        decompress_sub_blocks<SB_ELEM_COUNT>(&s_mem_c, &s_mem_d);

        char* data_d_pos = (char*)data_d + b.idx * b.size_d;
        write_out_uncompressed_data<SB_ELEM_COUNT>(&s_mem_d, size_table, table_position);
    }
}

uint32_t inc(uint32_t x)
{
    return x + 1;
}

int main()
{
    const size_t elements = 1 << 20;
    size_t data_size = elements * sizeof(uint32_t);
    size_t worst_size = data_size + data_size / 8;
    uint32_t* in = (uint32_t*)malloc(data_size);
    uint32_t* res = (uint32_t*)malloc(data_size);
    uint8_t* out = (uint8_t*)malloc(worst_size);
    fast_prng rng(42);
    for (size_t i = 0; i < elements; i++) {
        uint32_t a = rng.rand();
        uint32_t b = rng.rand();
        uint32_t c = rng.rand();
        in[i] = a % ((b >> (31 - (c & 0b11111))) + 1);
    }
    size_t compressed_size = compress_block<elements>(in, out);
    printf("compressed %zu data bytes into %zu / %zu\n", data_size, compressed_size, worst_size);
    printf("ratio: %.3f\n", (float)data_size / (float)compressed_size);
    decompress_block<elements>(compressed_size, out, res);
    for (size_t i = 0; i < elements; i++) {
        if (in[i] != res[i]) {
            printf("FAIL @ %zu : I %u != O %u\n", i, in[i], res[i]);
            exit(1);
        }
    }
    printf("PASS %zu\n", elements);
    free(in);
    free(out);
    free(res);
    return 0;

    kernel_inital_compress<1, 4, &inc><<<1, 1>>>(1, NULL, NULL);
}
