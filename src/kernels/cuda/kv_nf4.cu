// src/kernels/cuda/kv_nf4.cu - see include/strata/kernels/kv_nf4.hpp. NVFP4 KV with the q4_0 option's
// Walsh-Hadamard rotation (kv_q4.cu's fwht256, reused through its public launcher).
#include "strata/kernels/kv_nf4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_nf4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// One head-cell's 16 groups in 16 threads - one group per thread end to end: the group's amax, its scale and
// its 8 packed bytes need nothing from any other lane, so no reduction consensus (q4's warp-wide tie-break)
// exists to get wrong. `pool` points at the row's single block_nf4.
__device__ __forceinline__ void nf4_group_store(uint8_t* pool, long long row, int g, const float* x16) {
    block_nf4* blk = reinterpret_cast<block_nf4*>(pool + row * (long long) sizeof(block_nf4));
    uint8_t qs[NF4_GROUP / 2];
    nf4_quantize_group16(x16, blk->sc[g], qs);
#pragma unroll
    for (int i = 0; i < NF4_GROUP / 2; ++i) blk->qs[g * (NF4_GROUP / 2) + i] = qs[i];
}

// The verify window's K/V cells: grid (heads, steps * planes), 16 threads. Mirrors kv_append_q4_kernel.
__global__ void kv_append_nf4_kernel(uint8_t* __restrict__ k_nf4, uint8_t* __restrict__ v_nf4,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                     int step_stride, int planes, const float* __restrict__ kcur,
                                     const float* __restrict__ vcur, int kv_heads, int head_dim, int page_size,
                                     KvHostPools host) {
    const int step_idx = (int) (blockIdx.y / (unsigned) planes);
    const bool is_v = (blockIdx.y % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, g = threadIdx.x;
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    float x[NF4_GROUP];
    const float* src = (is_v ? vcur : kcur) + step_off + h * head_dim + g * NF4_GROUP;
#pragma unroll
    for (int i = 0; i < NF4_GROUP; ++i) x[i] = src[i];
    const long long page = (long long) table[pos / page_size];
    const long long host_row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    // the pools hold SLOT rows (a ring or a streamed ring table maps the page), the host copy logical ones
    if (page >= 0)
        nf4_group_store(is_v ? v_nf4 : k_nf4, (page * kv_heads + h) * page_size + (pos % page_size), g, x);
    if (host.k_nf4 != nullptr) nf4_group_store(is_v ? host.v_nf4 : host.k_nf4, host_row, g, x);
}

// The prompt path: grid (T, kv_heads * 2); also into the staging pool (identity layout) when given.
__global__ void kv_append_nf4_batch_kernel(uint8_t* __restrict__ k_nf4, uint8_t* __restrict__ v_nf4,
                                           const int32_t* __restrict__ table, int64_t pos0, const float* K,
                                           const float* V, int kv_heads, int head_dim, int page_size,
                                           KvHostPools host, KvHostPools stage) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y >> 1, g = threadIdx.x;
    const bool is_v = (blockIdx.y & 1) != 0;
    float x[NF4_GROUP];
    const float* src = (is_v ? V : K) + t * (kv_heads * head_dim) + h * head_dim + g * NF4_GROUP;
#pragma unroll
    for (int i = 0; i < NF4_GROUP; ++i) x[i] = src[i];
    const long long page = (long long) table[pos / page_size];
    const long long row = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) nf4_group_store(is_v ? v_nf4 : k_nf4, (page * kv_heads + h) * page_size + (pos % page_size), g, x);
    if (host.k_nf4 != nullptr) nf4_group_store(is_v ? host.v_nf4 : host.k_nf4, row, g, x);
    if (stage.k_nf4 != nullptr) nf4_group_store(is_v ? stage.v_nf4 : stage.k_nf4, row, g, x);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths): one block per (id, head).
__global__ void kv_gather_nf4_kernel(const uint8_t* __restrict__ k_nf4, const uint8_t* __restrict__ v_nf4,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                     const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                     uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= n_ids * kv_heads) return;
    const int g = threadIdx.x;
    const long long id = blk_idx / kv_heads;
    const int h = (int) (blk_idx % kv_heads);
    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = ((page * kv_heads) + h) * page_size + (cell % page_size);
    const block_nf4* kb = reinterpret_cast<const block_nf4*>(k_nf4 + row * (long long) sizeof(block_nf4));
    const block_nf4* vb = reinterpret_cast<const block_nf4*>(v_nf4 + row * (long long) sizeof(block_nf4));
    const float ks = nf4_e4m3_decode(kb->sc[g]);
    const float vs = nf4_e4m3_decode(vb->sc[g]);
    const long long dst = ((id * kv_heads + h) * head_dim) + g * NF4_GROUP;
#pragma unroll
    for (int i = 0; i < NF4_GROUP / 2; ++i) {   // element 2i: byte i's low nibble; 2i+1: its high nibble
        const uint8_t kbyte = kb->qs[g * (NF4_GROUP / 2) + i];
        const uint8_t vbyte = vb->qs[g * (NF4_GROUP / 2) + i];
        k_scratch[dst + 2 * i] = f16_from_f32(nf4_e2m1_decode(kbyte & 0x0F) * ks);
        k_scratch[dst + 2 * i + 1] = f16_from_f32(nf4_e2m1_decode(kbyte >> 4) * ks);
        v_scratch[dst + 2 * i] = f16_from_f32(nf4_e2m1_decode(vbyte & 0x0F) * vs);
        v_scratch[dst + 2 * i + 1] = f16_from_f32(nf4_e2m1_decode(vbyte >> 4) * vs);
    }
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void kv_append_nf4_steps(uint8_t* k_nf4, uint8_t* v_nf4, const int32_t* page_table, const int32_t* step,
                         int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                         void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_nf4");
    const int planes = (v_nf4 == nullptr || (v_nf4 == k_nf4 && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (n_steps * planes));
    kv_append_nf4_kernel<<<grid, NF4_GROUP, 0, (cudaStream_t) stream>>>(
        k_nf4, v_nf4, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_nf4 launch");
}

void kv_append_nf4_step(uint8_t* k_nf4, uint8_t* v_nf4, const int32_t* page_table, const int32_t* step,
                        const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                        const KvHostPools* host) {
    kv_append_nf4_steps(k_nf4, v_nf4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_nf4(uint8_t* k_nf4, uint8_t* v_nf4, const int32_t* page_table, int64_t pos0, int64_t T,
                   const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                   const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_nf4");
    const dim3 grid((unsigned) T, (unsigned) (s.n_head_kv * 2));
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    kv_append_nf4_batch_kernel<<<grid, NF4_GROUP, 0, (cudaStream_t) stream>>>(
        k_nf4, v_nf4, page_table, pos0, K, V, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, h, st);
    check("kv_append_nf4 batch launch");
}

void kv_gather_nf4_step(const uint8_t* k_nf4, const uint8_t* v_nf4, const int32_t* page_table,
                        const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                        uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int64_t total = max_ids * s.n_head_kv;
    const int rows_per_block = 8;
    const unsigned num_blocks = (unsigned) ((total + rows_per_block - 1) / rows_per_block);
    kv_gather_nf4_kernel<<<dim3(num_blocks), dim3(NF4_GROUP, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_nf4, v_nf4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_nf4 launch");
}

}  // namespace strata::kernels
