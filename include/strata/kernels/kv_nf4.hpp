// include/strata/kernels/kv_nf4.hpp - NVFP4 KV storage with Walsh-Hadamard rotation for the QSA layers
// (`--kv nvfp4`).
//
// K and V are rotated by kv_q4.hpp's orthonormal 256-point Hadamard H (spread over all 256 dimensions, what
// makes 4 bits usable - the q4_0 option's own measured trick), then stored as NVFP4: 4-bit E2M1 codes with one
// E4M3 scale per GROUP OF 16 (nvfp4_codec.hpp). 144 B per head per cell - the same bytes as q4_0 and 45% under
// INT8 - but the scale granularity is twice as fine (16 vs 32) and both code and scale are floats with a wide
// per-group range, the format NInfer found best for KV at 4 bits (its K8V4 uses it for V with the same
// rotation). The query is rotated the same way (<Hq, Hk> = <q, k>), the output rotated back; the loss is the
// 4-bit rounding alone.
//
// One group is one THREAD: the 16 amaxes, the scale encode and the 8 packed bytes need no cross-lane
// agreement, so every writer (decode step, verify window, prompt batch, the drafter) is bit for bit the same
// code in the same order - the parity test pins the host reference to exactly these bits.
#pragma once

#include "strata/kernels/nvfp4_codec.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

inline constexpr int NF4_GROUP = 16;

#pragma pack(push, 1)
struct block_nf4 {
    uint8_t qs[128];   // 16 groups of 16 codes: group g's element j (j < 8) in the low nibble of qs[g*8+j],
                       // element j+8 in the high nibble
    uint8_t sc[16];    // one E4M3 scale per group (nvfp4_codec.hpp; never negative, never set to 0x7F)
};
#pragma pack(pop)

static_assert(sizeof(block_nf4) == 144, "block_nf4 must be 144 bytes (128 code + 16 scale)");

/// Bytes per cell and KV head: head_dim 256 = 16 groups -> 144.
inline uint64_t kv_nf4_bytes_per_head(int head_dim) { return (uint64_t) (head_dim / NF4_GROUP) * 9; }

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_nf4_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_nf4_bytes_per_head((int) s.head_dim) * 2;
}

/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there
/// too, and to VRAM only if its block is resident - q4_0's contract unchanged.
void kv_append_nf4_step(uint8_t* k_nf4, uint8_t* v_nf4, const int32_t* page_table, const int32_t* step,
                        const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                        const KvHostPools* host = nullptr);

void kv_append_nf4_steps(uint8_t* k_nf4, uint8_t* v_nf4, const int32_t* page_table, const int32_t* step,
                         int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                         void* stream, const KvHostPools* host = nullptr);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage`
/// (identity layout) when given.
void kv_append_nf4(uint8_t* k_nf4, uint8_t* v_nf4, const int32_t* page_table, int64_t pos0, int64_t T,
                   const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                   const KvHostPools* stage);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_nf4_step(const uint8_t* k_nf4, const uint8_t* v_nf4, const int32_t* page_table,
                        const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                        uint16_t* k_scratch, uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
