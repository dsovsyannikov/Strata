// src/kernels/gdn_state.cpp - see include/strata/kernels/gdn_state.hpp.
#include "strata/kernels/gdn_state.hpp"

#include <atomic>
#include <cstring>

namespace strata::kernels {
namespace {

std::atomic<bool> g_f16{false};

/// F32 -> binary16 bits, round-to-nearest-even, NaN and infinity preserved: the value `__float2half`
/// on the device produces. Written out because the conversation cache converts whole checkpoints on
/// the host, where <cuda_fp16.h> is not available.
uint16_t f32_to_f16_bits(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = (uint16_t) ((x >> 16) & 0x8000u);
    const uint32_t biased = (x >> 23) & 0xFFu;
    uint32_t man = x & 0x7FFFFFu;
    if (biased == 0xFFu) {                       // infinity / NaN
        uint16_t m = (uint16_t) (man >> 13);
        if (man != 0 && m == 0) m = 1;           // a NaN whose payload shifted away stays a NaN
        return (uint16_t) (sign | 0x7C00u | m);
    }
    int32_t exp = (int32_t) biased - 127 + 15;
    if (exp >= 0x1F) return (uint16_t) (sign | 0x7C00u);   // overflow rounds to infinity
    if (exp <= 0) {
        if (exp < -10) return sign;              // and any nonzero mantissa is below half the smallest subnormal
        man |= 0x800000u;                        // the implicit one
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t sub = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (sub & 1u))) ++sub;
        return (uint16_t) (sign | sub);
    }
    uint32_t sub = man >> 13;
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (sub & 1u))) {
        ++sub;
        if (sub == 0x400u) { sub = 0; ++exp; }   // the mantissa rounded up into the next exponent
        if (exp >= 0x1F) return (uint16_t) (sign | 0x7C00u);
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10) | sub);
}

/// binary16 bits -> F32: exact, no rounding anywhere.
float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (man == 0) bits = sign;               // signed zero
        else {                                    // a subnormal: normalize it into the F32 exponent
            // man = m >> s with bit 10 of m set, so the value is 1.(m & 0x3FF) * 2^(-14 - s)
            uint32_t m = man;
            int32_t s = 0;
            while (!(m & 0x400u)) { m <<= 1; ++s; }
            bits = sign | ((uint32_t) (127 - 14 - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);  // infinity / NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

void gdn_state_set_f16(bool on) { g_f16.store(on, std::memory_order_relaxed); }
bool gdn_state_f16() { return g_f16.load(std::memory_order_relaxed); }

uint64_t gdn_state_esize() { return g_f16.load(std::memory_order_relaxed) ? 2 : 4; }

uint64_t gdn_state_rec_bytes(int64_t state_size, int64_t v_heads) {
    return (uint64_t) state_size * (uint64_t) v_heads * (uint64_t) state_size * gdn_state_esize();
}

uint64_t gdn_state_layer_bytes(int64_t state_size, int64_t v_heads, int64_t conv_channels, int64_t d_conv) {
    return gdn_state_rec_bytes(state_size, v_heads) +
           (uint64_t) conv_channels * (uint64_t) (d_conv - 1) * 4;
}

void gdn_state_convert_arena(void* dst, bool dst_f16, const void* src, size_t rows, size_t rec, size_t conv_floats) {
    if (dst == nullptr || src == nullptr || rows == 0) return;
    const bool to_f16 = dst_f16;
    uint8_t* d = (uint8_t*) dst;
    const uint8_t* s = (const uint8_t*) src;
    const size_t conv_bytes = conv_floats * 4;
    for (size_t r = 0; r < rows; ++r) {
        if (to_f16) {
            uint16_t* dr = (uint16_t*) d;
            const float* sr = (const float*) s;
            for (size_t i = 0; i < rec; ++i) dr[i] = f32_to_f16_bits(sr[i]);
        } else {
            float* dr = (float*) d;
            const uint16_t* sr = (const uint16_t*) s;
            for (size_t i = 0; i < rec; ++i) dr[i] = f16_bits_to_f32(sr[i]);
        }
        d += rec * (to_f16 ? 2 : 4);
        s += rec * (to_f16 ? 4 : 2);
        std::memcpy(d, s, conv_bytes);
        d += conv_bytes;
        s += conv_bytes;
    }
}

}  // namespace strata::kernels
