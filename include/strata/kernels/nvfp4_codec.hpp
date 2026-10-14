// include/strata/kernels/nvfp4_codec.hpp - the NVFP4 element pair: 4-bit E2M1 codes with one 8-bit E4M3 scale
// per group of 16 (kv_nf4.hpp's `--kv nvfp4`).
//
// Pure integer/float bit math, no intrinsics, so the SAME functions compile into the device kernels and the
// host reference of the parity test - every write path (decode, verify windows, prefill, the drafter) then
// quantizes bit for bit identically, which is what the verify window's bitwise-equality contract needs.
//
// E2M1 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}; code = sign<<3 | index, round-to-nearest with ties to the
// EVEN code (IEEE roundTiesToEven on the 15-point grid). -0.0f encodes as +0: the code grid has a -0, but no
// writer distinguishes one and every reader's -0 * s is -0 * s.
//
// E4M3 follows e4m3fn (OCP): no infinities, the max finite is 448 (0x7E), 0x7F is NaN, subnormals reach down
// to 2^-9. Encoding is RNE with SATURATION to +-448 (the scale of a group whose amax/6 overflows must stay
// finite - the codes still carry the magnitude's sign and most of its range). Scales are always >= 0 here, so
// the sign bit of an encoded scale is never set.
#pragma once

#include <cmath>
#include <cstdint>

namespace strata::kernels {

/// Host and device compile the same codec lines (the parity test pins the kernels to these bits); a plain
/// header cannot say __host__ __device__ to a host-only compiler, so it says nothing there.
#if defined(__CUDACC__)
#define NF4_HD __host__ __device__
#else
#define NF4_HD
#endif

inline NF4_HD float nf4_e2m1_decode(uint8_t code) {
    constexpr float v[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};   // function-local: device-visible
    const float m = v[code & 7u];
    return (code & 8u) ? -m : m;
}

/// Nearest E2M1 code of `x`, ties to the even code index.
inline NF4_HD uint8_t nf4_e2m1_encode(float x) {
    const float a = fabsf(x);
    int i = a >= 5.0f   ? 7
            : a >= 3.5f ? 6
            : a >= 2.5f ? 5
            : a >= 1.75f ? 4
            : a >= 1.25f ? 3
            : a >= 0.75f ? 2
            : a >= 0.25f ? 1
                         : 0;
    // exact midpoints: roundTiesToEven picks the even neighbour (0.75 -> code 2, 1.25 -> code 2, ...)
    if (a == 0.25f) i = 0;
    else if (a == 0.75f) i = 2;
    else if (a == 1.25f) i = 2;
    else if (a == 1.75f) i = 4;
    else if (a == 2.5f) i = 4;
    else if (a == 3.5f) i = 6;
    else if (a == 5.0f) i = 6;
    return (uint8_t) ((x < 0.0f ? 8 : 0) | i);
}

/// e4m3fn bits -> float. 0x7F is NaN; every other pattern is finite.
inline NF4_HD float nf4_e4m3_decode(uint8_t b) {
    if (b == 0x7F) {
        union { uint32_t u; float f; } nan_bits{0x7FC00000u};
        return nan_bits.f;
    }
    const float sign = (b & 0x80u) ? -1.0f : 1.0f;
    const int exp = (b >> 3) & 0xF;
    const int man = b & 7;
    if (exp == 0) return sign * (float) man * (1.0f / 512.0f);              // man * 2^-9
    // (1 + man/8) * 2^(exp-7): both factors are exact powers-of-two fractions, so the product is exact
    float v = sign * (1.0f + (float) man * 0.125f);
    for (int k = exp - 7; k > 0; --k) v *= 2.0f;
    for (int k = 7 - exp; k > 0; --k) v *= 0.5f;
    return v;
}

/// float -> e4m3fn bits, round-to-nearest-even, saturating (magnitude past 448 encodes 448).
inline NF4_HD uint8_t nf4_e4m3_encode(float x) {
    float a = fabsf(x);
    const uint8_t sign = x < 0.0f ? 0x80 : 0x00;
    if (a != a) return 0x7F;                                   // NaN
    if (a <= 1.0f / 1024.0f) return sign;                      // below half the min subnormal (2^-10): +-0
    if (a >= 464.0f) return sign | 0x7E;                       // past 448's rounding midpoint: saturate
    if (a < 0.015625f) {                                       // subnormal range [2^-10, 2^-6): man = RNE(a * 512)
        const float q = a * 512.0f;
        int man = (int) q;                                     // q in (1, 8); floor first
        const float frac = q - (float) man;
        if (frac > 0.5f || (frac == 0.5f && (man & 1))) ++man;
        return (uint8_t) (sign | (man == 8 ? 0x08 : man));     // man 8 rounds up into the min normal
    }
    int e = 0;                                                 // a in [2^-6, 464): find the binade
    float lo = 0.015625f;                                      // lo = 2^(e-6), so the exp FIELD is e + 1
    while (a >= lo * 2.0f) { lo *= 2.0f; ++e; }                // a in [lo, 2*lo)
    float q = (a / lo - 1.0f) * 8.0f;                          // mantissa in [0, 8)
    int man = (int) q;
    const float frac = q - (float) man;
    if (frac > 0.5f || (frac == 0.5f && (man & 1))) ++man;
    int exp = e + 1;
    if (man == 8) { man = 0; ++exp; }
    return (uint8_t) (sign | ((uint8_t) exp << 3) | (uint8_t) man);
}

/// IEEE round-to-nearest division, the one op in the quantizer that -use_fast_math would otherwise turn
/// approximate on the device (kv_q4.cu's scales divide by powers of two and never noticed).
inline NF4_HD float nf4_div_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

/// One group of 16 floats -> its scale byte and 8 packed code bytes, in NInfer's exact order
/// (src/ops/kv_cache/nvfp4_group16_codec.cuh): amax / 6 clamped to [2^-9, 448] before the E4M3 RNE encode,
/// then every element DIVIDED by the decoded scale (not multiplied by a reciprocal) before its E2M1 RNE code.
/// Host and device compile the same lines, so the parity test pins the kernels to these bits.
inline NF4_HD void nf4_quantize_group16(const float* x, uint8_t& scale_out, uint8_t* qs8) {
    float amax = 0.0f;
#if defined(__CUDACC__)
#pragma unroll
#endif
    for (int i = 0; i < 16; ++i) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {   // an all-zero group: zero scale, zero codes (NInfer returns the zeroed struct)
        scale_out = 0;
#if defined(__CUDACC__)
#pragma unroll
#endif
        for (int j = 0; j < 8; ++j) qs8[j] = 0;
        return;
    }
    const float raw = nf4_div_rn(amax, 6.0f);
    const float bounded = fminf(448.0f, fmaxf(1.0f / 512.0f, raw));
    scale_out = nf4_e4m3_encode(bounded);
    const float s = nf4_e4m3_decode(scale_out);
#if defined(__CUDACC__)
#pragma unroll
#endif
    for (int j = 0; j < 8; ++j)
        qs8[j] = (uint8_t) (nf4_e2m1_encode(nf4_div_rn(x[j], s)) |
                            (nf4_e2m1_encode(nf4_div_rn(x[j + 8], s)) << 4));
}

/// One group's 16 floats back, exactly the decode side of the same file (E2M1 value * E4M3 scale - a product
/// exact in FP32, as NInfer proves it is in FP16).
inline NF4_HD void nf4_dequant_group16(const uint8_t* qs8, uint8_t scale, float* out16) {
    const float s = nf4_e4m3_decode(scale);
#if defined(__CUDACC__)
#pragma unroll
#endif
    for (int j = 0; j < 8; ++j) {
        out16[j] = nf4_e2m1_decode(qs8[j] & 0x0F) * s;
        out16[j + 8] = nf4_e2m1_decode(qs8[j] >> 4) * s;
    }
}

}  // namespace strata::kernels
