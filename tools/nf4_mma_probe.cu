// One-warp microtest of mma.m16n8k64.mxf4nvf4 against a CPU reference, on EXACT fp4-representable values
// (scales are powers of two), so any wiring error in A/B/SFA/SFB shows as an exact, diagnosable pattern.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

__device__ __forceinline__ uint8_t e4m3_encode_pow2(int e) {   // 2^e for e in [-6, 8] -> e4m3 bits
    if (e < -6) return 0;                                       // subnormal/zero: not exact, avoid
    return (uint8_t) ((e + 7) << 3);                            // man 0
}
__device__ __forceinline__ float e4m3_decode_pow2(uint8_t b) {
    const int e = (b >> 3) - 7;
    float v = 1.0f;
    for (int i = 0; i < e; ++i) v *= 2.0f;
    for (int i = 0; i < -e; ++i) v *= 0.5f;
    return v;
}
__device__ __forceinline__ uint8_t e2m1_code(float x) {   // x ON the grid {0,.5,1,1.5,2,3,4,6}
    const float a = fabsf(x);
    int i = a >= 5.0f ? 7 : a >= 3.5f ? 6 : a >= 2.5f ? 5 : a >= 1.75f ? 4 : a >= 1.25f ? 3 : a >= 0.75f ? 2
            : a >= 0.25f ? 1 : 0;
    return (uint8_t) ((x < 0.0f ? 8 : 0) | i);
}
__device__ __forceinline__ float e2m1_val(uint8_t c) {
    constexpr float v[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}

// host-side twins
static uint8_t h_e4m3_pow2(int e) { return e < -6 ? 0 : (uint8_t) ((e + 7) << 3); }
static float h_e4m3_pow2v(uint8_t b) { const int e = (b >> 3) - 7; return std::ldexp(1.0f, e); }
static float h_e2m1(uint8_t c) {
    constexpr float v[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}

__global__ void probe(const uint8_t* __restrict__ q_codes, const uint8_t* __restrict__ q_sc,
                      const uint8_t* __restrict__ k_codes, const uint8_t* __restrict__ k_sc,
                      const uint8_t* __restrict__ lo_codes, const uint8_t* __restrict__ lo_sc,
                      float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    // A: 16 rows x 64 dims (codes [row][32B], scales [row][4B]); B: 8 cols x 64 dims the same way
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(q_codes + (gid)*32 + 4 * tig);
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(q_codes + (gid + 8)*32 + 4 * tig);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(q_codes + (gid)*32 + 16 + 4 * tig);
    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(q_codes + (gid + 8)*32 + 16 + 4 * tig);
    const uint32_t l0 = *reinterpret_cast<const uint32_t*>(lo_codes + (gid)*32 + 4 * tig);
    const uint32_t l1 = *reinterpret_cast<const uint32_t*>(lo_codes + (gid + 8)*32 + 4 * tig);
    const uint32_t l2 = *reinterpret_cast<const uint32_t*>(lo_codes + (gid)*32 + 16 + 4 * tig);
    const uint32_t l3 = *reinterpret_cast<const uint32_t*>(lo_codes + (gid + 8)*32 + 16 + 4 * tig);
    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(k_codes + (gid)*32 + 4 * tig);
    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(k_codes + (gid)*32 + 16 + 4 * tig);
    const int sfrow = ((lane & 1) << 3) | (lane >> 2);
    const uint32_t sfa = *reinterpret_cast<const uint32_t*>(q_sc + sfrow * 4);
    const uint32_t sfl = *reinterpret_cast<const uint32_t*>(lo_sc + sfrow * 4);
    const uint32_t sfb = *reinterpret_cast<const uint32_t*>(k_sc + gid * 4);
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((unsigned short) 0),
                   "h"((unsigned short) 0), "r"(sfb), "h"((unsigned short) 0), "h"((unsigned short) 0));
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(l0), "r"(l1), "r"(l2), "r"(l3), "r"(b0), "r"(b1), "r"(sfl), "h"((unsigned short) 0),
                   "h"((unsigned short) 0), "r"(sfb), "h"((unsigned short) 0), "h"((unsigned short) 0));
    out[(gid)*8 + 2 * tig] = c[0];
    out[(gid)*8 + 2 * tig + 1] = c[1];
    out[(gid + 8)*8 + 2 * tig] = c[2];
    out[(gid + 8)*8 + 2 * tig + 1] = c[3];
}

int main() {
    // 16 Q rows and 8 K cols, 64 dims each: 4 groups of 16, group g's values on the grid SCALED by 2^(g-1)
    // (group scale 2^(g-1)), all exact in fp4 and in fp32
    std::vector<uint8_t> qc(16 * 32, 0), qs(16 * 4, 0), kc(8 * 32, 0), ks(8 * 4, 0);
    std::vector<float> ref(16 * 8, 0.0f);
    auto rngv = [](int i) {   // deterministic grid value, |v| <= 6
        constexpr float g[7] = {0.5f, -1.0f, 1.5f, -2.0f, 3.0f, -4.0f, 6.0f};
        return g[(unsigned) i % 7];
    };
    for (int r = 0; r < 16; ++r)
        for (int g = 0; g < 4; ++g) {
            qs[r * 4 + g] = h_e4m3_pow2(((g + r) % 5) - 2);   // distinct per (row, group): 2^(g+r mod 5 - 2)
            for (int m = 0; m < 8; ++m) {          // byte m of the group: elements 2m (low), 2m+1 (high)
                const float v0 = rngv(r * 64 + g * 16 + 2 * m), v1 = rngv(r * 64 + g * 16 + 2 * m + 1);
                // store the CODE of the value divided by the group scale (a power of two: still on-grid)
                const float s0 = 1.0f / h_e4m3_pow2v(qs[r * 4 + g]), s1 = s0;
                uint8_t c0, c1;
                {
                    float a0 = v0 * s0, a1 = v1 * s1;   // exact
                    c0 = 0; c1 = 0;
                    (void) a0; (void) a1;
                }
                // host code path mirrors the device helper through the same table
                auto hcode = [](float x) {
                    const float a = std::fabs(x);
                    int i = a >= 5 ? 7 : a >= 3.5 ? 6 : a >= 2.5 ? 5 : a >= 1.75 ? 4 : a >= 1.25 ? 3 : a >= .75 ? 2
                            : a >= .25 ? 1 : 0;
                    return (uint8_t) ((x < 0 ? 8 : 0) | i);
                };
                qc[(size_t) r * 32 + g * 8 + m] = (uint8_t) (hcode(v0 * s0) | (hcode(v1 * s1) << 4));
            }
        }
    for (int n = 0; n < 8; ++n)
        for (int g = 0; g < 4; ++g) {
            ks[n * 4 + g] = h_e4m3_pow2(((g + 2 * n) % 5) - 2);   // distinct per (col, group)
            for (int m = 0; m < 8; ++m) {
                const float v0 = rngv(1000 + n * 64 + g * 16 + 2 * m) * 0.5f,
                            v1 = rngv(1000 + n * 64 + g * 16 + 2 * m + 1) * 0.5f;
                auto hcode = [](float x) {
                    const float a = std::fabs(x);
                    int i = a >= 5 ? 7 : a >= 3.5 ? 6 : a >= 2.5 ? 5 : a >= 1.75 ? 4 : a >= 1.25 ? 3 : a >= .75 ? 2
                            : a >= .25 ? 1 : 0;
                    return (uint8_t) ((x < 0 ? 8 : 0) | i);
                };
                const float s0 = 1.0f / h_e4m3_pow2v(ks[n * 4 + g]);
                kc[(size_t) n * 32 + g * 8 + m] = (uint8_t) (hcode(v0 * s0) | (hcode(v1 * s0) << 4));
            }
        }
    // the lo half: a small residual plane (codes of the hi error / 4, scales two binades down)
    std::vector<uint8_t> lc(16 * 32, 0), ls(16 * 4, 0);
    for (int r = 0; r < 16; ++r)
        for (int g = 0; g < 4; ++g) {
            ls[r * 4 + g] = h_e4m3_pow2(((g + r) % 5) - 4);   // 2^(...-4): two-plus binades below hi's
            for (int m = 0; m < 8; ++m) {
                const float v0 = rngv(5000 + r * 64 + g * 16 + 2 * m) * 0.125f,
                            v1 = rngv(5000 + r * 64 + g * 16 + 2 * m + 1) * 0.125f;
                auto hcode = [](float x) {
                    const float a = std::fabs(x);
                    int i = a >= 5 ? 7 : a >= 3.5 ? 6 : a >= 2.5 ? 5 : a >= 1.75 ? 4 : a >= 1.25 ? 3 : a >= .75 ? 2
                            : a >= .25 ? 1 : 0;
                    return (uint8_t) ((x < 0 ? 8 : 0) | i);
                };
                const float s0 = 1.0f / h_e4m3_pow2v(ls[r * 4 + g]);
                lc[(size_t) r * 32 + g * 8 + m] = (uint8_t) (hcode(v0 * s0) | (hcode(v1 * s0) << 4));
            }
        }
    // reference: dequantize exactly and dot (hi + lo)
    for (int r = 0; r < 16; ++r)
        for (int n = 0; n < 8; ++n) {
            double d = 0;
            for (int g = 0; g < 4; ++g) {
                const float qsc = h_e4m3_pow2v(qs[r * 4 + g]), ksc = h_e4m3_pow2v(ks[n * 4 + g]),
                            lsc = h_e4m3_pow2v(ls[r * 4 + g]);
                for (int m = 0; m < 8; ++m) {
                    const uint8_t qb = qc[(size_t) r * 32 + g * 8 + m], kb = kc[(size_t) n * 32 + g * 8 + m],
                                  lb = lc[(size_t) r * 32 + g * 8 + m];
                    d += (double) h_e2m1(qb & 0xF) * qsc * h_e2m1(kb & 0xF) * ksc;
                    d += (double) h_e2m1(qb >> 4) * qsc * h_e2m1(kb >> 4) * ksc;
                    d += (double) h_e2m1(lb & 0xF) * lsc * h_e2m1(kb & 0xF) * ksc;
                    d += (double) h_e2m1(lb >> 4) * lsc * h_e2m1(kb >> 4) * ksc;
                }
            }
            ref[(size_t) r * 8 + n] = (float) d;
        }
    uint8_t *dqc, *dqs, *dkc, *dks, *dlc, *dls; float* dout;
    cudaMalloc(&dqc, qc.size()); cudaMalloc(&dqs, qs.size()); cudaMalloc(&dkc, kc.size()); cudaMalloc(&dks, ks.size());
    cudaMalloc(&dlc, lc.size()); cudaMalloc(&dls, ls.size());
    cudaMalloc(&dout, ref.size() * 4);
    cudaMemcpy(dqc, qc.data(), qc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dqs, qs.data(), qs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dkc, kc.data(), kc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dks, ks.data(), ks.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dlc, lc.data(), lc.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dls, ls.data(), ls.size(), cudaMemcpyHostToDevice);
    probe<<<1, 32>>>(dqc, dqs, dkc, dks, dlc, dls, dout);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("probe launch: %s\n", cudaGetErrorString(e)); return 1; }
    std::vector<float> got(16 * 8);
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    int bad = 0; double worst = 0; int worst_r = -1, worst_n = -1;
    for (int r = 0; r < 16; ++r)
        for (int n = 0; n < 8; ++n) {
            const double d = std::fabs((double) got[(size_t) r * 8 + n] - ref[(size_t) r * 8 + n]);
            if (d > 1e-4) { if (d > worst) { worst = d; worst_r = r; worst_n = n; } ++bad; }
        }
    std::printf("%s: %d of 128 differ, worst %.6g at row %d col %d (ref %.6g got %.6g)\n",
                bad ? "MISMATCH" : "EXACT", bad, worst, worst_r, worst_n,
                worst_r >= 0 ? ref[(size_t) worst_r * 8 + worst_n] : 0.0,
                worst_r >= 0 ? got[(size_t) worst_r * 8 + worst_n] : 0.0);
    if (bad) {   // dump one bad row: per-column got/ref ratios expose which group's scale went where
        for (int n = 0; n < 8; ++n)
            std::printf("  col %d: ref %10.4f got %10.4f ratio %8.5f\n", n, ref[(size_t) worst_r * 8 + n],
                        got[(size_t) worst_r * 8 + n], ref[(size_t) worst_r * 8 + n] / got[(size_t) worst_r * 8 + n]);
    }
    return bad ? 1 : 0;
}
