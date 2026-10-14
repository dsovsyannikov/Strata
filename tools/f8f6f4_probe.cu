// Layout discovery for mma.m16n8k32.kind::f8f6f4 (A=e4m3, B=e2m1): tries the plausible fragment
// mappings against a CPU reference on exact values; prints which one matches.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

#include <cuda_runtime.h>

__device__ __host__ inline uint8_t e4m3_of(float v) {
    const bool neg = v < 0;
    float x = neg ? -v : v;
    int e = 0;
    while (x >= 2.0f) { x *= 0.5f; ++e; }
    while (x < 1.0f && x > 0) { x *= 2.0f; --e; }
    int man = (int) ((x - 1.0f) * 8.0f + 0.5f);
    int expf = e + 7;
    if (man == 8) { man = 0; ++expf; }
    return (uint8_t) ((neg ? 0x80 : 0) | ((expf << 3) | man));
}
__device__ __host__ inline float e4m3_val(uint8_t c) {
    const float sign = (c & 0x80) ? -1.0f : 1.0f;
    c &= 0x7F;
    const int exp = (c >> 3) & 0xF, man = c & 7;
    if (exp == 0) return sign * (float) man * (1.0f / 512.0f);
    float v = 1.0f + (float) man * 0.125f;
    for (int i = exp - 7; i > 0; --i) v *= 2.0f;
    for (int i = 7 - exp; i > 0; --i) v *= 0.5f;
    return sign * v;
}
__device__ __host__ inline float e2m1_val(uint8_t c) {
    constexpr float v[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}

// variant: 0/1 = A reg order (row-pair then k-half | k-half then row-pair); +2 = B nibbles raw-packed
__global__ void probe(const uint8_t* __restrict__ a_bytes, const uint8_t* __restrict__ b_bytes,
                      const uint8_t* __restrict__ b8_bytes, int variant, float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const uint8_t* ar0 = a_bytes + (size_t) gid * 32;
    const uint8_t* ar8 = a_bytes + (size_t) (gid + 8) * 32;
    uint32_t a[4];
    if ((variant & 1) == 0) {   // a0/a1 = rows g/g+8 at k lo, a2/a3 = the k hi half
        a[0] = *(const uint32_t*) (ar0 + 4 * tig);
        a[1] = *(const uint32_t*) (ar8 + 4 * tig);
        a[2] = *(const uint32_t*) (ar0 + 16 + 4 * tig);
        a[3] = *(const uint32_t*) (ar8 + 16 + 4 * tig);
    } else {                    // a0/a1 = row g at k lo/hi, a2/a3 = row g+8
        a[0] = *(const uint32_t*) (ar0 + 4 * tig);
        a[1] = *(const uint32_t*) (ar0 + 16 + 4 * tig);
        a[2] = *(const uint32_t*) (ar8 + 4 * tig);
        a[3] = *(const uint32_t*) (ar8 + 16 + 4 * tig);
    }
    const uint8_t* bn = b_bytes + (size_t) gid * 32;
    uint32_t b[2];
    const int shift = (variant >> 2) & 3;   // extra: e2m1's bit position within its byte
    if ((variant & 2) == 0) {   // one e2m1 per byte, in bits [shift]
        const uint8_t* p0 = bn + 4 * tig;
        const uint8_t* p1 = bn + 16 + 4 * tig;
        b[0] = (uint32_t) ((p0[0] << shift) | ((p0[1] << shift) << 8) | ((p0[2] << shift) << 16) |
                           ((p0[3] << shift) << 24));
        b[1] = (uint32_t) ((p1[0] << shift) | ((p1[1] << shift) << 8) | ((p1[2] << shift) << 16) |
                           ((p1[3] << shift) << 24));
    } else {                    // two raw nibbles per byte
        const uint8_t* p = bn + 2 * tig;
        const uint8_t* q = bn + 16 + 2 * tig;
        b[0] = (uint32_t) ((p[0] << 2) | (((p[0] >> 4) << 2) << 8) | ((p[1] << 2) << 16) | (((p[1] >> 4) << 2) << 24));
        b[1] = (uint32_t) ((q[0] << 2) | (((q[0] >> 4) << 2) << 8) | ((q[1] << 2) << 16) | (((q[1] >> 4) << 2) << 24));
    }
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    if (variant == 12) {   // sanity: both sides e4m3, the f8 GEMM's own shape
        const uint8_t* bn8 = b8_bytes + (size_t) gid * 32;
        uint32_t b8[2];
        b8[0] = *(const uint32_t*) (bn8 + 4 * tig);
        b8[1] = *(const uint32_t*) (bn8 + 16 + 4 * tig);
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b8[0]), "r"(b8[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    out[gid * 8 + 2 * tig] = c[0];
    out[gid * 8 + 2 * tig + 1] = c[1];
    out[(gid + 8) * 8 + 2 * tig] = c[2];
    out[(gid + 8) * 8 + 2 * tig + 1] = c[3];
}

// The P.V construction exactly as prompt_attn_nf4_kernel builds it: P folded and quantized to e4m3
// (hi + lo residual) with e4m3x2 pairs, V's nibbles in bits [7:4], two chained MMAs - against a CPU
// emulation on random floats.
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return (uint32_t) r;
}
__device__ __forceinline__ void mma_pair(const uint32_t* ah, const uint32_t* al, const uint32_t* b, float* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(ah[0]), "r"(ah[1]), "r"(ah[2]), "r"(ah[3]), "r"(b[0]), "r"(b[1]));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(al[0]), "r"(al[1]), "r"(al[2]), "r"(al[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float e4m3_sdec(uint8_t c) {
    const float sign = (c & 0x80) ? -1.0f : 1.0f;
    c &= 0x7F;
    const int exp = (c >> 3) & 0xF, man = c & 7;
    if (exp == 0) return sign * (float) man * (1.0f / 512.0f);
    float v = 1.0f + (float) man * 0.125f;
    for (int i = exp - 7; i > 0; --i) v *= 2.0f;
    for (int i = 7 - exp; i > 0; --i) v *= 0.5f;
    return sign * v;
}
__global__ void pv_probe(const float* __restrict__ p,    // [16 rows][32 cells] random
                         const uint8_t* __restrict__ vc, // [8 cols][32 cells] raw nibble codes
                         const uint8_t* __restrict__ pc, // the same P pre-encoded on the host (order 4)
                         int order, float* __restrict__ out, uint32_t* __restrict__ dbg) {
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, tig = lane & 3;
    uint32_t ah[4], al[4];
    if (order & 4) {   // the sweep's own load path: raw u32 reads of the host-encoded codes
        ah[0] = *(const uint32_t*) (pc + (size_t) gid * 32 + 4 * tig);
        ah[1] = *(const uint32_t*) (pc + (size_t) (gid + 8) * 32 + 4 * tig);
        ah[2] = *(const uint32_t*) (pc + (size_t) gid * 32 + 16 + 4 * tig);
        ah[3] = *(const uint32_t*) (pc + (size_t) (gid + 8) * 32 + 16 + 4 * tig);
    } else {
    for (int r = 0; r < 4; ++r) {
        const int row = (r & 1) ? gid + 8 : gid;
        const int c0 = (r >> 1) ? 16 + 4 * tig : 4 * tig;
        float x[4], dq[4];
        for (int i = 0; i < 4; ++i) x[i] = p[row * 32 + c0 + i];
        uint32_t h = 0, l = 0;
        for (int i = 0; i < 2; ++i) {
            uint32_t ph = (order & 1) ? e4m3x2(x[2 * i + 1], x[2 * i]) : e4m3x2(x[2 * i], x[2 * i + 1]);
            dq[2 * i] = e4m3_sdec((uint8_t) (ph & 0xFF));
            dq[2 * i + 1] = e4m3_sdec((uint8_t) (ph >> 8));
            if (order & 1) { const float tt = dq[2 * i]; dq[2 * i] = dq[2 * i + 1]; dq[2 * i + 1] = tt; }
            h |= (ph & 0xFFFFu) << (16 * i);
            if (!(order & 2)) {
                uint32_t pl = e4m3x2(x[2 * i] - dq[2 * i], x[2 * i + 1] - dq[2 * i + 1]);
                l |= (pl & 0xFFFFu) << (16 * i);
            }
        }
        ah[r] = h;
        al[r] = l;
    }
    }
    if (order & 4) { al[0] = al[1] = al[2] = al[3] = 0; }   // the sweep ran hi-only
    // B: column gid, bytes = nibble in bits [5:2] at cells 4tig..+4 (b0) and 16+4tig..+4 (b1)
    uint32_t b[2];
    for (int half = 0; half < 2; ++half) {
        uint32_t bb = 0;
        for (int i = 0; i < 4; ++i) bb |= ((uint32_t) (vc[gid * 32 + half * 16 + 4 * tig + i] & 0xF)) << (8 * i + 2);
        b[half] = bb;
    }
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    mma_pair(ah, al, b, c);
    if (order == 0 && lane < 4) { dbg[lane * 4 + 0] = ah[0]; dbg[lane * 4 + 1] = ah[1]; dbg[lane * 4 + 2] = b[0]; dbg[lane * 4 + 3] = __float_as_int(c[0]); }
    out[gid * 8 + 2 * tig] = c[0];
    out[gid * 8 + 2 * tig + 1] = c[1];
    out[(gid + 8) * 8 + 2 * tig] = c[2];
    out[(gid + 8) * 8 + 2 * tig + 1] = c[3];
}

__global__ void e4m3x2_test(const float* x, uint32_t* r) {
    r[0] = e4m3x2(x[0], x[1]);
    r[1] = e4m3x2(x[2], x[3]);
}

int pv_main() {
    std::vector<float> P(16 * 32);
    std::vector<uint8_t> Vc(8 * 32);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> pd(-1.0f, 1.0f);
    std::uniform_int_distribution<int> nb(0, 15);
    if (std::getenv("PV_SMALL")) {   // values exactly representable in e4m3: any layout/packing bug
        // shows EXACT; larger values then isolate range behavior
        constexpr float g[6] = {0.5f, 1.0f, 1.5f, 2.0f, 4.0f, 8.0f};
        for (size_t i = 0; i < P.size(); ++i) P[i] = g[i % 6] * ((i % 5) == 0 ? -1.0f : 1.0f);
    } else {
        for (auto& x : P) x = pd(rng) * 200.0f;
    }
    if (std::getenv("PV_V1")) {   // every V = +1.0 (code 2): D[r][n] = the row sum of P, trivial to check
        for (auto& x : Vc) x = 2;
    } else {
        for (auto& x : Vc) x = (uint8_t) nb(rng);
    }
    // CPU emulation of the device sequence (e4m3 encode emulated by scanning the grid)
    auto enc = [](float v) -> uint8_t {   // RNE to e4m3, saturating - mirror cvt.rn.satfinite
        float a = std::fabs(v);
        if (a > 464.0f) return v < 0 ? 0x80 | 0x7E : 0x7E;
        float best = 0; uint8_t bc = 0;   // brute force the nearest representable
        double bd = 1e30;
        for (int c = 0; c < 256; ++c) {
            const uint8_t cc = (uint8_t) c;
            const float w = e4m3_val(cc);
            const double d = std::fabs((double) w - (double) v);
            if (d < bd - 1e-12) { bd = d; best = w; bc = cc; }
        }
        return bc;
    };
    std::vector<float> ref(16 * 8, 0.f);
    for (int r = 0; r < 16; ++r)
        for (int n = 0; n < 8; ++n) {
            double d = 0;
            for (int k = 0; k < 32; ++k) {
                const uint8_t h = enc(P[r * 32 + k]);
                const float hd = e4m3_val(h);
                const uint8_t l = enc(P[r * 32 + k] - hd);
                const float ld = e4m3_val(l);
                const float v = e2m1_val((uint8_t) (Vc[n * 32 + k] & 0xF));
                d += (double) (hd + ld) * v;
            }
            ref[r * 8 + n] = (float) d;
        }
    float *dP, *dO; uint8_t *dV, *dPC;
    cudaMalloc(&dP, P.size() * 4); cudaMalloc(&dO, ref.size() * 4); cudaMalloc(&dV, Vc.size());
    std::vector<uint8_t> PC(P.size());
    for (size_t i = 0; i < P.size(); ++i) PC[i] = enc(P[i]);   // the host-encoded hi codes (order 4)
    cudaMalloc(&dPC, PC.size());
    cudaMemcpy(dPC, PC.data(), PC.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dP, P.data(), P.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, Vc.data(), Vc.size(), cudaMemcpyHostToDevice);
    for (int order = 0; order < 8; ++order) {
        if (order != 0 && order != 2 && order != 4 && order != 6) continue;
        uint32_t* ddbg;
        cudaMalloc(&ddbg, 64);
        pv_probe<<<1, 32>>>(dP, dV, dPC, order, dO, ddbg);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::printf("pv_probe: %s\n", cudaGetErrorString(e)); return 1; }
        if (order == 0) {
            uint32_t h[16];
            cudaMemcpy(h, ddbg, 64, cudaMemcpyDeviceToHost);
            for (int L = 0; L < 4; ++L)
                std::printf("  lane %d: ah0=%08x ah1=%08x b0=%08x c0=%g (ref %g)\n", L, h[L * 4], h[L * 4 + 1],
                            h[L * 4 + 2], *(float*) &h[L * 4 + 3], ref[(L >> 2) * 8 + 2 * (L & 3)]);
        }
        std::vector<float> got(16 * 8);
        cudaMemcpy(got.data(), dO, got.size() * 4, cudaMemcpyDeviceToHost);
        int bad = 0; double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = std::fabs((double) got[i] - ref[i]) / (1.0 + std::fabs(ref[i]));
            if (d > 1e-5) { ++bad; if (d > worst) worst = d; }
        }
        std::printf("pv construction (order %d, %s%s): %s (%d differ, worst rel %.6g)\n", order,
                    (order & 1) ? "bytes swapped" : "bytes as packed",
                    (order & 2) ? ", hi only" : ", hi+lo", bad ? "MISMATCH" : "EXACT", bad, worst);
    }
    return 0;
}

int main() {
    {   // e4m3x2 in isolation: (1.0, 2.0) must give bytes {0x38, 0x40}, (0.5, -1.5) {0x30, 0xBC}
        float din[4] = {1.0f, 2.0f, 0.5f, -1.5f};
        float* dd; uint32_t* dr;
        cudaMalloc(&dd, 16); cudaMalloc(&dr, 16);
        cudaMemcpy(dd, din, 16, cudaMemcpyHostToDevice);
        e4m3x2_test<<<1, 1>>>(dd, dr);
        cudaDeviceSynchronize();
        uint32_t r[2];
        cudaMemcpy(r, dr, 8, cudaMemcpyDeviceToHost);
        std::printf("e4m3x2(1,2)=%04x (want 4038)  e4m3x2(0.5,-1.5)=%04x (want bc30)\n", r[0] & 0xFFFF,
                    r[1] & 0xFFFF);
    }
    const int pv = pv_main();

    std::vector<uint8_t> A(16 * 32), B(8 * 32);
    auto rngv = [](int i) {
        constexpr float g[6] = {0.5f, 1.0f, 1.5f, 2.0f, 4.0f, 8.0f};
        const float v = g[(unsigned) i % 6];
        return (i % 5) == 0 ? -v : v;
    };
    for (int r = 0; r < 16; ++r)
        for (int k = 0; k < 32; ++k)
            A[(size_t) r * 32 + k] = e4m3_of(rngv(r * 32 + k));
    auto rngn = [](int i) {
        const int m[4] = {1, 2, 5, 7};
        return (uint8_t) (((i % 3) == 0 ? 8 : 0) | m[(unsigned) i % 4]);
    };
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 32; ++k)
            B[(size_t) n * 32 + k] = rngn(n * 32 + k);   // raw nibble per byte position k
    // the e4m3-typed B twin (variant 12): exact magnitudes, distinct per column for the reference
    std::vector<uint8_t> B8(8 * 32);
    for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 32; ++k)
            B8[(size_t) n * 32 + k] = e4m3_of(rngv(1000 + n * 32 + k));
    std::vector<float> ref(16 * 8, 0.f), ref8(16 * 8, 0.f);
    for (int r = 0; r < 16; ++r)
        for (int n = 0; n < 8; ++n) {
            double d = 0, d8 = 0;
            for (int k = 0; k < 32; ++k) {
                d += (double) e4m3_val(A[(size_t) r * 32 + k]) * e2m1_val(B[(size_t) n * 32 + k]);
                d8 += (double) e4m3_val(A[(size_t) r * 32 + k]) * e4m3_val(B8[(size_t) n * 32 + k]);
            }
            ref[(size_t) r * 8 + n] = (float) d;
            ref8[(size_t) r * 8 + n] = (float) d8;
        }
    uint8_t *dA, *dB, *dB8; float* dO;
    cudaMalloc(&dA, A.size()); cudaMalloc(&dB, B.size()); cudaMalloc(&dB8, B8.size());
    cudaMalloc(&dO, ref.size() * 4);
    cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB8, B8.data(), B8.size(), cudaMemcpyHostToDevice);
    for (int v = 0; v <= 12; ++v) {
        probe<<<1, 32>>>(dA, dB, dB8, v, dO);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::printf("variant %d: %s\n", v, cudaGetErrorString(e)); return 1; }
        const std::vector<float>& rr = v == 12 ? ref8 : ref;
        std::vector<float> got(16 * 8);
        cudaMemcpy(got.data(), dO, got.size() * 4, cudaMemcpyDeviceToHost);
        int bad = 0; double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = std::fabs((double) got[i] - rr[i]);
            if (d > 1e-4 * (1.0 + std::fabs(rr[i]))) { ++bad; if (d > worst) worst = d; }
        }
        std::printf("variant %2d (%s): %s (%d differ, worst %.6g)\n", v,
                    v == 12 ? "e4m3 x e4m3 sanity" : "e4m3 x e2m1",
                    bad ? "MISMATCH" : "EXACT", bad, worst);
    }
    return 0;
}
