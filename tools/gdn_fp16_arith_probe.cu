// tools/gdn_fp16_arith_probe.cu - what would F16 ARITHMETIC (not just storage) cost in the GDN
// recurrence?
//
// The engine's step order (src/kernels/cuda/native_gdn.cu, arithmetic pinned to llama.cpp):
//     kv    = sum_i s[i] * k[i]
//     delta = (v - g * kv) * beta
//     s[i]  = g * s[i] + k[i] * delta
//     o     = sum_i s[i] * q[i] * scale        (from the POST-update state)
// One thread per (h, j) state column, layout (i, h, j) = ((i*H_V+h)*S+j), like gdn.cu.
//
// Five variants over the same inputs, against an F64 reference:
//   f64/f64  - reference (double storage and arithmetic)
//   f32/f32  - engine default
//   f16/f32  - --gdn-f16 (narrow at store only; gdn_state_io.cuh)
//   f16/f16  - every product AND every accumulation rounds to half (a naive __half rewrite)
//   f16p/f32 - products round to half, accumulations stay F32 (the most favorable f16 reading)
//
// Metric: rel L1 of o over all steps and per window (does the error compound?), rel L1 of the
// final state, non-finite output count.  dec is log-uniform in [exp(-6.6), exp(-2.0)] - the
// regime gdn_parity.cpp documents as the real model's ("gate comfortably negative"); beta in
// [0.3, 1].  NOTE: a wider adversarial range (dec up to ~0.98 with fully random keys) walks the
// state past half's 65504 ceiling around t~284 and then even f16 STORAGE overflows to inf/NaN -
// half's dynamic range is a real hazard independent of its mantissa.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int S = 128;   // head dim = state rows/cols
constexpr int H_V = 48;  // v heads; state is [S, H_V, S]
constexpr int H_K = 16;  // k heads (qh = h % H_K)
int T = 4096;            // decode steps (argv[1] may lower it)
constexpr float OSCALE = 0.08838834764831845f;  // 1/sqrt(128)

#define CK(x)                                                                                \
    do {                                                                                     \
        const cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

__device__ __forceinline__ float st_get(float v) { return v; }
__device__ __forceinline__ float st_get(__half v) { return __half2float(v); }
__device__ __forceinline__ void st_put(float& p, float v) { p = v; }
__device__ __forceinline__ void st_put(__half& p, float v) { p = __float2half(v); }

// MODE 0: F32 arithmetic, storage at ST's width (f32/f32 and f16/f32)
// MODE 1: F64 reference (ST = double)
// MODE 2: F16 arithmetic - products and sums round to half (ST = __half)
// MODE 3: half products, F32 accumulation (ST = __half)
template <typename ST, int MODE>
__global__ void step_kernel(ST* __restrict__ st, const float* __restrict__ q, const float* __restrict__ k,
                            const float* __restrict__ v, const float* __restrict__ dec,
                            const float* __restrict__ beta, float* __restrict__ o_all, int t) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    const int qh = h % H_K;
    const float* kt = k + (size_t) ((size_t) t * H_K + qh) * S;
    const float* qt = q + (size_t) ((size_t) t * H_K + qh) * S;
    const float g = dec[t * H_V + h];
    const float b = beta[t * H_V + h];
    const float vraw = v[(size_t) (t * H_V + h) * S + j];
    const size_t col = (size_t) j;  // the column this thread owns, at row i: ((i*H_V+h)*S+j)

    if constexpr (MODE == 1) {
        double s[S];
#pragma unroll
        for (int i = 0; i < S; ++i) s[i] = st[((size_t) i * H_V + h) * S + col];
        double kv = 0.0;
#pragma unroll
        for (int i = 0; i < S; ++i) kv += s[i] * (double) kt[i];
        const double d = ((double) vraw - (double) g * kv) * (double) b;
        double attn = 0.0;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            s[i] = (double) g * s[i] + (double) kt[i] * d;
            attn += s[i] * (double) qt[i];
        }
        o_all[(size_t) (t * H_V + h) * S + col] = (float) (attn * (double) OSCALE);
#pragma unroll
        for (int i = 0; i < S; ++i) st[((size_t) i * H_V + h) * S + col] = s[i];
    } else if constexpr (MODE == 2) {
        const __half gh = __float2half(g), bh = __float2half(b), vh = __float2half(vraw);
        __half s[S];
#pragma unroll
        for (int i = 0; i < S; ++i) s[i] = st[((size_t) i * H_V + h) * S + col];
        __half kv = __float2half(0.f);
#pragma unroll
        for (int i = 0; i < S; ++i) kv = __hadd(kv, __hmul(s[i], __float2half(kt[i])));
        const __half d = __hmul(__hsub(vh, __hmul(gh, kv)), bh);
        __half attn = __float2half(0.f);
#pragma unroll
        for (int i = 0; i < S; ++i) {
            s[i] = __hadd(__hmul(gh, s[i]), __hmul(__float2half(kt[i]), d));
            attn = __hadd(attn, __hmul(s[i], __float2half(qt[i])));
        }
        o_all[(size_t) (t * H_V + h) * S + col] = __half2float(attn) * OSCALE;
#pragma unroll
        for (int i = 0; i < S; ++i) st[((size_t) i * H_V + h) * S + col] = s[i];
    } else if constexpr (MODE == 3) {
        const __half gh = __float2half(g);
        __half s[S];
#pragma unroll
        for (int i = 0; i < S; ++i) s[i] = st[((size_t) i * H_V + h) * S + col];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < S; ++i) kv += __half2float(__hmul(s[i], __float2half(kt[i])));
        const __half d = __float2half((vraw - g * kv) * b);
        float attn = 0.f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            st_put(s[i], __half2float(__hadd(__hmul(gh, s[i]), __hmul(__float2half(kt[i]), d))));
            attn += __half2float(__hmul(s[i], __float2half(qt[i])));
        }
        o_all[(size_t) (t * H_V + h) * S + col] = attn * OSCALE;
#pragma unroll
        for (int i = 0; i < S; ++i) st[((size_t) i * H_V + h) * S + col] = s[i];
    } else {
        float s[S];
#pragma unroll
        for (int i = 0; i < S; ++i) s[i] = st_get(st[((size_t) i * H_V + h) * S + col]);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < S; ++i) kv += s[i] * kt[i];
        const float d = (vraw - g * kv) * b;
        float attn = 0.f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            s[i] = g * s[i] + kt[i] * d;
            attn += s[i] * qt[i];
        }
        o_all[(size_t) (t * H_V + h) * S + col] = attn * OSCALE;
#pragma unroll
        for (int i = 0; i < S; ++i) st_put(st[((size_t) i * H_V + h) * S + col], s[i]);
    }
}

__global__ void h2f(float* __restrict__ dst, const __half* __restrict__ src, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

struct Rel {
    double num = 0, den = 0;
    long long nonfinite = 0;
    void add(float got, double ref) {
        if (!std::isfinite(got)) ++nonfinite;
        num += std::fabs((double) got - ref);
        den += std::fabs(ref);
    }
    double val() const { return den > 1e-30 ? num / den : 0.0; }
};

struct Tally {
    Rel all, first, last, st;
    long long first_bad = -1;
    float bad_got = 0.f;
    double bad_ref = 0.0;
};

static Tally compare(const float* got, const float* ref, const float* st_got, const double* st_ref,
                     size_t n_o, size_t n_st) {
    Tally t;
    const size_t per_t = (size_t) H_V * S;
    for (size_t i = 0; i < n_o; ++i) {
        const int step = int(i / per_t);
        t.all.add(got[i], ref[i]);
        if (step < 512) t.first.add(got[i], ref[i]);
        if (step >= T - 512) t.last.add(got[i], ref[i]);
        if (t.first_bad < 0 && !std::isfinite(got[i])) {
            t.first_bad = (long long) i;
            t.bad_got = got[i];
            t.bad_ref = ref[i];
        }
    }
    for (size_t i = 0; i < n_st; ++i) t.st.add(st_got[i], st_ref[i]);
    return t;
}

static void report(const char* name, const Tally& t) {
    const long long fb = t.first_bad;
    const long long per_t = (long long) H_V * S;
    std::printf("%-9s  o(all) %.3e  o(first512) %.3e  o(last512) %.3e  state %.3e  nonfinite %lld", name,
                t.all.val(), t.first.val(), t.last.val(), t.st.val(), t.all.nonfinite + t.st.nonfinite);
    if (fb >= 0)
        std::printf("  first@t=%lld h=%lld j=%lld got=%.3e ref=%.3e", fb / per_t, (fb / S) % H_V, fb % S,
                    t.bad_got, t.bad_ref);
    std::printf("\n");
}

int main(int argc, char** argv) {
    if (argc > 1) {
        const int want = std::atoi(argv[1]);
        if (want > 0 && want <= 4096) T = want;
    }
    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    const size_t nqk = (size_t) T * H_K * S, nv = (size_t) T * H_V * S, nst = (size_t) S * H_V * S;

    std::vector<float> q(nqk), k(nqk), v(nv), dec((size_t) T * H_V), beta((size_t) T * H_V);
    for (auto& x : q) x = gauss(rng);
    for (auto& x : k) x = gauss(rng);
    for (auto& x : v) x = gauss(rng);
    for (auto& x : dec) x = std::exp(-(2.0f + 4.6f * uni(rng)));  // [exp(-6.6), exp(-2.0)]: the real regime
    for (auto& x : beta) x = 0.3f + 0.7f * uni(rng);

    float *d_q, *d_k, *d_v, *d_dec, *d_beta;
    CK(cudaMalloc(&d_q, q.size() * 4)); CK(cudaMalloc(&d_k, k.size() * 4)); CK(cudaMalloc(&d_v, v.size() * 4));
    CK(cudaMalloc(&d_dec, dec.size() * 4)); CK(cudaMalloc(&d_beta, beta.size() * 4));
    CK(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dec, dec.data(), dec.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_beta, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));

    double* d_st64; float *d_st32, *d_st16f, *d_st16h, *d_st16p;
    float *d_o64, *d_o32, *d_o16f, *d_o16h, *d_o16p, *d_st_f;
    CK(cudaMalloc(&d_st64, nst * 8)); CK(cudaMalloc(&d_st32, nst * 4)); CK(cudaMalloc(&d_st16f, nst * 2));
    CK(cudaMalloc(&d_st16h, nst * 2)); CK(cudaMalloc(&d_st16p, nst * 2)); CK(cudaMalloc(&d_st_f, nst * 4));
    CK(cudaMalloc(&d_o64, nv * 4)); CK(cudaMalloc(&d_o32, nv * 4)); CK(cudaMalloc(&d_o16f, nv * 4));
    CK(cudaMalloc(&d_o16h, nv * 4)); CK(cudaMalloc(&d_o16p, nv * 4));
    CK(cudaMemset(d_st64, 0, nst * 8)); CK(cudaMemset(d_st32, 0, nst * 4));
    CK(cudaMemset(d_st16f, 0, nst * 2)); CK(cudaMemset(d_st16h, 0, nst * 2));
    CK(cudaMemset(d_st16p, 0, nst * 2));

    const dim3 grid(H_V), block(S);
    for (int t = 0; t < T; ++t) {
        step_kernel<double, 1><<<grid, block>>>(d_st64, d_q, d_k, d_v, d_dec, d_beta, d_o64, t);
        step_kernel<float, 0><<<grid, block>>>(d_st32, d_q, d_k, d_v, d_dec, d_beta, d_o32, t);
        step_kernel<__half, 0><<<grid, block>>>((__half*) d_st16f, d_q, d_k, d_v, d_dec, d_beta, d_o16f, t);
        step_kernel<__half, 2><<<grid, block>>>((__half*) d_st16h, d_q, d_k, d_v, d_dec, d_beta, d_o16h, t);
        step_kernel<__half, 3><<<grid, block>>>((__half*) d_st16p, d_q, d_k, d_v, d_dec, d_beta, d_o16p, t);
        if ((t & 511) == 511) CK(cudaDeviceSynchronize());
    }
    CK(cudaDeviceSynchronize());

    std::vector<float> ref_o(nv), got_o(nv);
    std::vector<double> ref_st(nst);
    std::vector<float> got_st(nst);
    CK(cudaMemcpy(ref_o.data(), d_o64, nv * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref_st.data(), d_st64, nst * 8, cudaMemcpyDeviceToHost));
    const int f32s = int((nst * 4 + 3) / 4);

    auto report_variant = [&](const char* name, const float* d_o, const void* d_st, bool st_is_half) {
        CK(cudaMemcpy(got_o.data(), d_o, nv * 4, cudaMemcpyDeviceToHost));
        if (st_is_half) {
            h2f<<<(unsigned) ((nst + 255) / 256), 256>>>(d_st_f, (const __half*) d_st, nst);
            CK(cudaGetLastError());
            CK(cudaMemcpy(got_st.data(), d_st_f, nst * 4, cudaMemcpyDeviceToHost));
        } else {
            CK(cudaMemcpy(got_st.data(), d_st, nst * 4, cudaMemcpyDeviceToHost));
        }
        report(name, compare(got_o.data(), ref_o.data(), got_st.data(), ref_st.data(), nv, nst));
    };

    std::printf("GDN recurrence, S=%d h_v=%d h_k=%d, T=%d steps, dec in [%.4f, %.4f]\n", S, H_V, H_K, T,
                std::exp(-6.6f), std::exp(-2.0f));
    report_variant("f32/f32", d_o32, d_st32, false);
    report_variant("f16/f32", d_o16f, d_st16f, true);
    report_variant("f16/f16", d_o16h, d_st16h, true);
    report_variant("f16p/f32", d_o16p, d_st16p, true);
    (void) f32s;
    return 0;
}
