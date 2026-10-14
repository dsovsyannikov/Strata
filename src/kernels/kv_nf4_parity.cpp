// src/kernels/kv_nf4_parity.cpp - the NVFP4 KV codec and kernels against the host reference (kv_nf4.hpp).
// Pins: every E2M1/E4M3 round trip, the append kernels vs the shared host reference
// (nf4_quantize_group16), the batched steps form bitwise against per-cell, the gather vs the host
// decode, and the Hadamard invariance with fwht256 itself. Synthetic inputs, no model.
#include "strata/kernels/kv_nf4.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::kernels;

namespace {
const QsaShapes kShapes = qsa_real_shapes();   // n_head_kv 2, head_dim 256, page 4
#define CK(x) do { if (!(x)) { std::printf("*** FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++bad; } } while (0)
int run() {
    int bad = 0;
    for (int c = 0; c < 16; ++c) {
        const float v = nf4_e2m1_decode((uint8_t) c);
        CK(nf4_e2m1_decode(nf4_e2m1_encode(v)) == v);   // -0 re-encodes as +0 by design
    }
    for (int b = 0; b < 0x7F; ++b) {
        const float v = nf4_e4m3_decode((uint8_t) b);
        CK(nf4_e4m3_decode(nf4_e4m3_encode(v)) == v);
    }
    CK(nf4_e4m3_encode(1000.0f) == 0x7E);
    CK(nf4_e4m3_encode(-1000.0f) == 0xFE);
    CK(std::isnan(nf4_e4m3_decode(0x7F)));
    const int64_t cells = 64;
    const int64_t pages = cells / kShapes.page_size;
    std::vector<int32_t> table((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) table[(size_t) i] = (int32_t) i;
    const size_t nvk = (size_t) kShapes.n_head_kv * kShapes.head_dim;
    std::vector<float> kcur((size_t) cells * nvk), vcur((size_t) cells * nvk);
    auto seed = [](std::vector<float>& v, double salt) {
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = (float) (0.9 * std::sin(salt + 0.37 * (double) i) + 0.05 * std::cos(salt * 1.7 + 0.11 * (double) i));
    };
    seed(kcur, 1.25);
    seed(vcur, 4.75);
    const size_t pool_bytes = (size_t) cells * kShapes.n_head_kv * kv_nf4_bytes_per_head((int) kShapes.head_dim);
    uint8_t *k_dev = nullptr, *v_dev = nullptr, *k2_dev = nullptr, *v2_dev = nullptr;
    int32_t *table_dev = nullptr, *step_dev = nullptr, *steps_dev = nullptr;
    float *kcur_dev = nullptr, *vcur_dev = nullptr;
    uint16_t *ks_dev = nullptr, *vs_dev = nullptr;
    cudaMalloc(&k_dev, pool_bytes); cudaMalloc(&v_dev, pool_bytes);
    cudaMalloc(&k2_dev, pool_bytes); cudaMalloc(&v2_dev, pool_bytes);
    cudaMalloc(&table_dev, table.size() * 4);
    cudaMalloc(&step_dev, 8 * 4);
    cudaMalloc(&kcur_dev, kcur.size() * 4); cudaMalloc(&vcur_dev, vcur.size() * 4);
    cudaMalloc(&ks_dev, (size_t) cells * nvk * 2); cudaMalloc(&vs_dev, (size_t) cells * nvk * 2);
    cudaMemcpy(table_dev, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(kcur_dev, kcur.data(), kcur.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(vcur_dev, vcur.data(), vcur.size() * 4, cudaMemcpyHostToDevice);
    std::vector<int32_t> step(8, 0);
    for (int64_t c = 0; c < cells; ++c) {
        step[kStepPos] = (int32_t) c;
        cudaMemcpy(step_dev, step.data(), 8 * 4, cudaMemcpyHostToDevice);
        kv_append_nf4_step(k_dev, v_dev, table_dev, step_dev, kcur_dev + (size_t) c * nvk, vcur_dev + (size_t) c * nvk, kShapes, nullptr);
    }
    std::vector<int32_t> steps((size_t) cells * 8, 0);
    for (int64_t c = 0; c < cells; ++c) steps[(size_t) c * 8 + kStepPos] = (int32_t) c;
    cudaMalloc(&steps_dev, steps.size() * 4);
    cudaMemcpy(steps_dev, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice);
    kv_append_nf4_steps(k2_dev, v2_dev, table_dev, steps_dev, 8, (int) cells, kcur_dev, vcur_dev, kShapes, nullptr);
    cudaDeviceSynchronize();
    std::vector<uint8_t> k_host(pool_bytes), v_host(pool_bytes), k2_host(pool_bytes), v2_host(pool_bytes);
    cudaMemcpy(k_host.data(), k_dev, pool_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_host.data(), v_dev, pool_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(k2_host.data(), k2_dev, pool_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v2_host.data(), v2_dev, pool_bytes, cudaMemcpyDeviceToHost);
    size_t block_diffs = 0, cell_diffs = 0, value_checks = 0;
    double worst = 0.0, sum_rel = 0.0;
    for (int64_t c = 0; c < cells; ++c)
        for (int h = 0; h < kShapes.n_head_kv; ++h) {
            const size_t row = (size_t) ((c / kShapes.page_size) * kShapes.n_head_kv + h) * kShapes.page_size + (size_t) (c % kShapes.page_size);
            const size_t base = row * sizeof(block_nf4);
            for (int g = 0; g < 16; ++g) {
                float ref_x[16];
                uint8_t ref_qs[8], ref_sb;
                for (int i = 0; i < 16; ++i)
                    ref_x[i] = kcur[(size_t) c * nvk + (size_t) h * kShapes.head_dim + (size_t) g * 16 + (size_t) i];
                nf4_quantize_group16(ref_x, ref_sb, ref_qs);
                if (std::memcmp(k_host.data() + base + (size_t) g * 8, ref_qs, 8) != 0 || k_host[base + 128 + (size_t) g] != ref_sb)
                    ++block_diffs;
                float got[16], want[16];
                nf4_dequant_group16(k_host.data() + base + (size_t) g * 8, k_host[base + 128 + (size_t) g], got);
                nf4_dequant_group16(ref_qs, ref_sb, want);
                for (int i = 0; i < 16; ++i) {
                    ++value_checks;
                    const double rel = std::fabs((double) got[i] - (double) want[i]) / (std::fabs((double) want[i]) + 1e-9);
                    sum_rel += rel;
                    if (rel > worst) worst = rel;
                }
            }
            if (std::memcmp(k_host.data() + base, k2_host.data() + base, sizeof(block_nf4)) != 0 ||
                std::memcmp(v_host.data() + base, v2_host.data() + base, sizeof(block_nf4)) != 0)
                ++cell_diffs;
        }
    CK(block_diffs == 0);
    CK(cell_diffs == 0);
    CK(worst == 0.0);
    std::printf("  %-44s %zu of 2048 blocks differ\n", "append bytes vs host reference", block_diffs);
    std::printf("  %-44s %zu of 64 cells differ\n", "batched steps vs per-cell (bitwise)", cell_diffs);
    std::printf("  %-44s %.3e worst, %.3e mean\n", "decoded value vs reference", worst, sum_rel / (double) value_checks);

    {   // a RING-style table (page -> a shuffled slot): the pools hold SLOT rows, the host copy logical
        // ones - the identity table above cannot tell the two apart (v1's append wrote logical rows into
        // the pools and only the ring's overflow crash caught it)
        std::vector<int32_t> rtable((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) rtable[(size_t) i] = (int32_t) ((i * 7 + 3) % pages);
        uint8_t *rk = nullptr, *rv = nullptr;
        int32_t* rt = nullptr;
        cudaMalloc(&rk, pool_bytes); cudaMalloc(&rv, pool_bytes);
        cudaMalloc(&rt, rtable.size() * 4);
        cudaMemcpy(rt, rtable.data(), rtable.size() * 4, cudaMemcpyHostToDevice);
        cudaMemset(rk, 0xA5, pool_bytes); cudaMemset(rv, 0x5A, pool_bytes);
        strata::kernels::KvHostPools host;
        std::vector<uint8_t> hk(pool_bytes, 0), hv(pool_bytes, 0);
        uint8_t *dhk = nullptr, *dhv = nullptr;
        cudaMalloc(&dhk, pool_bytes); cudaMalloc(&dhv, pool_bytes);
        cudaMemset(dhk, 0, pool_bytes); cudaMemset(dhv, 0, pool_bytes);
        host.k_nf4 = dhk; host.v_nf4 = dhv;
        kv_append_nf4_steps(rk, rv, rt, steps_dev, 8, (int) cells, kcur_dev, vcur_dev, kShapes, nullptr, &host);
        cudaDeviceSynchronize();
        std::vector<uint8_t> rgot(pool_bytes), hgot(pool_bytes);
        cudaMemcpy(rgot.data(), rk, pool_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hgot.data(), dhk, pool_bytes, cudaMemcpyDeviceToHost);
        size_t slot_bad = 0, host_bad = 0;
        for (int64_t c = 0; c < cells; ++c)
            for (int h = 0; h < kShapes.n_head_kv; ++h) {
                const int64_t sl = rtable[(size_t) (c / kShapes.page_size)];
                const size_t srow = (size_t) ((sl * kShapes.n_head_kv + h) * kShapes.page_size + (c % kShapes.page_size));
                const size_t lrow = (size_t) (((c / kShapes.page_size) * kShapes.n_head_kv + h) * kShapes.page_size + (c % kShapes.page_size));
                if (std::memcmp(rgot.data() + srow * sizeof(block_nf4), k_host.data() + lrow * sizeof(block_nf4),
                                sizeof(block_nf4)) != 0) ++slot_bad;
                if (std::memcmp(hgot.data() + lrow * sizeof(block_nf4), k_host.data() + lrow * sizeof(block_nf4),
                                sizeof(block_nf4)) != 0) ++host_bad;
            }
        // a wrong row formula also lands past a ring pool smaller than the context - the engine caught
        // that one as an illegal access before this test existed
        CK(slot_bad == 0);
        CK(host_bad == 0);
        std::printf("  %-44s %zu cell rows wrong (host %zu)\n", "ring-table append: slot vs logical rows", slot_bad,
                    host_bad);
        cudaFree(rk); cudaFree(rv); cudaFree(rt); cudaFree(dhk); cudaFree(dhv);
    }
    const int64_t n_ids = 32;
    std::vector<int32_t> ids((size_t) n_ids);
    for (int64_t i = 0; i < n_ids; ++i) ids[(size_t) i] = (int32_t) (i * 2);
    step[kStepWidth] = (int32_t) n_ids;
    int32_t* ids_dev = nullptr;
    cudaMalloc(&ids_dev, ids.size() * 4);
    cudaMemcpy(ids_dev, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(step_dev, step.data(), 8 * 4, cudaMemcpyHostToDevice);
    kv_gather_nf4_step(k_dev, v_dev, table_dev, ids_dev, step_dev, n_ids, kShapes, ks_dev, vs_dev, nullptr);
    cudaDeviceSynchronize();
    std::vector<uint16_t> ks((size_t) n_ids * nvk), vs_g((size_t) n_ids * nvk);
    cudaMemcpy(ks.data(), ks_dev, ks.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(vs_g.data(), vs_dev, vs_g.size() * 2, cudaMemcpyDeviceToHost);
    size_t gather_bad = 0;
    for (int64_t id = 0; id < n_ids; ++id)
        for (int h = 0; h < kShapes.n_head_kv; ++h)
            for (int d = 0; d < kShapes.head_dim; ++d) {
                const int64_t c = ids[(size_t) id];
                const size_t row = (size_t) ((c / kShapes.page_size) * kShapes.n_head_kv + h) * kShapes.page_size + (size_t) (c % kShapes.page_size);
                const block_nf4* kb = reinterpret_cast<const block_nf4*>(k_host.data() + row * sizeof(block_nf4));
                float ref16[16];
                nf4_dequant_group16(kb->qs + (d / 16) * 8, kb->sc[d / 16], ref16);
                const float got = f32_from_f16(ks[((size_t) id * kShapes.n_head_kv + (size_t) h) * kShapes.head_dim + (size_t) d]);
                if (got != ref16[d % 16]) ++gather_bad;
            }
    CK(gather_bad == 0);
    std::printf("  %-44s %zu of %lld values differ\n", "gather vs host decode", gather_bad, (long long) (n_ids * kShapes.n_head_kv * kShapes.head_dim));
    {
        const int64_t rows = 8;
        std::vector<float> q((size_t) rows * 256), k((size_t) rows * 256);
        seed(q, 2.5);
        seed(k, 7.25);
        float *q_dev = nullptr, *k_dev2 = nullptr, *qh_dev = nullptr, *kh_dev = nullptr;
        cudaMalloc(&q_dev, q.size() * 4); cudaMalloc(&k_dev2, k.size() * 4);
        cudaMalloc(&qh_dev, q.size() * 4); cudaMalloc(&kh_dev, k.size() * 4);
        cudaMemcpy(q_dev, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(k_dev2, k.data(), k.size() * 4, cudaMemcpyHostToDevice);
        fwht256_cuda(q_dev, qh_dev, rows, nullptr);
        fwht256_cuda(k_dev2, kh_dev, rows, nullptr);
        std::vector<float> qh(q.size()), kh(k.size());
        cudaMemcpy(qh.data(), qh_dev, qh.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(kh.data(), kh_dev, kh.size() * 4, cudaMemcpyDeviceToHost);
        double worst_ip = 0.0;
        for (int64_t r = 0; r < rows; ++r) {
            double ip = 0.0, iph = 0.0;
            for (int d = 0; d < 256; ++d) {
                ip += (double) q[(size_t) r * 256 + d] * (double) k[(size_t) r * 256 + d];
                iph += (double) qh[(size_t) r * 256 + d] * (double) kh[(size_t) r * 256 + d];
            }
            const double rel = std::fabs(ip - iph) / (std::fabs(ip) + 1e-9);
            if (rel > worst_ip) worst_ip = rel;
        }
        std::printf("  %-44s %.3e worst\n", "<Hq, Hk> vs <q, k>", worst_ip);
        CK(worst_ip < 1e-5);
        cudaFree(q_dev); cudaFree(k_dev2); cudaFree(qh_dev); cudaFree(kh_dev);
    }

// ---- 6. the STREAMED prompt shape: batch append into host+stage with a page table of -1, then
//     qsa_decode_attn_batch over the STAGING pool vs the same cells in an fp16 pool (the exact
//     sequence a --kv-resident layer runs: nothing resident, everything staged).
{
    const int64_t T = 64;   // the pools above are sized for cells=64: exactly 16 pages
    const int64_t pages = cells / kShapes.page_size;
    std::vector<int32_t> none((size_t) pages, -1);   // nothing resident
    int32_t* none_dev = nullptr;
    cudaMalloc(&none_dev, none.size() * 4);
    cudaMemcpy(none_dev, none.data(), none.size() * 4, cudaMemcpyHostToDevice);
    const size_t hbytes = (size_t) cells * kShapes.n_head_kv * kv_nf4_bytes_per_head((int) kShapes.head_dim);
    uint8_t *st_k = nullptr, *st_v = nullptr, *h_k = nullptr, *h_v = nullptr;
    uint16_t *fp_k = nullptr, *fp_v = nullptr;
    cudaMalloc(&st_k, hbytes); cudaMalloc(&st_v, hbytes);
    cudaMalloc(&fp_k, (size_t) cells * nvk * 2); cudaMalloc(&fp_v, (size_t) cells * nvk * 2);
    if (cudaHostAlloc((void**) &h_k, hbytes, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc((void**) &h_v, hbytes, cudaHostAllocMapped) != cudaSuccess) {
        std::printf("  (mapped host alloc failed; skipping the streamed shape)\n");
    } else {
        uint8_t *h_kd = nullptr, *h_vd = nullptr;
        cudaHostGetDevicePointer((void**) &h_kd, h_k, 0);
        cudaHostGetDevicePointer((void**) &h_vd, h_v, 0);
        std::memset(h_k, 0, hbytes); std::memset(h_v, 0, hbytes);
        KvHostPools host; host.k_nf4 = h_kd; host.v_nf4 = h_vd;
        KvHostPools stage; stage.k_nf4 = st_k; stage.v_nf4 = st_v;
        kv_append_nf4(st_k, st_v, none_dev, 0, T, kcur_dev, vcur_dev, kShapes, nullptr, &host, &stage);
        cudaDeviceSynchronize();
        // the fp16 reference: the same cells, dequantized through the shared codec
        std::vector<float> kf((size_t) T * nvk), vf((size_t) T * nvk);
        std::vector<uint8_t> hk(hbytes), hv(hbytes);
        cudaMemcpy(hk.data(), h_kd, hbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hv.data(), h_vd, hbytes, cudaMemcpyDeviceToHost);
        for (int64_t c = 0; c < T; ++c)
            for (int h = 0; h < kShapes.n_head_kv; ++h) {
                const size_t row = (size_t) ((c / kShapes.page_size) * kShapes.n_head_kv + h) * kShapes.page_size + (size_t) (c % kShapes.page_size);
                const size_t base = row * sizeof(block_nf4);
                for (int g = 0; g < 16; ++g) {
                    float dec[16];
                    nf4_dequant_group16(hk.data() + base + (size_t) g * 8, hk[base + 128 + (size_t) g], dec);
                    for (int i = 0; i < 16; ++i) kf[(size_t) c * nvk + (size_t) h * kShapes.head_dim + (size_t) g * 16 + (size_t) i] = dec[i];
                    nf4_dequant_group16(hv.data() + base + (size_t) g * 8, hv[base + 128 + (size_t) g], dec);
                    for (int i = 0; i < 16; ++i) vf[(size_t) c * nvk + (size_t) h * kShapes.head_dim + (size_t) g * 16 + (size_t) i] = dec[i];
                }
            }
        std::vector<uint16_t> kf16(kf.size()), vf16(vf.size());
        for (size_t i = 0; i < kf.size(); ++i) kf16[i] = f16_from_f32(kf[i]);
        for (size_t i = 0; i < vf.size(); ++i) vf16[i] = f16_from_f32(vf[i]);
        cudaMemcpy(fp_k, kf16.data(), kf16.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(fp_v, vf16.data(), vf16.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(table_dev, table.data(), table.size() * 4, cudaMemcpyHostToDevice);   // identity for both
        // one query over all T cells, both pool kinds, compare the attention output
        std::vector<float> q((size_t) kShapes.n_head * kShapes.head_dim), q_dev_h(q.size());
        seed(q, 9.5);
        float* q_dev = nullptr;
        cudaMalloc(&q_dev, q.size() * 4);
        cudaMemcpy(q_dev, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
        std::vector<int32_t> all_ids((size_t) T);
        for (int64_t i = 0; i < T; ++i) all_ids[(size_t) i] = (int32_t) i;
        int32_t* ids2 = nullptr;
        cudaMalloc(&ids2, all_ids.size() * 4);
        cudaMemcpy(ids2, all_ids.data(), all_ids.size() * 4, cudaMemcpyHostToDevice);
        step[kStepPos] = 0; step[kStepWidth] = (int32_t) T;
        cudaMemcpy(step_dev, step.data(), 8 * 4, cudaMemcpyHostToDevice);
        QsaAttnPools pn; pn.k_nf4 = st_k; pn.v_nf4 = st_v; pn.page_table = table_dev;
        QsaAttnPools pf; pf.k_pool = fp_k; pf.v_pool = fp_v; pf.page_table = table_dev;
        const int64_t sc = qsa_decode_attn_scratch_floats(T, kShapes);
        float *sc1 = nullptr, *sc2 = nullptr, *a1 = nullptr, *a2 = nullptr;
        cudaMalloc(&sc1, (size_t) sc * 4); cudaMalloc(&sc2, (size_t) sc * 4);
        cudaMalloc(&a1, q.size() * 4); cudaMalloc(&a2, q.size() * 4);
        qsa_decode_attn_step(q_dev, pn, ids2, step_dev, T, kShapes, sc1, a1, nullptr);
        qsa_decode_attn_step(q_dev, pf, ids2, step_dev, T, kShapes, sc2, a2, nullptr);
        cudaDeviceSynchronize();
        q_dev_h.assign(q.size(), 0.0f);
        std::vector<float> o1(q.size()), o2(q.size());
        cudaMemcpy(o1.data(), a1, o1.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o2.data(), a2, o2.size() * 4, cudaMemcpyDeviceToHost);
        // nf4 rounds K and V to 4 bits: the softmax reweights, so compare each head's DIRECTION and
        // magnitude (cosine + norm), not element-wise ratios
        size_t bad_attn = 0;
        double worst_cos = 1.0, worst_norm = 0.0;
        for (int h = 0; h < 24; ++h) {
            double dot = 0, n1 = 0, n2 = 0;
            for (int d = 0; d < 256; ++d) {
                const double a = (double) o1[(size_t) h * 256 + d], b = (double) o2[(size_t) h * 256 + d];
                dot += a * b; n1 += a * a; n2 += b * b;
            }
            const double cos = dot / (std::sqrt(n1) * std::sqrt(n2) + 1e-12);
            const double nrel = std::fabs(std::sqrt(n1) - std::sqrt(n2)) / (std::sqrt(n2) + 1e-12);
            if (cos < worst_cos) worst_cos = cos;
            if (nrel > worst_norm) worst_norm = nrel;
            if (cos < 0.98 || nrel > 0.08) ++bad_attn;
        }
        std::printf("  %-44s cos %.5f worst, norm %.3f worst\n", "staged nf4 attention vs fp16 (per head)",
                    worst_cos, worst_norm);
        for (size_t i = 0; i < 6 && i < o1.size(); ++i)
            std::printf("    i=%zu nf4=%.4g fp16=%.4g\n", i, (double) o1[i], (double) o2[i]);
        CK(bad_attn == 0);
        CK(std::isfinite(worst_cos) && std::isfinite(worst_norm));
        cudaFree(q_dev); cudaFree(ids2); cudaFree(sc1); cudaFree(sc2); cudaFree(a1); cudaFree(a2);
        cudaFreeHost(h_k); cudaFreeHost(h_v);
    }
    cudaFree(none_dev); cudaFree(st_k); cudaFree(st_v); cudaFree(fp_k); cudaFree(fp_v);
}
    cudaFree(k_dev); cudaFree(v_dev); cudaFree(k2_dev); cudaFree(v2_dev);
    cudaFree(table_dev); cudaFree(step_dev); cudaFree(steps_dev); cudaFree(kcur_dev); cudaFree(vcur_dev);
    cudaFree(ks_dev); cudaFree(vs_dev); cudaFree(ids_dev);
    return bad;
}
}  // namespace
int main() {
    const int n = run();
    if (n == 0) std::printf("kv_nf4_parity: OK\n");
    return n == 0 ? 0 : 1;
}
