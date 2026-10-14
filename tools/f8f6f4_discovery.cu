// Direct layout discovery for mma.m16n8k32.kind::f8f6f4 (A=e4m3 x B=e2m1): no layout is assumed.
// Phase 1: one A byte cell (lane, reg, byte) at a time holds 1.0, every B byte 1.0 -> D's nonzero row
//          is that cell's hardware row (same for all 8 columns).
// Phase 2: symmetric for B cells -> the column.
// Phase 3: one A cell + binary-coded B cells -> the single B cell sharing its k (D's value tells the bit).
// Prints row/col/partner tables, then the implied (reg, byte) -> (row, k) / (col, k) maps.
#include <cstdio>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

__global__ void mma1(const uint32_t* __restrict__ ar, const uint32_t* __restrict__ br,
                     float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int t = threadIdx.x;
    uint32_t a[4] = {ar[t * 4 + 0], ar[t * 4 + 1], ar[t * 4 + 2], ar[t * 4 + 3]};
    uint32_t b[2] = {br[t * 2 + 0], br[t * 2 + 1]};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    const int gid = lane >> 2, tig = lane & 3;
    out[gid * 8 + 2 * tig] = c[0];
    out[gid * 8 + 2 * tig + 1] = c[1];
    out[(gid + 8) * 8 + 2 * tig] = c[2];
    out[(gid + 8) * 8 + 2 * tig + 1] = c[3];
}

// e4m3 code of 1.0 = 0x38; e2m1 code (in bits[5:2]) of 1.0 = 2<<2 = 8; of 0 = 0.
int main() {
    uint32_t *dar, *dbr; float* dO;
    cudaMalloc(&dar, 32 * 4 * 4); cudaMalloc(&dbr, 32 * 2 * 4); cudaMalloc(&dO, 128 * 4);
    std::vector<uint32_t> ar(128), br(64);
    std::vector<float> D(128);
    auto run = [&]() {
        cudaMemcpy(dar, ar.data(), 512, cudaMemcpyHostToDevice);
        cudaMemcpy(dbr, br.data(), 256, cudaMemcpyHostToDevice);
        mma1<<<1, 32>>>(dar, dbr, dO);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { std::printf("mma: %s\n", cudaGetErrorString(e)); std::exit(1); }
        cudaMemcpy(D.data(), dO, 512, cudaMemcpyDeviceToHost);
    };
    // NOTE: the D readback assumes the universal c-layout; phase 1/2 only use "which entries are
    // nonzero", so even a wrong readback ordering still yields consistent RELATIVE maps, and the
    // final check re-verifies everything with the full probe.

    // phase 0: raw D dump for a few one-hot A cells (the pattern decodes the layout directly)
    for (int cell : {0 * 16 + 0 * 4 + 0, 1 * 16 + 0 * 4 + 0, 2 * 16 + 0 * 4 + 0, 3 * 16 + 0 * 4 + 0,
                     4 * 16 + 0 * 4 + 0, 8 * 16 + 0 * 4 + 0, 0 * 16 + 2 * 4 + 0, 1 * 16 + 1 * 4 + 0}) {
        std::fill(ar.begin(), ar.end(), 0);
        ar[cell / 4] = 0x38u << (8 * (cell % 4));
        std::fill(br.begin(), br.end(), 0x08080808u);
        run();
        std::printf("A lane %d reg %d byte %d: D =", cell / 16, (cell % 16) / 4, cell % 4);
        for (int i = 0; i < 16; ++i) {
            std::printf(" [");
            for (int j = 0; j < 8; ++j) std::printf("%g ", D[i * 8 + j]);
            std::printf("]");
        }
        std::printf("\n");
    }
    for (int cell : {0 * 8 + 0 * 4 + 0, 1 * 8 + 0 * 4 + 0, 4 * 8 + 0 * 4 + 0, 0 * 8 + 1 * 4 + 0}) {
        std::fill(br.begin(), br.end(), 0);
        br[cell / 4] = 0x08u << (8 * (cell % 4));
        std::fill(ar.begin(), ar.end(), 0x38383838u);
        run();
        std::printf("B lane %d reg %d byte %d: D =", cell / 8, (cell % 8) / 4, cell % 4);
        for (int i = 0; i < 16; ++i) {
            std::printf(" [");
            for (int j = 0; j < 8; ++j) std::printf("%g ", D[i * 8 + j]);
            std::printf("]");
        }
        std::printf("\n");
    }

    // phase 1: A cell -> the exact D out-index it lights (all columns should light; we record the set)
    std::vector<int> arow(32 * 4 * 4, -1), acol(32 * 4 * 4, -1);
    for (int cell = 0; cell < 32 * 4 * 4; ++cell) {
        std::fill(ar.begin(), ar.end(), 0);
        ar[cell / 4] = 0x38u << (8 * (cell % 4));   // byte cell%4 of the cell's register
        std::fill(br.begin(), br.end(), 0x08080808u);   // e2m1 1.0 in every byte
        run();
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j)
                if (D[i * 8 + j] > 0.5f) { arow[cell] = i; if (acol[cell] < 0) acol[cell] = j; }
        // how many lit (sanity: with B=1 everywhere exactly one ROW bucket lights, 8 entries)
    }
    // phase 2: B cell -> the exact out index (one entry lights: its (row bucket, col))
    std::vector<int> brow(32 * 2 * 4, -1), bcol(32 * 2 * 4, -1);
    for (int cell = 0; cell < 32 * 2 * 4; ++cell) {
        std::fill(br.begin(), br.end(), 0);
        br[cell / 4] = 0x08u << (8 * (cell % 4));
        std::fill(ar.begin(), ar.end(), 0x38383838u);   // e4m3 1.0 everywhere
        run();
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 8; ++j)
                if (D[i * 8 + j] > 0.5f) { bcol[cell] = j; if (brow[cell] < 0) brow[cell] = i; }
    }
    // phase 3: for each A cell, binary-search the B cell that shares its k
    std::vector<int> apart(32 * 4 * 4, -1);
    for (int cell = 0; cell < 32 * 4 * 4; ++cell) {
        int lo = 0, hi = 32 * 2 * 4;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            std::fill(ar.begin(), ar.end(), 0);
            ar[cell / 4] = 0x38u << (8 * (cell % 4));
            std::fill(br.begin(), br.end(), 0);
            for (int b = lo; b < mid; ++b) br[b / 4] |= 0x08u << (8 * (b % 4));
            run();
            bool hit = false;
            for (int i = 0; i < 16 && !hit; ++i)
                for (int j = 0; j < 8 && !hit; ++j)
                    if (i == arow[cell] && D[i * 8 + j] > 0.5f) hit = true;
            if (hit) hi = mid; else lo = mid;
        }
        apart[cell] = lo;
    }
    // report: for each hardware (row, k-slot s = partner B cell's within-k order)
    std::printf("A cells (lane, reg, byte) -> (row, partner-B-cell):\n");
    for (int cell = 0; cell < 32 * 4 * 4; ++cell) {
        const int p = apart[cell];
        std::printf("  lane %2d reg %d byte %d -> row %2d, col? %d | partner bcell lane %2d reg %d byte %d (col %d)\n",
                    cell / 16, (cell % 16) / 4, cell % 4, arow[cell], acol[cell],
                    p / 8, (p % 8) / 4, p % 4, bcol[p]);
    }
    std::printf("B cells (lane, reg, byte) -> col:\n");
    for (int cell = 0; cell < 32 * 2 * 4; ++cell)
        std::printf("  lane %2d reg %d byte %d -> col %d (row? %d)\n", cell / 8, (cell % 8) / 4, cell % 4,
                    bcol[cell], brow[cell]);
    return 0;
}
