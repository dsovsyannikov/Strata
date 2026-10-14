// include/strata/kernels/gdn_state_io.cuh - the recurrent state's load/store pair, templated on its width.
//
// The kernels keep their arithmetic in F32 whatever `gdn_state_f16()` says; these two are the only places an
// element narrows.  With `ST = float` both compile away, so the F32 path keeps its exact bits.
#pragma once

#include <cuda_fp16.h>

namespace strata::kernels {

__device__ __forceinline__ float gdn_st_get(float v) { return v; }
__device__ __forceinline__ float gdn_st_get(__half v) { return __half2float(v); }
__device__ __forceinline__ void gdn_st_put(float& p, float v) { p = v; }
__device__ __forceinline__ void gdn_st_put(__half& p, float v) { p = __float2half(v); }

}  // namespace strata::kernels
