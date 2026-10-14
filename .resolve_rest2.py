import io, re

def read(p):
    return io.open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')
def write(p, s):
    io.open(p, 'w', encoding='utf-8', newline='').write(s)

# ---------- fused_gdn.cu ----------
p = 'src/kernels/cuda/fused_gdn.cu'
s = read(p)
s = re.sub(r'<<<<<<< HEAD\n#include "strata/core/geometry.hpp"\n=======\n(#include "strata/kernels/gdn_state.hpp"\n#include "strata/kernels/gdn_state_io.cuh"\n)>>>>>>> [^\n]*\n',
           '#include "strata/core/geometry.hpp"\n\\1', s, count=1)
s = re.sub(r'<<<<<<< HEAD\n((?://[^\n]*\n)*)template <bool SILU>\n__global__ void __launch_bounds__\(S \* RG\) gdn_step_norm_kernel\(float\* __restrict__ state, const float\* __restrict__ q,\n=======\ntemplate <typename ST>\n__global__ void __launch_bounds__\(S \* RG\) gdn_step_norm_kernel\(ST\* __restrict__ state, const float\* __restrict__ q,\n>>>>>>> [^\n]*\n',
           '\\1template <bool SILU, typename ST>\n__global__ void __launch_bounds__(S * RG) gdn_step_norm_kernel(ST* __restrict__ state, const float* __restrict__ q,\n', s, count=1)
m = re.search(r'<<<<<<< HEAD\n(namespace \{\ntemplate <bool SILU>\nvoid step_norm\(float\* state.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m, 'fused #2'
head = m.group(1).replace('template <bool SILU>\nvoid step_norm(float* state',
                          'template <bool SILU, typename ST>\nvoid step_norm(ST* state')
head = head.replace('gdn_step_norm_kernel<SILU><<<', 'gdn_step_norm_kernel<SILU, ST><<<')
s = s[:m.start()] + head + s[m.end():]
mw = re.search(r'void fused_gdn_step_norm\(float\* state,.*?\n\}\n', s, re.S)
assert mw, 'public wrapper'
newpub = '''void fused_gdn_step_norm(void* state_v, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (strata::kernels::gdn_state_f16()) step_norm<false, __half>((__half*) state_v, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
    else step_norm<false, float>((float*) state_v, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
}

void fused_gdn_step_norm_silu(void* state_v, const float* q, const float* k, const float* v, const float* gate,
                              const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k,
                              int h_v, void* stream) {
    if (strata::kernels::gdn_state_f16()) step_norm<true, __half>((__half*) state_v, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
    else step_norm<true, float>((float*) state_v, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, stream);
}
'''
s = s[:mw.start()] + newpub + s[mw.end():]
assert '<<<<<<<' not in s, 'fused markers left'
write(p, s)
print('fused_gdn.cu ok')

# ---------- native_gdn.cu ----------
p = 'src/kernels/cuda/native_gdn.cu'
s = read(p)
s = re.sub(r'<<<<<<< HEAD\n#include "strata/core/geometry.hpp"\n=======\n(#include "strata/kernels/gdn_state.hpp"\n#include "strata/kernels/gdn_state_io.cuh"\n)>>>>>>> [^\n]*\n',
           '#include "strata/core/geometry.hpp"\n\\1', s, count=1)
assert '<<<<<<<' not in s
write(p, s)
print('native_gdn.cu ok')

# ---------- verify.cpp ----------
p = 'src/core/verify.cpp'
s = read(p)
m = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m and 'gdn_state.hpp' in m.group(2)
s = s[:m.start()] + m.group(1) + m.group(2) + s[m.end():]
assert '<<<<<<<' not in s
write(p, s)
print('verify.cpp ok')

# ---------- verify_kernels.hpp ----------
p = 'include/strata/kernels/verify_kernels.hpp'
s = read(p)
m = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m
head = m.group(1).replace('float* state[8];', 'void* state[8];')
s = s[:m.start()] + head + m.group(2) + s[m.end():]
assert '<<<<<<<' not in s
write(p, s)
print('verify_kernels.hpp ok')

# ---------- prefill/kernels.hpp ----------
p = 'include/strata/prefill/kernels.hpp'
s = read(p)
m = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m
head = m.group(1).replace('void gdn_recurrence_variant(int variant, float* state', 'void gdn_recurrence_variant(int variant, void* state')
s = s[:m.start()] + head + m.group(2) + s[m.end():]
s = s.replace('void gdn_recurrence(float* state,', 'void gdn_recurrence(void* state,')
s = s.replace('void gdn_recurrence_state_only(float* state,', 'void gdn_recurrence_state_only(void* state,')
assert '<<<<<<<' not in s
write(p, s)
print('prefill/kernels.hpp ok')

# ---------- conversation_state.cpp ----------
p = 'src/core/conversation_state.cpp'
s = read(p)
m = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m
merged = '''    const void* gdn_src = hit != nullptr ? hit->p : (const void*) c.gdn.data();
    if (c.gdn.size() == z.gdn) {
        if (!copy(ss.gdn_state, gdn_src, c.gdn.size(), error) ||
            !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    } else {
        // a checkpoint from the other state width (an F32 file into an F16 run or the reverse): the recurrent
        // part converts on the host, the conv history copies through unchanged
        std::vector<uint8_t> native(z.gdn);
        strata::kernels::gdn_state_convert_arena(native.data(), strata::kernels::gdn_state_f16(), gdn_src,
                                                  (size_t) ss.gdn_alloc,
                                                 (size_t) g.ssm_state_size * (size_t) g.ssm_v_heads *
                                                     (size_t) g.ssm_state_size,
                                                 (size_t) g.ssm_conv_channels * (size_t) (g.ssm_d_conv - 1));
        if (!copy(ss.gdn_state, native.data(), native.size(), error) ||
            !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    }
'''
s = s[:m.start()] + merged + s[m.end():]
assert '<<<<<<<' not in s
write(p, s)
print('conversation_state.cpp ok')

# ---------- DETAILS.md ----------
p = 'docs/DETAILS.md'
s = read(p)
m = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m
s = s[:m.start()] + m.group(1) + '\n' + m.group(2) + s[m.end():]
assert '<<<<<<<' not in s
write(p, s)
print('DETAILS.md ok')
