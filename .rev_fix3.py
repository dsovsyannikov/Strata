import io

p = 'src/kernels/cuda/qsa_decode_attn.cu'
s = io.open(p, encoding='utf-8', newline='').read().replace(chr(13) + chr(10), chr(10))
decls = ('    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);' + chr(10) +
         '    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);' + chr(10))
assert s.count(decls) == 1, s.count(decls)
s = s.replace(decls, '', 1)
anchor = '    const bool nf4_rows = pools.k_nf4 != nullptr;'
assert s.count(anchor) == 1
s = s.replace(anchor, decls + anchor, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('decls moved above nf4 block')
