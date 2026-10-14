import io, re

def read(p):
    return io.open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')
def write(p, s):
    io.open(p, 'w', encoding='utf-8', newline='').write(s)

p = 'src/kernels/cuda/qsa_decode_attn.cu'
s = read(p)
ms = list(re.finditer(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S))
assert len(ms) == 3, len(ms)

# --- #0: upstream's helpers, with attn_kv_mode extended to the nf4 pools (mode 4), + our batch head using it
head0 = ms[0].group(1)
head0 = head0.replace(
    'return p.k_q4 != nullptr ? 2 : (p.k_q != nullptr && p.v_q4 != nullptr ? 3 : (p.k_q != nullptr ? 1 : 0));',
    'return p.k_nf4 != nullptr ? 4 : (p.k_q4 != nullptr ? 2 : (p.k_q != nullptr && p.v_q4 != nullptr ? 3 : (p.k_q != nullptr ? 1 : 0)));')
ours0 = ms[0].group(2)
ours0 = ours0.replace(
    '''    const int kv_mode = pools.k_nf4 != nullptr ? 4 : (pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0)));''',
    '''    const int kv_mode = attn_kv_mode(pools);   // 4 = --kv nvfp4 (k_nf4 pools)''')
s = s[:ms[0].start()] + head0 + ours0 + s[ms[0].end():]

# --- #1 (batch path) and #2 (single-query): rewrite via one regex each, v3 branch + mode-4 else-if
m1 = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m1 and 'kernel < 0' in m1.group(1)
merged1 = '''#if !defined(__HIPCC__)
    if (kernel < 0 && kv_mode == 1 && attn_v3_on())   // (the parity harness names its kernel: never the environment)
        attn_chunk_q8_v3_kernel<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else if (kv_mode == 4)
        STRATA_ATTN_CHUNK(4)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
'''
s = s[:m1.start()] + merged1 + s[m1.end():]

m2 = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m2 and 'attn_v3_on()' in m2.group(1) and 'ids, step' in m2.group(1)
merged2 = '''#if !defined(__HIPCC__)
    if (kv_mode == 1 && attn_v3_on())
        attn_chunk_q8_v3_kernel<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else if (kv_mode == 4)
        STRATA_ATTN_CHUNK(4)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else
#endif
    if (kv_mode == 3)
'''
s = s[:m2.start()] + merged2 + s[m2.end():]

assert s.count('<<<<<<<') == 0, 'markers left'
write(p, s)
print('qsa_decode_attn.cu resolved,', len(s.splitlines()), 'lines')
