import io, re

def read(p):
    return io.open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')
def write(p, s):
    io.open(p, 'w', encoding='utf-8', newline='').write(s)

p = 'src/kernels/qsa_prompt_attn_parity.cpp'
s = read(p)
ms = list(re.finditer(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S))
assert len(ms) == 3

# #0: our qref block, with err_im added to the counters
ours0 = ms[0].group(2).replace(
    'double err_old = 0, err_new = 0, ref_scale = 0, err_dump = 0;',
    'double err_old = 0, err_new = 0, err_im = 0, ref_scale = 0, err_dump = 0;')
s = s[:ms[0].start()] + ours0 + s[ms[0].end():]

# #1: keep both trackers (upstream's IMMA bound and our dump tracker)
m1 = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
both1 = m1.group(1) + m1.group(2)
s = s[:m1.start()] + both1 + s[m1.end():]

# #2: upstream's IMMA block, with our fmt-4 gate and nvfp4 in the name
m2 = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
head2 = m2.group(1)
head2 = head2.replace(
    'const bool ok2 = diff <= 1e-4 * scale;',
    '''// fmt 4's new kernel quantizes Q to FP4 (hi + lo): against the raw-Q old kernel the gap is that
    // approximation by design, so the gate is 5% of the scale (the printed number tracks it)
    const bool ok2 = diff <= (fmt == 4 ? 5e-2 : 1e-4) * scale;''')
head2 = head2.replace(
    'fmt == 2 ? "q4_0" : fmt == 1 ? "int8" : "fp16"',
    'fmt == 4 ? "nvfp4" : fmt == 2 ? "q4_0" : fmt == 1 ? "int8" : "fp16"')
s = s[:m2.start()] + head2 + s[m2.end():]

assert s.count('<<<<<<<') == 0, 'markers left'
write(p, s)
print('qsa_prompt_attn_parity.cpp resolved')
