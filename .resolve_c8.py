import io, re

def read(p):
    return io.open(p, encoding='utf-8', newline='').read().replace('\r\n', '\n')
def write(p, s):
    io.open(p, 'w', encoding='utf-8', newline='').write(s)

# ---------- draft_policy.hpp: upstream members + our incumbent_ ----------
p = 'include/strata/spec/draft_policy.hpp'
s = read(p)
m = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m
merged = m.group(1) + '    int incumbent_ = 0;                                    // best_depth\'s last answer (0 = none yet)\n'
s = s[:m.start()] + merged + s[m.end():]
assert '<<<<<<<' not in s
write(p, s)
print('draft_policy.hpp ok')

# ---------- draft_policy_test.cpp: both test blocks ----------
p = 'src/spec/draft_policy_test.cpp'
s = read(p)
m = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m
s = s[:m.start()] + m.group(1) + '\n    {\n' + m.group(2) + '    }\n' if False else s[:m.start()] + m.group(1) + m.group(2) + s[m.end():]
assert '<<<<<<<' not in s
write(p, s)
print('draft_policy_test.cpp ok')

# ---------- generate.cpp ----------
p = 'src/program/generate.cpp'
s = read(p)
ms = list(re.finditer(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S))
assert len(ms) == 4

# #0: both (upstream spec_cost machinery + our spec_adapt flag)
s = s[:ms[0].start()] + ms[0].group(1) + ms[0].group(2) + s[ms[0].end():]

# #1: our adapt block + upstream's new round head (T=1/set_context/merged); drop our old T computation
m1 = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
ours1 = m1.group(2)
ours_adapt = '''                if (spec_adapt && use_mtp && !first_window) {   // STRATA_SPEC_ADAPT: the round's cap
                    S_mtp = policy.best_depth(S_mtp_cap);
                    mtp.set_max_drafts(S_mtp - 1);
                    ++depth_pick[std::min(S_mtp, 8)];
                }
'''
s = s[:m1.start()] + ours_adapt + m1.group(1) + s[m1.end():]

# #2: upstream deleted the declaration site; re-add ours right before its (moved) use.
# Find where S_mtp is now first used; the serve loop head 'const int S = o.spec;' is a stable anchor.
m2 = re.search(r'<<<<<<< HEAD\n\n=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
if m2:
    s = s[:m2.start()] + '\n' + s[m2.end():]
    # insert declarations after the 'const int S = o.spec;' line of the serve loop
    anchor = 'const int S = o.spec;'
    i = s.index(anchor) + len(anchor) + 1
    decl = '        const int S_mtp_cap = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP\'s windows; suffixes go up to S\n        int S_mtp = S_mtp_cap;   // STRATA_SPEC_ADAPT=1 narrows this per round (policy.best_depth)\n'
    s = s[:i] + decl + s[i:]
else:
    m2b = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
    s = s[:m2b.start()] + m2b.group(2) + s[m2b.end():]

# #3: serial loop: upstream retry label first, then our adapt block
m3 = re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', s, re.S)
assert m3 and 'retry_round' in m3.group(1)
s = s[:m3.start()] + m3.group(1) + m3.group(2) + s[m3.end():]

assert s.count('<<<<<<<') == 0, 'markers left'
write(p, s)
print('generate.cpp ok')
