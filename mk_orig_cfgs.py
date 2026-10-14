import json

base = json.load(open('test-eq-nf4m6.json'))
for kv, port in (('fp16', 8261), ('int8', 8262)):
    cfg = dict(base)
    a = list(base['args'])
    out = []
    skip = False
    for x in a:
        if skip:
            skip = False
            continue
        if x == '--vram-reserve-mib':
            skip = True
            continue
        out.append(x)
    for j, x in enumerate(out):
        if x == '--kv':
            out[j + 1] = kv
    cfg['args'] = out
    cfg['exe'] = r'C:\Users\user\Strata\engine\strata.exe'
    cfg['port'] = port
    cfg['log'] = f'C:/Users/user/strata-win-mod/qa-orig-{kv}.log'
    json.dump(cfg, open(f'test-orig-{kv}.json', 'w'), indent=1)
    print(kv, 'ok')
