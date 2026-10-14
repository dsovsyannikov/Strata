import json

base = json.load(open('test-config-ram.json'))          # int8 base, no vram-reserve
args = base['args']

def set_arg(args, name, value):
    for j, a in enumerate(args):
        if a == name:
            args[j + 1] = value
            return
    args.extend([name, value])

for tag, kv, port in (('int8', 'int8', 8253), ('nf4m5', 'nvfp4', 8254), ('nf4m6', 'nvfp4', 8255)):
    cfg = dict(base)
    a = list(args)
    set_arg(a, '--kv', kv)
    set_arg(a, '--kv-resident', '65536')
    set_arg(a, '--max-context', '65536')
    if '--vram-reserve-mib' not in a:
        a += ['--vram-reserve-mib', '2048']
    cfg['args'] = a
    cfg['port'] = port
    cfg['log'] = f'C:/Users/user/strata-win-mod/qa-eq-{tag}.log'
    json.dump(cfg, open(f'test-eq-{tag}.json', 'w'), indent=1)
    print(tag, 'written')
