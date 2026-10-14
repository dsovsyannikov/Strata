import json

base = json.load(open('test-config-ram.json'))   # fork exe, int8, 32768/262144, no reserve
base['log'] = 'C:/Users/user/strata-win-mod/qa-exp-fint8.log'
base['port'] = 8272
json.dump(base, open('test-exp-fint8.json', 'w'), indent=1)

nf4 = dict(base)
nf4['args'] = [x for x in base['args']]
for j, x in enumerate(nf4['args']):
    if x == '--kv':
        nf4['args'][j + 1] = 'nvfp4'
nf4['log'] = 'C:/Users/user/strata-win-mod/qa-exp-fnf4.log'
nf4['port'] = 8273
json.dump(nf4, open('test-exp-fnf4.json', 'w'), indent=1)

orig = dict(base)
orig['exe'] = r'C:\Users\user\Strata\engine\strata.exe'
orig['log'] = 'C:/Users/user/strata-win-mod/qa-exp-orig.log'
orig['port'] = 8271
json.dump(orig, open('test-exp-orig.json', 'w'), indent=1)
print('configs ok')
