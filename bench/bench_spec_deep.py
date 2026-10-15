"""Deep spec-window A/B: --spec 4 vs --spec 8 vs --spec 8 + adaptive cap.

min_p 0.5 already self-limits window length by drafter confidence, so --spec 8 raises only the
ceiling; STRATA_SPEC_DEPTH=1 prints the per-depth accepted/tested table (does alpha hold at depths
4-7?) and the fully-accepted share. Three legs, two interleaved rounds with rotated order."""
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE_CFG = r'C:\Users\user\strata-win-mod\test-eq-nf4m6.json'
OUT = r'C:\Users\user\.zcode\workspace\default\spec-deep'
PROMPTS = [
    ('code', 'A{tag}. Write a Python function that queries the Grafana HTTP API for all dashboards in a folder '
             'and prints each panel title and its datasource.'),
    ('code2', 'D{tag}. Write a bash script that loops over a list of SSH hosts, checks disk usage on each over ssh '
              'and prints a warning for any mount above 80 percent. Start with: #!/usr/bin/env bash'),
    ('ru', 'C{tag}. Напиши короткий рассказ о смотрителе маяка на северном острове.'),
]


def wait_ready(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def chat(port, prompt):
    body = json.dumps({'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': 220, 'temperature': 0,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def run_leg(port, tag, spec, adapt, round_no, log):
    cfg = json.load(open(BASE_CFG, encoding='utf-8'))
    args = cfg['args']
    args[args.index('--spec') + 1] = str(spec)
    cfg['args'] = args
    leg_log = f'{OUT}-{tag}-r{round_no}.log'
    cfg['log'] = leg_log.replace('\\', '/')
    cfg_path = f'{OUT}-{tag}-r{round_no}.json'
    json.dump(cfg, open(cfg_path, 'w', encoding='utf-8'))
    env = dict(os.environ)
    env['STRATA_SPEC_ADAPT'] = '1' if adapt else '0'
    env['STRATA_SPEC_DEPTH'] = '1'
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', cfg_path, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod', env=env)
    try:
        if not wait_ready(port):
            print(f'[{tag}] FAIL never ready', file=log, flush=True)
            return
        time.sleep(5)
        for rep in ('x', 'y'):
            for name, p in PROMPTS:
                t0 = time.time()
                out = chat(port, p.format(tag=f'r{round_no}{rep}'))
                dt = time.time() - t0
                n = out.get('usage', {}).get('completion_tokens', 0)
                print(f'[{tag}] {name}{rep}: {n} tokens in {dt:.1f}s', file=log, flush=True)
                time.sleep(3)
        for _ in range(6):
            try:
                if open(leg_log, encoding='utf-8', errors='ignore').read().count('spec windows') >= 6:
                    break
            except OSError:
                pass
            time.sleep(5)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    log = open(f'{OUT}-summary.log', 'a', buffering=1, encoding='utf-8')
    print(f'=== {time.strftime("%H:%M:%S")} spec-deep A/B start ===', file=log)
    port = 8273
    legs = (('s4', 4, False), ('s8', 8, False), ('s8a', 8, True))
    orders = (legs, (legs[2], legs[1], legs[0]))   # round 2 rotated
    for round_no, order in enumerate(orders, 1):
        for tag, spec, adapt in order:
            print(f'--- round {round_no} leg {tag} (spec {spec}, adapt {adapt}) ---', file=log, flush=True)
            run_leg(port, tag, spec, adapt, round_no, log)
            time.sleep(5)
    print(f'=== done ===', file=log)
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
