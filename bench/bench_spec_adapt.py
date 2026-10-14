"""STRATA_SPEC_ADAPT A/B: two engine legs per round (adapt on/off), order swapped between rounds.

Each leg runs the python serve wrapper with its own config copy (its own log file), env carrying
STRATA_SPEC_ADAPT / STRATA_SPEC_DEPTH. Same three prompts per leg inside a round (round-tagged, so
the conversation cache never restores them). The engine's per-request lines give decode tok/s,
drafts accepted, the per-depth table, the fully-accepted window share and commit ms/window."""
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE_CFG = r'C:\Users\user\strata-win-mod\test-eq-nf4m6.json'
OUT = r'C:\Users\user\.zcode\workspace\default\spec-adapt'
PROMPTS = [
    ('code', 'A{tag}. Write a Python function that queries the Grafana HTTP API for all dashboards in a folder '
             'and prints each panel title and its datasource.'),
    ('story', 'B{tag}. Write a short story about a lighthouse keeper.'),
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


def run_leg(port, tag, adapt, round_no, log):
    cfg = json.load(open(BASE_CFG, encoding='utf-8'))
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
        # two passes over the prompts: enough stderr volume that the wrapper's log buffer flushes
        # (a hard terminate loses the unflushed tail - the per-request stats live there)
        for rep in ('x', 'y'):
            for name, p in PROMPTS:
                t0 = time.time()
                out = chat(port, p.format(tag=f'r{round_no}{rep}'))
                dt = time.time() - t0
                n = out.get('usage', {}).get('completion_tokens', 0)
                print(f'[{tag}] {name}{rep}: {n} tokens in {dt:.1f}s', file=log, flush=True)
                time.sleep(3)
        # wait for the last request's stats to reach the log file before the kill
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
    print(f'=== {time.strftime("%H:%M:%S")} spec-adapt A/B start ===', file=log)
    port = 8271
    for round_no, order in ((1, ((0, 'off'), (1, 'on'))), (2, ((1, 'on'), (0, 'off')))):
        for adapt, name in order:
            print(f'--- round {round_no} leg {name} ---', file=log, flush=True)
            run_leg(port, f'adapt{name}', bool(adapt), round_no, log)
            time.sleep(5)
    print(f'=== done ===', file=log)
    log.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
