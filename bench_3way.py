"""Equal-config 3-way A/B: int8 vs nvfp4-mode5 vs nvfp4-mode6 (identical everything else).
One 14K and one 32K prefill per engine (unique texts, no cache hits), decode 220."""
import json
import os
import subprocess
import sys
import time
import urllib.request

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

def chat_raw(port, prompt, max_tokens):
    body = json.dumps({'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': max_tokens, 'temperature': 0,
                       'chat_template_kwargs': {'enable_thinking': False}}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out.get('usage', {}), time.time() - t0

def prompt_of(notes, variant):
    words = []
    for i in range(1, notes + 1):
        words.append(f"Note {i}: the ledger lists item {i * 7} with value {i % 97}; "
                     f"the clerk records every entry and the archive keeps pages in order {variant}.")
    return " ".join(words)

def run_leg(tag, config, port, extra_env):
    env = dict(os.environ)
    env.update(extra_env)
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', config, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod', env=env)
    try:
        if not wait_ready(port):
            print(f'[{tag}] FAIL: never ready')
            return
        time.sleep(5)
        for label, notes in (('prefill14K', 410), ('prefill32K', 950)):
            usage, dt = chat_raw(port, prompt_of(notes, tag), 1)
            n = usage.get('prompt_tokens', 0)
            print(f'[{tag}] {label}: {n} tokens in {dt:.1f}s = {n / dt:.0f} tok/s')
        usage, dt = chat_raw(port, 'Write a short story about a lighthouse keeper.', 220)
        gen = usage.get('completion_tokens', 220)
        print(f'[{tag}] decode: {gen} tokens in {dt:.1f}s = {gen / dt:.1f} tok/s')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

def main():
    legs = [('int8', 'test-eq-int8.json', 8253, {}),
            ('nf4m5', 'test-eq-nf4m5.json', 8254, {'STRATA_PROMPT_ATTN_NF4': '0'}),
            ('nf4m6', 'test-eq-nf4m6.json', 8255, {})]
    if len(sys.argv) > 1:
        want = sys.argv[1].split(',')
        legs = [l for l in legs if l[0] in want]
    for tag, cfg, port, env in legs:
        run_leg(tag, cfg, port, env)
    return 0

if __name__ == '__main__':
    sys.exit(main())
