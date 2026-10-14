"""A/B against the ORIGINAL engine (C:/Users/user/Strata/engine/strata.exe, base 7695859a):
original fp16 (default) and original int8 vs our fork's nvfp4 mode 6, same prompts, same session."""
import json
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

def run_leg(tag, config, port):
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', config, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod')
    try:
        if not wait_ready(port):
            print(f'[{tag}] FAIL: never ready')
            return
        time.sleep(5)
        for label, notes in (('prefill14K', 410), ('prefill32K', 950)):
            usage, dt = chat_raw(port, prompt_of(notes, tag), 1)
            n = usage.get('prompt_tokens', 0)
            print(f'[{tag}] {label}: {n} tokens in {dt:.1f}s = {n / dt:.0f} tok/s')
        for pr in ('Write a short story about a lighthouse keeper.',
                   'Explain how tides work, then list seven sea birds.',
                   'Describe a busy market street at dusk, one long paragraph.'):
            usage, dt = chat_raw(port, pr, 260)
            g = usage.get('completion_tokens', 260)
            print(f'[{tag}] decode: {g} tokens in {dt:.1f}s = {g / dt:.1f} tok/s')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

def main():
    for tag, cfg, port in (('orig-fp16', 'test-orig-fp16.json', 8261),
                           ('orig-int8', 'test-orig-int8.json', 8262),
                           ('fork-nf4m6', 'test-eq-nf4m6.json', 8255)):
        run_leg(tag, cfg, port)
    return 0

if __name__ == '__main__':
    sys.exit(main())
