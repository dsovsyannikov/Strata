"""Full-power engine A/B: nvfp4 (mode 6) vs int8, three prefill runs each + decode,
same configs, one engine at a time. Reports each run and the median."""
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

def prompt_of(notes):
    words = []
    for i in range(1, notes + 1):
        words.append(f"Note {i}: the ledger lists item {i * 7} with value {i % 97}; "
                     f"the clerk records every entry and the archive keeps the pages in order.")
    return " ".join(words)

P14K = prompt_of(410)
P32K = prompt_of(950)

def run_leg(tag, config, port):
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', config, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod')
    try:
        if not wait_ready(port):
            print(f'[{tag}] FAIL: never ready')
            return
        time.sleep(5)
        for label, prompt, reps in (('prefill14K', P14K, 3), ('prefill32K', P32K, 2)):
            toks = []
            for r in range(reps):
                usage, dt = chat_raw(port, prompt, 1)
                n = usage.get('prompt_tokens', 0)
                toks.append(n / dt if dt > 0 else 0)
                print(f'[{tag}] {label} run{r + 1}: {n} tokens in {dt:.1f}s = {n / dt:.0f} tok/s')
            toks.sort()
            print(f'[{tag}] {label} median: {toks[len(toks) // 2]:.0f} tok/s')
        # decode: short prompt, 220 tokens
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
    legs = sys.argv[1:] or ['nf4', 'int8']
    for tag in legs:
        cfg = 'test-config-ram-nvfp4.json' if tag == 'nf4' else 'test-config-ram-int8.json'
        port = 8241 if tag == 'nf4' else 8242
        run_leg(tag, cfg, port)
    return 0

if __name__ == '__main__':
    sys.exit(main())
