"""Interleaved decode A/B: original-int8 vs fork-nf4m6, alternating legs to cancel thermal drift."""
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

def leg(tag, config, port):
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', config, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod')
    try:
        if not wait_ready(port):
            print(f'[{tag}] FAIL')
            return
        time.sleep(5)
        chat_raw(port, 'Warm up with one short sentence about the sea.', 60)   # every leg warm
        rates = []
        for pr in ('Explain how tides work, then list seven sea birds.',
                   'Describe a busy market street at dusk, one long paragraph.',
                   'Outline the plot of a heist movie set on a night train.'):
            usage, dt = chat_raw(port, pr, 260)
            g = usage.get('completion_tokens', 260)
            rates.append(g / dt)
            print(f'[{tag}] decode: {g} tokens in {dt:.1f}s = {g / dt:.1f} tok/s')
        rates.sort()
        print(f'[{tag}] median: {rates[1]:.1f} tok/s')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

def main():
    import sys as _s
    legs = {'orig': ('test-orig-int8.json', 8262), 'pristine': ('test-pristine.json', 8263),
            'forkint8': ('test-eq-int8.json', 8253), 'forknf4': ('test-eq-nf4m6.json', 8255), 'rel': ('test-rel.json', 8264)}
    for tag in _s.argv[1:]:
        cfg, port = legs[tag]
        leg(tag, cfg, port)
    return 0

if __name__ == '__main__':
    sys.exit(main())
