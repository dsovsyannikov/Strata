"""End-to-end A/B of the native-FP4 prompt attention (mode 6) against int8 KV:
prefill speed on a ~16K-token prompt, decode speed, then the needle bench and a
factual battery.  One engine at a time; STRATA_PROMPT_ATTN_NF4=0 in the shell
runs the mode-5 control instead of mode 6.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, 'tools')
import needle_bench  # noqa: E402

FACTUAL = [
    ("Сколько будет 17 * 23? Ответь только числом.", ["391"]),
    ("Capital of Australia? One word.", ["canberra"]),
    ("What does this Python print, digits only: for i in range(3): print(i*i)", ["0 1 4"]),
    ("A train travels 60 km in 45 minutes. Speed in km/h, number only.", ["80"]),
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

def chat_raw(port, prompt, max_tokens):
    body = json.dumps({'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': max_tokens, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out['choices'][0]['message']['content'], out.get('usage', {}), time.time() - t0

def prompt_16k():
    words = []
    for i in range(1, 421):   # ~37 tokens per note -> ~15-16K tokens
        words.append(f"Note {i}: the ledger lists item {i * 7} with value {i % 97}; "
                     f"the clerk records every entry and the archive keeps the pages in order.")
    return " ".join(words)

def run_leg(tag, config, port):
    env = dict(os.environ)
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', config, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod', env=env)
    try:
        if not wait_ready(port):
            print(f'[{tag}] FAIL: server never ready')
            return 1
        # 1. prefill speed: one long prompt, one generated token
        _, usage, dt = chat_raw(port, prompt_16k(), 1)
        ptok = usage.get('prompt_tokens', 0)
        prefill = ptok / dt if dt > 0 else 0
        # 2. decode speed: short prompt, 200 tokens
        _, usage2, dt2 = chat_raw(port, 'Count from 1 to 100 slowly.', 200)
        gen2 = usage2.get('completion_tokens', 200)
        decode = gen2 / (dt2 - 0.05) if dt2 > 0 else 0
        print(f'[{tag}] prefill {prefill:.0f} tok/s ({ptok} tokens in {dt:.1f}s); decode {decode:.1f} tok/s')
        # 3. needle
        sys.argv = ['needle_bench', '--url', f'http://127.0.0.1:{port}', '--lengths', '32k',
                    '--depths', '10,50,90', '--out', f'bench/needle-{tag}.json']
        try:
            needle_bench.main()
        except SystemExit as e:
            print(f'[{tag}] needle exit {e.code}')
        # 4. factual
        answers = [chat_raw(port, q, 60)[0] for q, _ in FACTUAL]
        for (q, wants), a in zip(FACTUAL, answers):
            ok = any(w.lower() in (a or '').lower() for w in wants)
            print(f'[{tag}] {"OK " if ok else "MISS"} Q: {q[:44]} -> {(a or "")[:56]!r}')
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

def main():
    legs = sys.argv[1:] or ['nf4:8241', 'int8:8242']
    for leg in legs:
        tag, port = leg.split(':')
        cfg = {'nf4': 'test-config-ram-nvfp4.json', 'nf4r65': 'test-config-ram-nvfp4-r65.json',
               'nf4m6': 'test-config-ram-nvfp4-m6.json', 'int865': 'test-config-ram-int8-65.json'}.get(
            tag, 'test-config-ram-int8.json')
        run_leg(tag, cfg, int(port))
    for leg in legs:
        tag = leg.split(':')[0]
        try:
            rows = json.load(open(f'bench/needle-{tag}.json', encoding='utf-8'))
            found = sum(1 for r in rows if r.get('found'))
            print(f'SUMMARY {tag}: needle {found}/{len(rows)}')
        except Exception as e:
            print(f'SUMMARY {tag}: needle unreadable ({e})')
    return 0

if __name__ == '__main__':
    sys.exit(main())
