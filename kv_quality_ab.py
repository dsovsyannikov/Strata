"""Needle A/B for --kv nvfp4 vs --kv int8 (both with the same resident config, seeds fixed ->
identical needles), plus a small factual battery. Starts one engine at a time.
"""
import json
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, 'tools')
import needle_bench  # noqa: E402

QUESTIONS = [
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


def chat(port, prompt, max_tokens=200):
    body = json.dumps({'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': max_tokens, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)['choices'][0]['message']['content']


def run_leg(tag, config, port):
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', config, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod')
    try:
        if not wait_ready(port):
            print(f'[{tag}] FAIL: server never ready')
            return None
        sys.argv = ['needle_bench', '--url', f'http://127.0.0.1:{port}', '--lengths', '32k',
                    '--depths', '10,50,90', '--out', f'bench/needle-{tag}.json']
        try:
            needle_bench.main()
        except SystemExit as e:
            print(f'[{tag}] needle exit {e.code}')
        answers = [chat(port, q) for q, _ in QUESTIONS]
        return answers
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    legs = {}
    for tag, cfg, port in (('nvfp4', 'test-config-ram-nvfp4.json', 8241),
                           ('int8', 'test-config-ram-int8.json', 8242)):
        a = run_leg(tag, cfg, port)
        if a is None:
            return 1
        legs[tag] = a
    for tag in ('nvfp4', 'int8'):
        try:
            rows = json.load(open(f'bench/needle-{tag}.json', encoding='utf-8'))
            found = sum(1 for r in rows if r.get('found'))
            print(f'{tag}: needle {found}/{len(rows)}')
        except Exception as e:
            print(f'{tag}: needle results unreadable ({e})')
    for i, (q, wants) in enumerate(QUESTIONS):
        ok4 = any(w.lower() in legs['nvfp4'][i].lower() for w in wants)
        ok8 = any(w.lower() in legs['int8'][i].lower() for w in wants)
        print(f'{"OK " if ok4 and ok8 else "DIFF"} Q: {q[:50]} | nvfp4={legs["nvfp4"][i][:60]!r} int8={legs["int8"][i][:60]!r}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
