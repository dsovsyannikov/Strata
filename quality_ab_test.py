"""Quality A/B: the F32 and F16 engines on (1) the repo's own needle bench (32K, depths 10/50/90,
fixed seed -> identical needles for both) and (2) a battery of factual prompts at temperature 0,
each with a machine-checkable expectation. Also grabs each engine's decode tok/s from its log.

Both legs run sequentially on the same resident config; only --gdn-f16 differs.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, 'tools')
import needle_bench  # noqa: E402  (run with args patched below)

QUESTIONS = [
    ("Сколько будет 17 * 23? Ответь только числом.", ["391"]),
    ("Capital of Australia? Answer with the city name only.", ["canberra"]),
    ("What does this Python print, digits only, space-separated:\nfor i in range(3): print(i*i)",
     ["0 1 4"]),
    ("A train travels 60 km in 45 minutes. What is its speed in km/h? Answer with the number only.",
     ["80"]),
    ("Name the planets of the solar system that come after Earth, in order. Names only.",
     ["mars", "jupiter", "saturn"]),
    ("Напиши функцию Python is_palindrome(s), которая игнорирует регистр и пробелы. Только код.",
     ["def is_palindrome", "return"]),
    ("Переведи на русский: 'The quick brown fox jumps over the lazy dog.'",
     ["лис", "пёс"]),
    ("Write a SQL query that selects name and salary from employees where salary > 50000, ordered by salary descending.",
     ["select", "from employees", "order by", "50000"]),
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
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)['choices'][0]['message']['content']


def run_leg(tag, config, port, log):
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', config, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod')
    try:
        if not wait_ready(port):
            print(f'[{tag}] FAIL: server never ready')
            return None
        # a short chat to measure decode tok/s at full power
        chat(port, 'Скажи одним предложением, что такое второй закон Ньютона.', 60)
        time.sleep(1)
        needle_argv = ['needle_bench', '--url', f'http://127.0.0.1:{port}', '--lengths', '32k',
                       '--depths', '10,50,90', '--out', f'bench/needle-{tag}.json']
        old_argv, sys.argv = sys.argv, needle_argv
        try:
            results = needle_bench.main() if hasattr(needle_bench, 'main') else None
        except SystemExit as e:
            results = None
            print(f'[{tag}] needle exited with {e.code}')
        finally:
            sys.argv = old_argv
        answers = []
        for q, _ in QUESTIONS:
            a = chat(port, q, 260)
            answers.append(a)
            time.sleep(0.3)
        time.sleep(2)
        log_text = open(log, encoding='utf-8', errors='replace').read()
        tps = re.findall(r'(\d+) generated in [\d.]+ ms \(([\d.]+) tok/s\)', log_text)
        return {'needle': results, 'answers': answers, 'tps': tps[-6:]}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    legs = {}
    for tag, cfg, log in (('f32', 'test-config-ram.json', r'C:\Users\user\strata-win-mod\qa-f32-engine.log'),
                          ('f16', 'test-config-ram-f16.json', r'C:\Users\user\strata-win-mod\qa-f16-engine.log')):
        for f in (log,):
            if os.path.exists(f):
                os.remove(f)
        leg = run_leg(tag, cfg, 8231 if tag == 'f32' else 8232, log)
        if leg is None:
            return 1
        legs[tag] = leg

    print('\n=== decode tok/s (last requests, full power) ===')
    for tag in ('f32', 'f16'):
        print(tag, legs[tag]['tps'])

    print('\n=== needle bench (32K, depths 10/50/90, same needles both legs) ===')
    for tag in ('f32', 'f16'):
        n = legs[tag]['needle']
        if n is None:
            try:
                n = json.load(open(f'bench/needle-{tag}.json', encoding='utf-8'))
            except Exception:
                n = {'raw': 'see bench/needle-%s.json' % tag}
        rows = n if isinstance(n, list) else None
        if rows is None:
            try:
                rows = json.load(open(f'bench/needle-{tag}.json', encoding='utf-8'))
            except Exception:
                rows = []
        found = [r for r in rows if r.get('found')]
        print(f'{tag}: {len(found)}/{len([r for r in rows if "found" in r])} needles found')
        for r in rows:
            print('   ', r.get('length'), f"depth {r.get('depth')}%:", 'FOUND' if r.get('found') else 'MISSED',
                  ('err: ' + r['error']) if 'error' in r else '')

    print('\n=== factual answers (temperature 0) ===')
    fails = 0
    for i, (q, wants) in enumerate(QUESTIONS):
        line = f'Q{i + 1}: {q[:60]}...'
        ok32 = any(w.lower() in legs['f32']['answers'][i].lower() for w in wants)
        ok16 = any(w.lower() in legs['f16']['answers'][i].lower() for w in wants)
        mark = 'OK ' if (ok32 and ok16) else ('DIFF' if ok32 != ok16 else 'BOTH-FAIL')
        if mark != 'OK ':
            fails += 1
        print(f'{mark} {line}')
        print(f'      f32: {legs["f32"]["answers"][i][:110]!r}')
        print(f'      f16: {legs["f16"]["answers"][i][:110]!r}')
    print(f'\nfactual: {"ALL PASS" if fails == 0 else str(fails) + " prompts differ in verdict"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
