"""VRAM expert-cache comparison: original main engine vs fork (int8 and nvfp4 legs),
identical config otherwise. Starts each engine, waits for ready, kills, greps the slots line."""
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

def leg(tag, cfg, port):
    proc = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe', '-m', 'serve.server',
                             '--engine', 'strata', '--config', cfg, '--port', str(port)],
                            cwd=r'C:\Users\user\strata-win-mod')
    ok = wait_ready(port)
    time.sleep(2)
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f'[{tag}] {"READY" if ok else "FAILED"}')

def main():
    for tag, cfg, port in (('full-int8', 'test-exp-full8.json', 8275),
                           ('fork-int8', 'test-exp-fint8.json', 8272),
                           ('fork-nvfp4', 'test-exp-fnf4.json', 8273)):
        leg(tag, cfg, port)
    return 0

if __name__ == '__main__':
    sys.exit(main())
