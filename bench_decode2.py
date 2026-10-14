import json, subprocess, sys, time, urllib.request
def wait_ready(port, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as r:
                if r.status == 200: return True
        except Exception: pass
        time.sleep(3)
    return False
def chat_raw(port, prompt, max_tokens):
    body = json.dumps({'messages':[{'role':'user','content':prompt}],'max_tokens':max_tokens,
                       'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                                 headers={'Content-Type':'application/json'})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=3600) as r: out=json.load(r)
    return out.get('usage',{}), time.time()-t0
def leg(tag, cfg, port):
    p = subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe','-m','serve.server',
                          '--engine','strata','--config',cfg,'--port',str(port)], cwd=r'C:\Users\user\strata-win-mod')
    try:
        if not wait_ready(port): print(f'[{tag}] FAIL'); return
        time.sleep(5)
        for pr in ('Write a short story about a lighthouse keeper.',
                   'Explain how tides work, then list seven sea birds.',
                   'Describe a busy market street at dusk, one long paragraph.'):
            u, dt = chat_raw(port, pr, 260)
            g = u.get('completion_tokens', 260)
            print(f'[{tag}] decode: {g} tokens in {dt:.1f}s = {g/dt:.1f} tok/s')
    finally:
        p.terminate()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired: p.kill()
for tag, cfg, port in (('int8','test-eq-int8.json',8253),('nf4m6','test-eq-nf4m6.json',8255)):
    leg(tag, cfg, port)
