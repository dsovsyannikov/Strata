import json, subprocess, sys, time, urllib.request
def wait_ready(port, timeout=900):
    t0=time.time()
    while time.time()-t0<timeout:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5) as r:
                if r.status==200: return True
        except Exception: pass
        time.sleep(3)
    return False
def chat_raw(port, prompt, max_tokens):
    body=json.dumps({'messages':[{'role':'user','content':prompt}],'max_tokens':max_tokens,
                     'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}).encode()
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=body,
                               headers={'Content-Type':'application/json'})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=3600) as r: out=json.load(r)
    return out.get('usage',{}), time.time()-t0
def prompt_of(notes, variant):
    ws=[]
    for i in range(1,notes+1):
        ws.append(f"Note {i}: the ledger lists item {i*7} with value {i%97}; the clerk records pages in order {variant}.")
    return " ".join(ws)
def leg(tag, cfg, port):
    p=subprocess.Popen([r'C:\Users\user\Strata\.venv\Scripts\python.exe','-m','serve.server','--engine','strata',
                        '--config',cfg,'--port',str(port)], cwd=r'C:\Users\user\strata-win-mod')
    try:
        if not wait_ready(port): print(f'[{tag}] FAIL'); return
        time.sleep(5)
        for label, notes in (('prefill14K',410), ('prefill32K',950)):
            u,dt = chat_raw(port, prompt_of(notes, tag), 1)
            n = u.get('prompt_tokens',0)
            print(f'[{tag}] {label}: {n} tokens in {dt:.1f}s = {n/dt:.0f} tok/s')
        rates=[]
        for pr in ('Explain how tides work, then list seven sea birds.',
                   'Describe a busy market street at dusk, one long paragraph.',
                   'Outline the plot of a heist movie set on a night train.'):
            u,dt = chat_raw(port, pr, 260)
            g = u.get('completion_tokens',260); rates.append(g/dt)
            print(f'[{tag}] decode: {g} tokens in {dt:.1f}s = {g/dt:.1f} tok/s')
        rates.sort(); print(f'[{tag}] decode median: {rates[1]:.1f} tok/s')
    finally:
        p.terminate()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired: p.kill()
for tag, cfg, port in (('mainrel','test-mainrel.json',8266), ('orig-int8','test-orig-int8.json',8262)):
    leg(tag, cfg, port)
