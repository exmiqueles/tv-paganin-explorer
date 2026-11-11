import numpy as np, gzip, base64, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda *p: os.path.join(HERE, *p)
q = np.load(J('validation','ref','q_full.npy')).astype(np.int32)
meta = json.load(open(J('validation','ref','q_meta.json')))
H, W = q.shape
d = (np.diff(q, axis=1, prepend=0) % 65536).astype('<u2').tobytes()
planes = d[0::2] + d[1::2]                 # low bytes, then high bytes
b64 = base64.b64encode(gzip.compress(planes, 9, mtime=0)).decode()
core = open(J('src','core.js')).read()
assert '</script' not in core
html = open(J('src','template.html')).read()
val = open(J('validation','validation.md')).read() if os.path.exists(J('validation','validation.md')) else 'validation report not built'
assert '</script' not in val
html = html.replace('%%VALIDATION%%', val)   # raw text inside <script type=text/plain>
html = html.replace('%%CORE%%', core).replace('%%DATA%%', b64).replace('%%META%%', json.dumps(dict(H=H, W=W, rmin=meta['rmin'], rmax=meta['rmax'])))
open(J('index.html'), 'w').write(html)
print(len(html)/1e6, 'MB')
