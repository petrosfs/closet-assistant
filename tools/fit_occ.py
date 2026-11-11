"""Occasion probe: what a garment's photo says about sport and office/formal use, written into the MODEL of index.html.
Data: Fashion Product Images (Small) (via the Hugging Face copy ashraq/fashion-product-images-small), column `usage`:
Casual / Sports / Formal (Party, Smart Casual, Travel and Home have too few photos and are not used).
Sample: all Formal apparel, footwear, bags, belts and headwear (~2.000), up to 2.200 Sports and 3.200 Casual, spread over article types.
Embeddings: CLIP ViT-B/32 quantized, L2-normalised, made as in embcal.mjs (list file with the image path in the first column -> .f32).

Measured (5-fold CV on the product photos): sport at p>=0.9 precision 0.95 / recall 0.44, formal at p>=0.8 precision 0.96 / recall 0.77.
On 713 CC0 home photos (no occasion labels) sport>=0.9 marks 8% (sportswear, trainers, but also plain T-shirts) and formal>=0.8 marks 3%
(leather shoes, shirts, a blazer). The app only ADDS an occasion with these thresholds: sport -> 'sport', formal -> 'smart' (work).
usage: python3 fit_occ.py <index.html> <list.tsv: path, usage, ...> <list.f32>"""
import sys,json,base64,numpy as np
from sklearn.linear_model import LogisticRegression
html,lst,emb=sys.argv[1:4]
H=open(html,encoding='utf8').read();A=H.index('/*MODEL:BEGIN*/const MODEL=')+len('/*MODEL:BEGIN*/const MODEL=');B=H.index(';/*MODEL:END*/');M=json.loads(H[A:B])
L=[l.split('\t') for l in open(lst).read().split('\n') if l];E=np.fromfile(emb,dtype=np.float32).reshape(-1,512)
mp={'Casual':'casual','Sports':'sport','Formal':'formal'};k=np.array([l[1] in mp for l in L])
m=LogisticRegression(C=3,max_iter=5000,class_weight='balanced').fit(E[k],[mp[l[1]] for l in L if l[1] in mp])
W=m.coef_;s=float(np.abs(W).max()/127)
M['probe']['occ']={'l':[str(c) for c in m.classes_],'s':round(s,6),'b':[round(float(b),4) for b in m.intercept_],'w':base64.b64encode(np.round(W/s).astype(np.int8).tobytes()).decode()}
M['occthr']={'sport':0.9,'formal':0.8}
open(html,'w',encoding='utf8').write(H[:A]+json.dumps(M,separators=(',',':'))+H[B:]);print('ok',int(k.sum()),'photos')
