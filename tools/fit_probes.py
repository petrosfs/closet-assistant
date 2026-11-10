"""Category probe and loungewear sub-type probe, trained on freely licensed labelled photos, written into the MODEL of index.html.
Data (embeddings made with embcal.mjs-style scripts, CLIP ViT-B/32 quantized, L2-normalised):
 - Fashion Product Images (Small), MIT licence (Kaggle paramaggarwal, via the Hugging Face copy ashraq/fashion-product-images-small): list.tsv + fpi.f32
 - Clothing dataset small, CC0: cal_files.txt + cal.f32
 - Clothing Dataset for Second-Hand Fashion (Zenodo 8386668), CC BY 4.0: zen list.tsv + zen.f32
Measured (5-fold CV, mean over source classes) and on the 20 sample photos, which are never trained on: see docs/DEVELOPMENT.md.
usage: python3 fit_probes.py <index.html> <fpi list> <fpi f32> <cc0 files> <cc0 f32> <zen list> <zen f32>"""
import sys,json,base64,numpy as np,re
from sklearn.linear_model import LogisticRegression
html,fl,ff,cl,cf,zl,zf=sys.argv[1:8]
H=open(html,encoding='utf8').read();A=H.index('/*MODEL:BEGIN*/const MODEL=')+len('/*MODEL:BEGIN*/const MODEL=');B=H.index(';/*MODEL:END*/');M=json.loads(H[A:B])
def load(lst,emb):return [l.split('\t') for l in open(lst).read().split('\n') if l],np.fromfile(emb,dtype=np.float32).reshape(-1,512)
L1,E1=load(fl,ff);L3,E3=load(zl,zf);f2=open(cf if False else cl).read().split('\n');E2=np.fromfile(cf,dtype=np.float32).reshape(-1,512)
mp={'dress':'dress','hat':'hat','longsleeve':'top','shirt':'top','t-shirt':'top','outwear':'outer','pants':'bottom','shorts':'bottom','skirt':'bottom','shoes':'shoes'}
X=np.vstack([E1,E2,E3]);y=np.array([l[1].split('/')[0] for l in L1]+[mp[f.split('/')[-2]] for f in f2]+[l[1] for l in L3])
src=np.array(['fpi']*len(L1)+['cc0']*len(f2)+['zen']*len(L3));k=y!='hairacc';X,y,src=X[k],y[k],src[k]
def pack(m):
    W=m.coef_;s=float(np.abs(W).max()/127);return {'l':[str(c) for c in m.classes_],'s':round(s,6),'b':[round(float(b),4) for b in m.intercept_],'w':base64.b64encode(np.round(W/s).astype(np.int8).tobytes()).decode()}
m=LogisticRegression(C=10,max_iter=5000,class_weight='balanced').fit(X,y,sample_weight=np.where(src=='zen',2,np.where(src=='cc0',2,1.0)))
M['probe']['cat']=pack(m);M['catA']=0.4
hm={'Night suits':'pajamas','Nightdress':'nightgown','Baby Dolls':'nightgown','Bath Robe':'robe','Robe':'robe','Lounge Pants':'lounge','Lounge Shorts':'lounge','Lounge Tshirts':'lounge'}
k=[i for i,l in enumerate(L1) if l[2] in hm];mh=LogisticRegression(C=10,max_iter=5000,class_weight='balanced').fit(E1[k],[hm[L1[i][2]] for i in k])
M['probe']['homewear']=pack(mh);M['groups']['homewear']={c:[c] for c in mh.classes_}
# sub-type offsets checked on the labelled product photos: tunic vs other tops, wallet vs other bags
M['off'].update({'tunic':-0.4,'wallet':-1.5})
open(html,'w',encoding='utf8').write(H[:A]+json.dumps(M,separators=(',',':'))+H[B:]);print('ok',len(y),'photos')
