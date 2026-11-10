"""Builds the recognition data that ships inside index.html (text embeddings, trained probes, colour examples, starter examples)."""
import os,sys,base64;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from sub import *
from colfit2 import seeds_from,perm,predict as colpredict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict,StratifiedKFold
b64=lambda a:base64.b64encode(a.tobytes()).decode()
M={'v':1};metrics={}
# ---- text embeddings (int16, renormalised at load)
order=[];rng={}
def add(key,prompts):
    rng[key]=[len(order),len(prompts)];order.extend(prompts)
for c,subs in SUB.items():
    for s,pl in subs.items():add('s:'+s,pl)
for c,pl in P['cat'].items():add('c:'+c,pl)
for m,pl in P['mat'].items():add('m:'+m,pl)
vec=np.stack([tv(p) for p in order]);q16=np.stack([np.round(v*(32767/np.abs(v).max())) for v in vec]).astype('<i2')
M['txt']={'n':len(order),'d':b64(q16),'r':rng}
# ---- probes
def probe(X,y,C=10):
    P_=LogisticRegression(C=C,max_iter=8000,class_weight='balanced').fit(X,y);W=P_.coef_;s=float(np.abs(W).max()/127);w8=np.round(W/s).astype(np.int8)
    agree=float(((X@(w8*s).T+P_.intercept_).argmax(1)==(X@W.T+P_.intercept_).argmax(1)).mean())
    return {'l':[str(c) for c in P_.classes_],'s':round(s,6),'b':[round(float(b),4) for b in P_.intercept_],'w':b64(w8)},agree
skf=StratifiedKFold(5,shuffle=True,random_state=0)
M['probe']={};M['groups']={};M['off']={}
for cat in GROUPS:
    X,y,d=DATA[cat];M['probe'][cat],ag=probe(X,y);M['groups'][cat]=GROUPS[cat]
    pr=cross_val_predict(LogisticRegression(C=10,max_iter=8000,class_weight='balanced'),X,y,cv=skf,method='predict_proba');cl=sorted(set(y))
    top1=(np.array(cl)[pr.argmax(1)]==y).mean();top3=np.mean([y[i] in np.array(cl)[np.argsort(-pr[i])[:3]] for i in range(len(y))])
    metrics['sub_'+cat]={'n':int(len(y)),'cv_top1':round(float(top1),3),'cv_top3':round(float(top3),3),'int8_agreement':round(ag,4)}
    for g,mem in GROUPS[cat].items():
        if cat=='top' and len(mem)>1 and mem[0]==g:
            Xg=X[y==g];zm=zl(Xg,SUB[cat][g],{})
            for s in mem[1:]:M['off'][s]=min(0.0,round(-float(np.percentile(zl(Xg,SUB[cat][s],{})-zm,95)),2))
# pattern
mp=zp!='';M['probe']['pat'],ag=probe(ZE[mp],zp[mp])
pr=cross_val_predict(LogisticRegression(C=10,max_iter=8000,class_weight='balanced'),ZE[mp],zp[mp],cv=skf,method='predict_proba');cl=np.array(sorted(set(zp[mp])));pp=cl[pr.argmax(1)];cf_=pr.max(1);yp=zp[mp]
ns=(pp!='solid')&(cf_>=.6)
metrics['pattern']={'n':int(mp.sum()),'cv_top1':round(float((pp==yp).mean()),3),'suggest_precision':round(float((pp[ns]==yp[ns]).mean()),3),'suggest_recall':round(float((ns&(pp==yp)).sum()/(yp!='solid').sum()),3),'int8_agreement':round(ag,4)}
# material (proxy labels: knit = sweaters and cardigans, denim = jeans / denim trend, plain = the rest)
ym=np.array(['']*len(ztyp),dtype=object);ym[zt('t-shirt','blouse','shirt','tank top','top','dress','tunic','trousers','shorts','skirt','rain trousers','winter trousers','blazer','jacket','jacker')]='plain';ym[zt('sweater','cardigan')]='knit';ym[zt('jeans','denim jacket')|(ztrend=='denim')]='denim'
mm=ym!='';y2=ym[mm].astype(str);M['probe']['mat'],ag=probe(ZE[mm],y2)
pr=cross_val_predict(LogisticRegression(C=10,max_iter=8000,class_weight='balanced'),ZE[mm],y2,cv=skf,method='predict_proba');cm=sorted(set(y2))
metrics['material']={k:{'precision@0.8':round(float((y2[pr[:,cm.index(k)]>=.8]==k).mean()),2),'recall@0.8':round(float(((pr[:,cm.index(k)]>=.8)&(y2==k)).sum()/(y2==k).sum()),2)} for k in ['knit','denim']}
# leather / suede: zero-shot margin over ordinary fabric prompts must beat what 98% of ordinary clothes reach
def lg(E,pl):return np.max(np.stack([100*E@tv(x) for x in pl],1),1)
allE=np.vstack([ZE,CE]);ref=np.maximum(lg(allE,P['mat']['cotton']),lg(allE,P['mat']['synthetic']))
M['matthr']={k:round(float(np.percentile(lg(allE,P['mat'][k])-ref,98)),2) for k in ['leather','suede']}
# ---- colour examples (Lab -> family), 30 per colour name
SX,SY=seeds_from(perm,30)
BASE=['black','white','grey','beige','brown','navy','blue','teal','green','yellow','orange','red','burgundy','pink','purple']
arr=np.zeros((len(SX),4),dtype=np.uint8)
for i,(x,l) in enumerate(zip(SX,SY)):arr[i]=[int(round(np.clip(x[0],0,100)*2.55)),int(round(np.clip(x[1],-128,127)+128)),int(round(np.clip(x[2],-128,127)+128)),BASE.index(l)]
M['col']=b64(arr);metrics['colour_examples']=len(SX)
# ---- starter examples: the 20 sample photos
UL=[('bag','handbag'),('bag','handbag'),('top','polo'),('bottom','jeans'),('top','crop'),('top','sweater'),('top','sweater'),('top','tank'),('outer','coat'),('outer','puffer'),('outer','puffer'),('outer','puffer'),('belt',None),('bottom','skirtmidi'),('bottom','skirtlong'),('bag','backpack'),('dress','dressmaxi'),('dress','dressmidi'),('dress','dressmaxi'),('sunglasses',None)]
UPAT={7:'stripes',15:'floral',17:'animal'};UMAT={1:'leather',2:'leather',13:'leather',16:'leather',4:'denim',6:'knit',7:'knit',9:'wool',10:'synthetic',11:'synthetic',12:'synthetic',18:'velvet',19:'satin'}
seeds=[]
for i,(c,s) in enumerate(UL):
    e=UE[i]/np.linalg.norm(UE[i]);o={'c':c,'e':b64(np.clip(np.round(e*127),-127,127).astype(np.int8))}
    if s:o['s']=s
    if i+1 in UPAT:o['p']=UPAT[i+1]
    if i+1 in UMAT:o['m']=UMAT[i+1]
    seeds.append(o)
M['seeds']=seeds
json.dump(M,open(W+'/model.json','w'),separators=(',',':'));json.dump(metrics,open(W+'/metrics.json','w'),indent=1)
print(json.dumps(metrics,indent=1));print('offsets',M['off'],'matthr',M['matthr'],'model bytes',os.path.getsize(W+'/model.json'))
