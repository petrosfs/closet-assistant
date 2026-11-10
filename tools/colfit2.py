import os,sys;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from colfit import *
from sklearn.cluster import KMeans
STRICT={'blue':['blue','navy'],'red':['red','burgundy'],'turquoise':['teal'],'white':['white'],'black':['black'],'pink':['pink'],'grey':['grey'],'beige':['beige'],'green':['green'],'purple':['purple'],'brown':['brown'],'yellow':['yellow'],'orange':['orange']}
def seedlabel(c,l):
    fams=STRICT[c];pal={k:v for k,v in AL.items() if FAM[k] in fams};return FAM[near(l,pal)]
X=np.array([l for _,c,l,_ in rows]);Y=[c for _,c,l,_ in rows];YS=[seedlabel(c,l) for _,c,l,_ in rows]
PALF=[FAM[near(l,AL)] for l in X]
def predict(l,palf,SX,SY,w0=.6,R=10,ws=1.0,sc=5):
    w=collections.defaultdict(float);w[palf]+=w0
    d=np.linalg.norm(SX-l,axis=1)
    for i in np.where(d<=R)[0]:w[SY[i]]+=ws/(1+d[i]/sc)
    return max(w,key=w.get)
def seeds_from(idx,per):
    SX=[];SY=[]
    by=collections.defaultdict(list)
    for i in idx:by[YS[i]].append(X[i])
    for k,v in by.items():
        v=np.array(v)
        if per and len(v)>per:
            km=KMeans(per,n_init=3,random_state=0).fit(v);cnt=np.bincount(km.labels_,minlength=per)
            for c,n in zip(km.cluster_centers_,cnt):
                if n>=2:SX.append(c);SY.append(k)
        else:
            for p in v:SX.append(p);SY.append(k)
    return np.array(SX),SY
rng=np.random.RandomState(0);perm=rng.permutation(len(X));folds=np.array_split(perm,5)
U=json.load(open(W+'/user_col.json'));UL=[np.array(u[0]['lab']) for u in U];UP=[FAM[near(l,AL)] for l in UL]
def cv(per,**kw):
    ok=0;oks=0
    for f in range(5):
        te=folds[f];tr=np.concatenate([folds[g] for g in range(5) if g!=f]);SX,SY=seeds_from(tr,per)
        for i in te:
            p=predict(X[i],PALF[i],SX,SY,**kw);ok+=p in MAP[Y[i]];oks+=p in STRICT[Y[i]]
    SX,SY=seeds_from(perm,per);u=[predict(UL[i],UP[i],SX,SY,**kw) for i in range(20)];uo=sum(u[i] in TR[i] for i in range(20))
    return ok/len(X),oks/len(X),uo,len(SX),[(i+1,u[i]) for i in range(20) if u[i] not in TR[i]]
print('palette only: lenient',np.mean([PALF[i] in MAP[Y[i]] for i in range(len(X))]).round(3),'strict',np.mean([PALF[i] in STRICT[Y[i]] for i in range(len(X))]).round(3))
for per in [0,8,16,30]:
    for kw in [dict(),dict(R=8),dict(R=14,sc=6),dict(ws=.5),dict(w0=1.0)]:
        a,b,u,n,bad=cv(per,**kw);print(f'per={per:2d} seeds={n:4d} {str(kw):22s} CV lenient {a:.3f} strict {b:.3f} | user {u}/20 {bad}')
