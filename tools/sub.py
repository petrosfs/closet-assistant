import os,sys;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from data import *
from sklearn.linear_model import LogisticRegression
GROUPS={'top':{'tshirt':['tshirt','longsleeve'],'polo':['polo'],'tank':['tank','bodysuit','bustier','sportstop'],'blouse':['blouse'],'shirt':['shirt'],'sweater':['sweater','sweatshirt'],'turtleneck':['turtleneck'],'hoodie':['hoodie'],'crop':['crop']},
 'outer':{'cardigan':['cardigan'],'blazer':['blazer'],'jacket':['jacket','coat','trench','puffer','vest','raincoat']},
 'bottom':{'pants':['jeans','trousers','sweatpants','leggings'],'shorts':['shorts'],'skirt':['skirtmini','skirtmidi','skirtlong']}}
DATA={'top':TOP,'outer':OUTER,'bottom':BOTTOM}
def zl(E,prompts,bias):return np.max(np.stack([100*E@tv(p)-bias.get(p,0.0) for p in prompts],1),1)
def fit_cat(cat,C=10):
    X,y,d=DATA[cat];P=LogisticRegression(C=C,max_iter=6000,class_weight='balanced').fit(X,y)
    bias={};off={}
    for g,mem in GROUPS[cat].items():
        Xg=X[y==g]
        for s in mem:
            for p in SUB[cat][s]:bias[p]=float((100*Xg@tv(p)).mean()) if len(mem)>1 else 0.0
        if len(mem)>1 and mem[0]==g:   # a main member with known examples: the others must beat it clearly
            zm=zl(Xg,SUB[cat][g],bias)
            for s in mem[1:]:off[s]=-float(np.percentile(zl(Xg,SUB[cat][s],bias)-zm,95))
    return P,bias,off
def predict(cat,E,model,calib=True):
    P,bias,off=model;q=P.predict_proba(E);out=[]
    for i in range(E.shape[0]):
        sc={}
        for gi,g in enumerate(P.classes_):
            mem=GROUPS[cat][g]
            if len(mem)==1:sc[mem[0]]=q[i,gi];continue
            z=np.array([zl(E[i:i+1],SUB[cat][s],bias if calib else {})[0]+off.get(s,0.0) for s in mem]);e=np.exp(z-z.max());e/=e.sum()
            for s,r in zip(mem,e):sc[s]=q[i,gi]*r
        out.append(sorted(sc.items(),key=lambda x:-x[1]))
    return out
if __name__=='__main__':
    M={c:fit_cat(c) for c in GROUPS}
    for c in GROUPS:print(c,'offsets',{k:round(v,2) for k,v in M[c][2].items()})
    # within-group zero-shot where labels exist: jeans vs trousers
    for calib in [True,False]:
        Xj=ZE[zt('jeans')];Xt=ZE[zt('trousers')]
        pj=[r[0][0] for r in predict('bottom',Xj,M['bottom'],calib)];pt=[r[0][0] for r in predict('bottom',Xt,M['bottom'],calib)]
        print(f'calib={calib}: Zen jeans->',collections.Counter(pj).most_common(4),'trousers->',collections.Counter(pt).most_common(4))
        res=[];t3=0
        for i,(tc,ts) in enumerate(ut):
            if tc not in GROUPS or not ts:continue
            r=predict(tc,UE[i:i+1],M[tc],calib)[0];res.append((i+1,r[0][0],round(float(r[0][1]),2),r[0][0] in ts));t3+=any(x[0] in ts for x in r[:3])
        print('   user subs top1',sum(x[3] for x in res),'/',len(res),'top3',t3,[x[:3] for x in res])
    # how often do the calibrated extra members fire on known main members / what do CC0 longsleeves become
    r=predict('top',CE[ccls=='t-shirt'],M['top']);print('CC0 t-shirt ->',collections.Counter(x[0][0] for x in r).most_common(5))
    r=predict('top',CE[ccls=='longsleeve'],M['top']);print('CC0 longsleeve ->',collections.Counter(x[0][0] for x in r).most_common(8))
    r=predict('top',XE[xcls=='Top'],M['top']);print('CC0 Top ->',collections.Counter(x[0][0] for x in r).most_common(8))
    r=predict('outer',CE[ccls=='outwear'],M['outer']);print('CC0 outwear ->',collections.Counter(x[0][0] for x in r).most_common(8))
    r=predict('bottom',CE[ccls=='skirt'],M['bottom']);print('CC0 skirt ->',collections.Counter(x[0][0] for x in r).most_common(8))
    r=predict('bottom',CE[ccls=='pants'],M['bottom']);print('CC0 pants ->',collections.Counter(x[0][0] for x in r).most_common(8))
