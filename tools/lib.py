import json,numpy as np,os
# working folders (see docs/DEVELOPMENT.md)
W=os.environ.get('CA_WORK','work');EV=os.environ.get('CA_EV','ev');DATA=os.environ.get('CA_DATA','data')
P=json.load(open(W+'/prompts.json'))
TXT={k:np.array(v,dtype=np.float32) for k,v in json.load(open(EV+'/texts.json')).items()}
def tv(p):return TXT['a photo of '+p]
SUB=P['sub']
CATS=['top','bottom','dress','jumpsuit','outer','shoes','bag','belt','hat','sunglasses','scarf','winterscarf','gloves','watch','necklace','earrings','bracelet','ring','tights']
def cat_prompts(P=P):
    L={}
    for c in CATS:
        if c in P['cat']:L[c]=list(P['cat'][c])
        else:L[c]=[x for l in P['sub'][c].values() for x in l]
    return L
def head(labels):
    """labels: {id:[prompts]} -> (ids, owner idx array, matrix)"""
    ids=list(labels);own=[];M=[]
    for i,k in enumerate(ids):
        for p in labels[k]:own.append(i);M.append(tv(p))
    return ids,np.array(own),np.stack(M)
def zs(E,labels,mode='sum',T=100.0):
    ids,own,M=head(labels);lg=T*E@M.T
    if mode=='max':
        out=np.full((E.shape[0],len(ids)),-1e9,dtype=np.float32)
        for j,o in enumerate(own):out[:,o]=np.maximum(out[:,o],lg[:,j])
        out=out-out.max(1,keepdims=True);e=np.exp(out);return ids,e/e.sum(1,keepdims=True)
    lg=lg-lg.max(1,keepdims=True);e=np.exp(lg);e/=e.sum(1,keepdims=True)
    out=np.zeros((E.shape[0],len(ids)),dtype=np.float32)
    for j,o in enumerate(own):out[:,o]+=e[:,j]
    return ids,out
def load_cc0():
    files=open(W+'/cc0.txt').read().split('\n');E=np.fromfile(W+'/cc0.f32',dtype=np.float32).reshape(-1,512)
    split=np.array([f.split('/')[-3] for f in files]);cls=np.array([f.split('/')[-2] for f in files]);return files,E,split,cls
CC0_CAT={'dress':'dress','hat':'hat','longsleeve':'top','outwear':'outer','pants':'bottom','shirt':'top','shoes':'shoes','shorts':'bottom','skirt':'bottom','t-shirt':'top'}
def load_user():
    files=open(EV+'/files.txt').read().split('\n');d=json.load(open(EV+'/imgs.json'));E=np.array([d[f] for f in files],dtype=np.float32)
    truth=[('bag',{'handbag','tote'}),('bag',{'handbag','tote'}),('top',{'polo','tshirt'}),('bottom',{'jeans','trousers'}),('top',{'crop','tank'}),('top',{'sweater'}),('top',{'sweater'}),('top',{'tank','crop','bustier'}),('outer',{'coat','jacket'}),('outer',{'puffer'}),('outer',{'puffer','coat'}),('outer',{'puffer'}),('belt',None),('bottom',{'skirtmidi'}),('bottom',{'skirtlong'}),('bag',{'backpack'}),('dress',{'dressmaxi'}),('dress',{'dressmidi'}),('dress',{'dressmaxi','dressmidi'}),('sunglasses',None)]
    return files,E,truth
def zs_g(E,labels,g=0.0,T=100.0):
    """log-sum-exp over a label's prompts minus g*log(n): g=0 is the plain sum, g=1 the mean."""
    ids,own,M=head(labels);lg=T*E@M.T;out=np.zeros((E.shape[0],len(ids)),dtype=np.float64)
    for i in range(len(ids)):
        x=lg[:,own==i];m=x.max(1,keepdims=True);out[:,i]=(m[:,0]+np.log(np.exp(x-m).sum(1)))-g*np.log(x.shape[1])
    out=out-out.max(1,keepdims=True);e=np.exp(out);return ids,e/e.sum(1,keepdims=True)
def fuse(ids,p,pids,q,a):
    """Re-rank only inside the label set the probe covers: mass on covered labels stays the same."""
    out=p.copy();idx=[ids.index(k) for k in pids];M=p[:,idx].sum(1,keepdims=True)
    out[:,idx]=(1-a)*p[:,idx]+a*M*q
    return out
import glob,collections
ZCAT={'sweater':'top','blouse':'top','top':'top','t-shirt':'top','shirt':'top','tank top':'top','tunic':'top','hoodie':'top','training top':'top','dress':'dress',
 'cardigan':'outer','jacket':'outer','jacker':'outer','blazer':'outer','vest':'outer','outerwear':'outer','rain jacket':'outer','denim jacket':'outer','winter jacket':'outer',
 'trousers':'bottom','jeans':'bottom','skirt':'bottom','shorts':'bottom','rain trousers':'bottom','winter trousers':'bottom'}
ZSUB={'sweater':'sweater','blouse':'blouse','tunic':'blouse','t-shirt':'tshirt','shirt':'shirt','tank top':'tank','hoodie':'hoodie','training top':'sportstop',
 'cardigan':'cardigan','jacket':'jacket','jacker':'jacket','blazer':'blazer','vest':'vest','denim jacket':'jacket','winter jacket':'puffer','rain jacket':'raincoat','jeans':'jeans','trousers':'trousers','shorts':'shorts'}
ZPAT={'plain':'solid','floral print':'floral','striped':'stripes','logo print':'print','geometric print':'print','animal print':'animal','dots':'dots','checkered print':'checks'}
def load_zen():
    files=open(W+'/zen.txt').read().split('\n');E=np.fromfile(W+'/zen.f32',dtype=np.float32).reshape(-1,512)
    lab={}
    for f in glob.glob(DATA+'/zen/x/**/labels_*.json',recursive=True):
        try:lab[os.path.basename(f)[7:-5]]=json.load(open(f))
        except:pass
    ids=[os.path.basename(f)[:-4] for f in files];keep=[i for i,x in enumerate(ids) if x in lab]
    E=E[keep];ids=[ids[i] for i in keep];J=[lab[x] for x in ids]
    typ=[str(j.get('type','')).strip().lower() for j in J]
    cat=np.array([ZCAT.get(t,'') for t in typ]);sub=[]
    for j,t in zip(J,typ):
        s=ZSUB.get(t,'');cut=[str(c).lower() for c in (j.get('cut') or [])]
        if t=='sweater' and 'turtle neck' in cut:s='turtleneck'
        sub.append(s)
    pat=np.array([ZPAT.get(str(j.get('pattern','')).strip().lower(),'') for j in J])
    ssn=np.array([str(j.get('season','')).strip().lower() for j in J])
    return ids,E,J,cat,np.array(sub),pat,ssn
def zs_cal(E,labels,bias,T=100.0,mode='max',scale=None):
    """zero-shot where each prompt's logit is first reduced by its own bias (how readily it matches anything)."""
    ids=list(labels);out=np.full((E.shape[0],len(ids)),-1e9)
    acc=np.zeros((E.shape[0],len(ids)))
    for i,k in enumerate(ids):
        ls=[]
        for p in labels[k]:
            l=T*E@tv(p)-bias.get(p,0.0)
            if scale:l=l/scale.get(p,1.0)
            ls.append(l)
        ls=np.stack(ls,1);out[:,i]=ls.max(1) if mode=='max' else ls.mean(1)
    out=out-out.max(1,keepdims=True);e=np.exp(out);return ids,e/e.sum(1,keepdims=True)
def prompt_bias(groups,labels,T=100.0):
    """groups: list of embedding arrays (one per true class) -> bias per prompt = mean of the class means"""
    b={}
    for k,pl in labels.items():
        for p in pl:b[p]=float(np.mean([(T*g@tv(p)).mean() for g in groups]))
    return b
