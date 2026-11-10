"""Adds the sub-types / categories that have no labelled training data (see docs/DEVELOPMENT.md, "Νέοι τύποι").
Reads the MODEL inside index.html, appends the text embeddings of the new prompts (made with embnew.mjs: 'a photo of <prompt>'),
sets the groups, and calibrates offsets on the CC0 clothing-dataset-small photos so a new type does not steal photos of the
ordinary types: a new sub-type may win at most ~5% of the photos of its host class; a new category must not become the top category
of the dataset photos. Writes calib.json next to this file and patches index.html.
usage: python3 extend_model.py <index.html> <newtx.json> <cal_files.txt> <cal.f32>"""
import sys,json,base64,numpy as np,os
html_p,newtx_p,files_p,emb_p=sys.argv[1:5]
here=os.path.dirname(os.path.abspath(__file__))
H=open(html_p,encoding='utf8').read();A=H.index('/*MODEL:BEGIN*/const MODEL=')+len('/*MODEL:BEGIN*/const MODEL=');B=H.index(';/*MODEL:END*/')
M=json.loads(H[A:B]);P=json.load(open(here+'/prompts.json'))
NT=json.load(open(newtx_p));l2=lambda v:v/np.linalg.norm(v)
raw=np.frombuffer(base64.b64decode(M['txt']['d']),dtype='<i2').reshape(-1,512).astype(np.float32);vecs=np.stack([l2(v) for v in raw])
V={k:vecs[s:s+n] for k,(s,n) in M['txt']['r'].items()}
SUBT_NEW={'top':['tunic','wraptop'],'bottom':['cargo','palazzo','bermuda','skort'],'outer':['parka','fleece','overshirt','poncho','kimono','fur'],
 'shoes':['mules','espadrilles','slippers','rainboots'],'hat':['beret','bucket','headband'],'bag':['wallet','suitcase'],
 'swim':['bikini','swimsuit','coverup'],'homewear':['pajamas','nightgown','robe','lounge']}
NEWCAT=['hairacc','umbrella','socks']
# the two puffer entries were first added without the 'a photo of' prefix: redo them like everything else
for s in ['puffer','puffercoat']+[x for l in SUBT_NEW.values() for x in l]:
    cat=[c for c in P['sub'] if s in P['sub'][c]][0]
    V['s:'+s]=np.stack([np.array(NT['a photo of '+t],dtype=np.float32) for t in P['sub'][cat][s]])
for c in NEWCAT:V['c:'+c]=np.stack([np.array(NT['a photo of '+t],dtype=np.float32) for t in P['cat'][c]])
files=open(files_p).read().split('\n');E=np.fromfile(emb_p,dtype=np.float32).reshape(-1,512);cls=np.array([f.split('/')[-2] for f in files])
zl=lambda key:np.max(100*E@V[key].T,1)
off=dict(M['off']);off['puffercoat']=-1.0
def margin(new,comp,host):
    z=np.max([zl('s:'+c)+min(0,off.get(c,0)) for c in comp],0);return (zl('s:'+new)-z)[np.isin(cls,host)]
def calib(new,comp,host,pct):
    off[new]=min(0.0,round(-float(np.percentile(margin(new,comp,host),pct)),2))
ex={'top':['blouse'],'bottom':None}
for s in ['tunic','wraptop']:calib(s,['blouse'],['shirt'],95)
for s in ['cargo','palazzo']:calib(s,['jeans','trousers','sweatpants','leggings'],['pants'],95)
calib('bermuda',['shorts'],['shorts'],95);calib('skort',['skirtmini','skirtmidi','skirtlong'],['skirt'],95)
for s in SUBT_NEW['outer']:calib(s,['jacket','coat','trench','puffer','puffercoat','vest','raincoat'],['outwear'],97)
for s in SUBT_NEW['shoes']:calib(s,['athletic','sneakers','boots','ankle','heels','platform','flats','sandals','loafers'],['shoes'],95)
for s in SUBT_NEW['hat']:calib(s,['cap','beanie','sunhat','fedora'],['hat'],95)
# the group each new type is judged in (a trained probe picks the group; the prompts pick inside it)
G=M['groups'];G['top']['blouse']+=['tunic','wraptop'];G['bottom']['pants']+=['cargo','palazzo'];G['bottom']['shorts']+=['bermuda'];G['bottom']['skirt']+=['skort']
G['outer']['jacket']=['jacket','coat','trench','puffer','puffercoat','vest','raincoat']+SUBT_NEW['outer']
# category level: softmax over the prompts of every category, as the app does; a bias per new category keeps it from winning photos of ordinary clothes
SUBT={'top':['tshirt','longsleeve','tank','crop','bustier','bodysuit','blouse','shirt','polo','sweater','turtleneck','sweatshirt','hoodie','sportstop','tunic','wraptop'],
 'bottom':['jeans','trousers','sweatpants','shorts','skirtmini','skirtmidi','skirtlong','leggings','cargo','palazzo','bermuda','skort'],
 'dress':['dressmini','dressmidi','dressmaxi'],'jumpsuit':['jumplong','romper','overalls'],
 'outer':['jacket','coat','trench','puffer','puffercoat','blazer','cardigan','vest','raincoat']+SUBT_NEW['outer'],
 'shoes':['athletic','sneakers','boots','ankle','heels','platform','flats','sandals','loafers']+SUBT_NEW['shoes'],
 'bag':['handbag','tote','crossbody','backpack','clutch','beltbag']+SUBT_NEW['bag'],'hat':['cap','beanie','sunhat','fedora']+SUBT_NEW['hat'],
 'swim':SUBT_NEW['swim'],'homewear':SUBT_NEW['homewear']}
OLD=['top','bottom','dress','jumpsuit','outer','shoes','bag','belt','hat','sunglasses','scarf','winterscarf','gloves','watch','necklace','earrings','bracelet','ring','tights']
def catvecs(cats,withnew):
    L=[]
    for c in cats:
        if 'c:'+c in V:ks=['c:'+c]
        else:ks=['s:'+s for s in SUBT[c] if withnew or s not in sum(SUBT_NEW.values(),[]) and s!='puffercoat' or s=='puffercoat']
        for k in ks:L+=[(c,v) for v in V[k]]
    return L
def catp(L,coff):
    lg=np.stack([100*E@v+coff.get(c,0) for c,v in L],1);lg-=lg.max(1,keepdims=True);e=np.exp(lg);e/=e.sum(1,keepdims=True)
    cats=sorted(set(c for c,_ in L));return cats,np.stack([sum(e[:,j] for j,(c2,_) in enumerate(L) if c2==c) for c in cats],1)
want={'dress':'dress','hat':'hat','longsleeve':'top','shirt':'top','t-shirt':'top','outwear':'outer','pants':'bottom','shorts':'bottom','skirt':'bottom','shoes':'shoes'}
y=np.array([want[c] for c in cls])
def acc(cats,pr):return float((np.array(cats)[pr.argmax(1)]==y).mean())
# before: the 19 original categories with the original prompts
Lo=[(c,v) for c in OLD for v in (V['c:'+c] if 'c:'+c in V else np.concatenate([V['s:'+s] for s in SUBT[c] if s not in sum(SUBT_NEW.values(),[]) and s!='puffercoat']))]
co,po=catp(Lo,{});print('category accuracy, before:',round(acc(co,po),3))
Ln=catvecs(OLD+['swim','homewear']+NEWCAT,True);coff={}
for c in ['swim','homewear']+NEWCAT:
    for b in range(0,-12,-1):
        trial=dict(coff);trial[c]=b;cn,pn=catp(Ln,trial);top=np.array(cn)[pn.argmax(1)]==c;mass=pn[:,cn.index(c)].mean()
        if top.mean()<=0.005 and mass<=0.015:coff[c]=b;break
    else:coff[c]=-12
# more prompts in a category make it win more often (the sum of prompt shares grows): pull the categories that received new prompts back
# to the accuracy they had, as far as a bias can (coordinate search, smallest bias that is within 0.005 of the best)
for _ in range(2):
    for c in ['outer','shoes','hat','bag','top','bottom']:
        res=[]
        for b in np.arange(0,-8.01,-0.5):
            t=dict(coff);t[c]=float(b);cn_,pn_=catp(Ln,t);res.append((acc(cn_,pn_),float(b)))
        best=max(r[0] for r in res);coff[c]=[b for a,b in res if a>=best-0.005][0]
coff={k:v for k,v in coff.items() if v}
cn,pn=catp(Ln,coff);print('category accuracy, after:',round(acc(cn,pn),3),'coff',coff)
for c in ['swim','homewear']+NEWCAT:print(' ',c,'top-1 on dataset photos:',round(float((np.array(cn)[pn.argmax(1)]==c).mean()),4),'mean share',round(float(pn[:,cn.index(c)].mean()),4))
# rate at which a new sub-type wins its host photos
def rate(new,comp,host):return float((margin(new,comp,host)+min(0,off[new])>0).mean())
print('new-type win rates on host photos:',{s:round(rate(s,c,h),3) for s,c,h in [('tunic',['blouse'],['shirt']),('cargo',['jeans','trousers','sweatpants','leggings'],['pants']),('bermuda',['shorts'],['shorts']),('skort',['skirtmini','skirtmidi','skirtlong'],['skirt'])]})
if os.environ.get('DIAG'):
    for c in sorted(set(cls)):
        m=cls==c;print(' ',c.ljust(10),'before',round(float((np.array(co)[po[m].argmax(1)]==y[m]).mean()),2),'after',round(float((np.array(cn)[pn[m].argmax(1)]==y[m]).mean()),2),'wrong->',dict(zip(*np.unique(np.array(cn)[pn[m].argmax(1)][np.array(cn)[pn[m].argmax(1)]!=y[m]],return_counts=True))))
M['off']=off;M['coff']=coff
add=[];n=M['txt']['n'];keep={k:v for k,v in M['txt']['r'].items() if k not in ['s:puffer','s:puffercoat']}
# rebuild the text block: old entries (minus the two puffer ones) + all new ones, as int16
blocks=[];R={};n=0
for k,(s,c) in sorted(M['txt']['r'].items(),key=lambda x:x[1][0]):
    if k in ['s:puffer','s:puffercoat']:continue
    R[k]=[n,c];blocks.append(raw[s:s+c]);n+=c
for k in ['s:puffer','s:puffercoat']+['s:'+s for l in SUBT_NEW.values() for s in l]+['c:'+c for c in NEWCAT]:
    v=V[k];q=np.stack([np.round(x*(32767/np.abs(x).max())) for x in v]).astype('<i2').astype(np.float32);R[k]=[n,len(v)];blocks.append(q);n+=len(v)
M['txt']={'n':n,'d':base64.b64encode(np.concatenate(blocks).astype('<i2').tobytes()).decode(),'r':R}
json.dump({'off':{k:off[k] for k in off if k in sum(SUBT_NEW.values(),[])+['puffercoat']},'coff':coff},open(here+'/calib.json','w'),indent=1)
open(html_p,'w',encoding='utf8').write(H[:A]+json.dumps(M,separators=(',',':'))+H[B:])
print('written; offsets:',{k:off[k] for k in sum(SUBT_NEW.values(),[]) if k in off})
