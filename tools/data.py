import os,sys;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
import csv
ids,ZE,J,zc,zsb,zp,zn=load_zen()
ztyp=np.array([str(j.get('type','')).strip().lower() for j in J])
zcut=[[str(c).lower() for c in (j.get('cut') or [])] for j in J]
ztrend=np.array([str(j.get('trend','')).strip().lower() for j in J])
zpatraw=np.array([str(j.get('pattern','')).strip().lower() for j in J])
cf,CE,csplit,ccls=load_cc0()
xf=open(W+'/cc0x.txt').read().split('\n');XE=np.fromfile(W+'/cc0x.f32',dtype=np.float32).reshape(-1,512)
_lab={r['image']:r['label'] for r in csv.DictReader(open(DATA+'/gh/clothing-dataset/images.csv'))}
xcls=np.array([_lab[os.path.basename(f)[:-4]] for f in xf])
uf,UE,ut=load_user()
def pool(spec):
    """spec: {label:[(E,mask,domain)...]} -> X,y,dom"""
    X=[];y=[];d=[]
    for k,parts in spec.items():
        for E,m,dom in parts:
            X.append(E[m]);y+= [k]*int(m.sum());d+=[dom]*int(m.sum())
    return np.vstack(X),np.array(y),np.array(d)
zt=lambda *t:np.isin(ztyp,t)
turtle=np.array(['turtle neck' in c for c in zcut]);cropped=np.array(['cropped' in c for c in zcut])
TOP=pool({
 'tshirt':[(ZE,zt('t-shirt')&~cropped,'z'),(CE,ccls=='t-shirt','c')],
 'polo':[(XE,xcls=='Polo','c')],
 'tank':[(ZE,zt('tank top')&~cropped,'z'),(XE,xcls=='Undershirt','c')],
 'blouse':[(ZE,zt('blouse','tunic')&~cropped,'z'),(XE,xcls=='Blouse','c')],
 'shirt':[(ZE,zt('shirt')&~cropped,'z'),(CE,ccls=='shirt','c')],
 'sweater':[(ZE,zt('sweater')&~turtle&~cropped,'z')],
 'turtleneck':[(ZE,zt('sweater','top','blouse','t-shirt')&turtle,'z')],
 'hoodie':[(ZE,zt('hoodie'),'z'),(XE,xcls=='Hoodie','c')],
 'crop':[(ZE,zt('top','t-shirt','tank top','blouse')&cropped,'z')]})
OUTER=pool({
 'cardigan':[(ZE,zt('cardigan'),'z')],
 'blazer':[(ZE,zt('blazer'),'z'),(XE,xcls=='Blazer','c')],
 'jacket':[(ZE,zt('jacket','jacker','vest','outerwear','rain jacket','denim jacket','winter jacket'),'z'),(CE,ccls=='outwear','c')]})
BOTTOM=pool({
 'pants':[(ZE,zt('jeans','trousers','rain trousers','winter trousers'),'z'),(CE,ccls=='pants','c')],
 'shorts':[(ZE,zt('shorts'),'z'),(CE,ccls=='shorts','c')],
 'skirt':[(ZE,zt('skirt'),'z'),(CE,ccls=='skirt','c')]})
