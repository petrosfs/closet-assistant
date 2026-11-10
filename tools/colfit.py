import json,glob,os,re,numpy as np,collections,subprocess
from lib import W,DATA
lab={}
for f in glob.glob(DATA+'/zen/x/**/labels_*.json',recursive=True):
    try:lab[os.path.basename(f)[7:-5]]=json.load(open(f))
    except:pass
det=json.load(open(W+'/zen_col.json'));U=json.load(open(W+'/user_col.json'))
TR=[['white','beige'],['beige'],['blue'],['black'],['beige'],['beige'],['white','beige','green'],['white'],['beige','brown'],['navy','black'],['white','beige'],['orange','red'],['white','beige'],['pink'],['white','beige'],['black'],['orange','beige','brown'],['brown','green','grey'],['red','burgundy'],['black']]
MAP={'white':['white','beige'],'black':['black'],'blue':['blue','navy','teal'],'pink':['pink'],'grey':['grey'],'red':['red','burgundy'],'beige':['beige','white','brown'],'green':['green','teal'],'purple':['purple','burgundy'],'brown':['brown','beige'],'yellow':['yellow'],'orange':['orange','red'],'turquoise':['teal','blue','green']}
# palette (base + shades) from the app
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','index.html'),encoding='utf8').read()
base=dict(re.findall(r"(\w+):\['(#[0-9a-f]{6})',\d+,[01],'",src))
shade={k:(h,f) for k,h,f in re.findall(r"(\w+):\['(#[0-9a-f]{6})',\d+,'[^']*','[^']*','(\w+)'\]",src)}
def rgb2lab(h):
    r,g,b=[int(h[i:i+2],16)/255 for i in (1,3,5)]
    f=lambda v:((v+.055)/1.055)**2.4 if v>.04045 else v/12.92
    r,g,b=f(r),f(g),f(b);x=(r*.4124+g*.3576+b*.1805)/.95047;y=r*.2126+g*.7152+b*.0722;z=(r*.0193+g*.1192+b*.9505)/1.08883
    g2=lambda t:t**(1/3) if t>.008856 else 7.787*t+16/116
    return np.array([116*g2(y)-16,500*(g2(x)-g2(y)),200*(g2(y)-g2(z))])
def de00(p,q):
    import math
    C1=math.hypot(p[1],p[2]);C2=math.hypot(q[1],q[2]);Cb=(C1+C2)/2;G=.5*(1-math.sqrt(Cb**7/(Cb**7+25**7)))
    a1=(1+G)*p[1];a2=(1+G)*q[1];c1=math.hypot(a1,p[2]);c2=math.hypot(a2,q[2])
    h1=math.degrees(math.atan2(p[2],a1))%360;h2=math.degrees(math.atan2(q[2],a2))%360
    dL=q[0]-p[0];dC=c2-c1;dh=0
    if c1*c2:
        dh=h2-h1
        if dh>180:dh-=360
        elif dh<-180:dh+=360
    dH=2*math.sqrt(c1*c2)*math.sin(math.radians(dh/2));Lb=(p[0]+q[0])/2;cb=(c1+c2)/2;hb=h1+h2
    if c1*c2:hb=(h1+h2)/2 if abs(h1-h2)<=180 else ((h1+h2+360)/2 if h1+h2<360 else (h1+h2-360)/2)
    T=1-.17*math.cos(math.radians(hb-30))+.24*math.cos(math.radians(2*hb))+.32*math.cos(math.radians(3*hb+6))-.2*math.cos(math.radians(4*hb-63))
    dt=30*math.exp(-(((hb-275)/25)**2));Rc=2*math.sqrt(cb**7/(cb**7+25**7));Sl=1+.015*(Lb-50)**2/math.sqrt(20+(Lb-50)**2);Sc=1+.045*cb;Sh=1+.015*cb*T;Rt=-math.sin(math.radians(2*dt))*Rc
    return math.sqrt((dL/Sl)**2+(dC/Sc)**2+(dH/Sh)**2+Rt*(dC/Sc)*(dH/Sh))
BL={k:rgb2lab(h) for k,h in base.items()}
AL=dict(BL);FAM={k:k for k in base}
for k,(h,f) in shade.items():AL[k]=rgb2lab(h);FAM[k]=f
def near(l,pal):return min(pal,key=lambda k:de00(l,pal[k]))
rows=[]
for id,d in det.items():
    j=lab.get(id)
    if not j:continue
    cols=[str(c).lower() for c in (j.get('colors') or []) if str(c).lower()!='none']
    if len(cols)==1 and cols[0] in MAP:rows.append((id,cols[0],np.array(d['list'][0]['lab']),d['filled']))
if __name__=='__main__':
    print('single-colour items',len(rows),'base',len(base),'shades',len(shade))
    a=sum(near(l,BL) in MAP[c] for _,c,l,_ in rows)/len(rows);b=sum(FAM[near(l,AL)] in MAP[c] for _,c,l,_ in rows)/len(rows)
    print(f'palette base nearest {a:.3f} | all shades->family {b:.3f}')
    ua=sum(near(np.array(u[0]['lab']),BL) in TR[i] for i,u in enumerate(U));ub=sum(FAM[near(np.array(u[0]['lab']),AL)] in TR[i] for i,u in enumerate(U))
    print('user20: base',ua,'shades',ub,[ (i+1,FAM[near(np.array(u[0]["lab"]),AL)]) for i,u in enumerate(U) if FAM[near(np.array(u[0]['lab']),AL)] not in TR[i]])
    conf=collections.Counter((c,FAM[near(l,AL)]) for _,c,l,_ in rows if FAM[near(l,AL)] not in MAP[c]);print(conf.most_common(14))
