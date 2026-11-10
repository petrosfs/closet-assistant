// ---- shared palette (copied from the app) ----
export const BASEP={black:'#1b1b1b',white:'#f6f6f4',grey:'#9b9b9b',beige:'#d9c6a5',brown:'#7b4f2e',navy:'#1f2b5c',blue:'#4b86d6',teal:'#1a9aa5',green:'#4c8c3b',yellow:'#f2cf3a',orange:'#ee8a24',red:'#d02a2a',burgundy:'#7a1f3d',pink:'#f08fb4',purple:'#7f55c4'};
export const hex2rgb=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
export function lab(r,g,b){[r,g,b]=[r,g,b].map(v=>{v/=255;return v>.04045?Math.pow((v+.055)/1.055,2.4):v/12.92});
 const x=(r*.4124+g*.3576+b*.1805)/.95047,y=r*.2126+g*.7152+b*.0722,z=(r*.0193+g*.1192+b*.9505)/1.08883;
 const f=t=>t>.008856?Math.cbrt(t):7.787*t+16/116;return [116*f(y)-16,500*(f(x)-f(y)),200*(f(y)-f(z))]}
export const d76=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
export function dE00(p,q){const rad=Math.PI/180,deg=180/Math.PI;
 const C1=Math.hypot(p[1],p[2]),C2=Math.hypot(q[1],q[2]),Cb=(C1+C2)/2,G=.5*(1-Math.sqrt(Cb**7/(Cb**7+25**7)));
 const a1=(1+G)*p[1],a2=(1+G)*q[1],c1=Math.hypot(a1,p[2]),c2=Math.hypot(a2,q[2]);
 let h1=Math.atan2(p[2],a1)*deg;if(h1<0)h1+=360;let h2=Math.atan2(q[2],a2)*deg;if(h2<0)h2+=360;
 const dL=q[0]-p[0],dC=c2-c1;let dh=0;if(c1*c2){dh=h2-h1;if(dh>180)dh-=360;else if(dh<-180)dh+=360}
 const dH=2*Math.sqrt(c1*c2)*Math.sin(dh*rad/2),Lb=(p[0]+q[0])/2,cb=(c1+c2)/2;
 let hb=h1+h2;if(c1*c2){hb=Math.abs(h1-h2)<=180?(h1+h2)/2:(h1+h2<360?(h1+h2+360)/2:(h1+h2-360)/2)}
 const T=1-.17*Math.cos((hb-30)*rad)+.24*Math.cos(2*hb*rad)+.32*Math.cos((3*hb+6)*rad)-.2*Math.cos((4*hb-63)*rad);
 const dt=30*Math.exp(-(((hb-275)/25)**2)),Rc=2*Math.sqrt(cb**7/(cb**7+25**7)),Sl=1+.015*(Lb-50)**2/Math.sqrt(20+(Lb-50)**2),Sc=1+.045*cb,Sh=1+.015*cb*T,Rt=-Math.sin(2*dt*rad)*Rc;
 return Math.sqrt((dL/Sl)**2+(dC/Sc)**2+(dH/Sh)**2+Rt*(dC/Sc)*(dH/Sh))}
// ---- OLD algorithm (as in the app now) ----
export function detectOld(d,N){
 const PL=Object.fromEntries(Object.entries(BASEP).map(([k,v])=>[k,lab(...hex2rgb(v))]));
 const nearest=rgb=>{const l=lab(...rgb);let best=null,bd=1e9;for(const k in PL){const p=PL[k],e=(l[0]-p[0])**2+(l[1]-p[1])**2+(l[2]-p[2])**2;if(e<bd){bd=e;best=k}}return best};
 const px=(i,j)=>{const o=(j*N+i)*4;return [d[o],d[o+1],d[o+2]]};
 const bd=[];for(let i=0;i<N;i++)for(const j of [0,1,N-2,N-1]){bd.push(px(i,j),px(j,i))}
 const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
 const m=[0,1,2].map(k=>bd.reduce((s,p)=>s+p[k],0)/bd.length);
 const uniform=bd.reduce((s,p)=>s+dist(p,m),0)/bd.length<28;
 const all=[],kept=[];
 for(let i=6;i<N-6;i++)for(let j=6;j<N-6;j++){const p=px(i,j);all.push(p);if(!(uniform&&dist(p,m)<45))kept.push(p)}
 const use=kept.length>=all.length*.08?kept:all;
 const B={};for(const p of use){const k=(p[0]>>5)<<6|(p[1]>>5)<<3|(p[2]>>5);const b=B[k]||(B[k]={n:0,s:[0,0,0]});b.n++;b.s[0]+=p[0];b.s[1]+=p[1];b.s[2]+=p[2]}
 let top=null;for(const k in B)if(!top||B[k].n>top.n)top=B[k];
 const rgb=top.s.map(v=>v/top.n);return {fam:nearest(rgb),rgb:rgb.map(Math.round)}}
