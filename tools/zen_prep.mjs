// Crop the garment out of each "front" photo of the second-hand dataset, turn it upright, save a 448px JPEG.
import sharp from 'sharp';import fs from 'fs';import path from 'path';
import {lab} from './colorlib.mjs';
const [,,root,out]=process.argv;fs.mkdirSync(out,{recursive:true});
function walk(d,a=[]){for(const f of fs.readdirSync(d)){const p=path.join(d,f);if(fs.statSync(p).isDirectory())walk(p,a);else if(/front_.*\.jpg$/.test(f))a.push(p)}return a}
const files=walk(root);let done=0,fb=0;const meta=[];
for(const f of files){
 const id=path.basename(f).replace('front_','').replace('.jpg','');const dst=path.join(out,id+'.jpg');
 if(fs.existsSync(dst)){continue}
 try{
  const W=240,H=135;const {data,info}=await sharp(f).resize(W,H,{fit:'fill'}).removeAlpha().raw().toBuffer({resolveWithObject:true});
  const L=new Float32Array(W*H*3);for(let i=0;i<W*H;i++){const l=lab(data[i*3],data[i*3+1],data[i*3+2]);L[i*3]=l[0];L[i*3+1]=l[1];L[i*3+2]=l[2]}
  // background model: a quadratic surface fitted to the lightness (the table is lit unevenly), refitted twice without outliers
  const fit=(sel)=>{const A=Array.from({length:6},()=>new Float64Array(7));
   for(let y=0;y<H;y+=2)for(let x=0;x<W;x+=2){const i=y*W+x;if(sel&&!sel(i))continue;const u=x/W-.5,v=y/H-.5,f=[1,u,v,u*u,v*v,u*v],z=L[i*3];for(let a=0;a<6;a++){for(let b=0;b<6;b++)A[a][b]+=f[a]*f[b];A[a][6]+=f[a]*z}}
   for(let c=0;c<6;c++){let p=c;for(let r=c+1;r<6;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;[A[c],A[p]]=[A[p],A[c]];if(Math.abs(A[c][c])<1e-9)continue;for(let r=0;r<6;r++)if(r!==c){const k=A[r][c]/A[c][c];for(let j=c;j<7;j++)A[r][j]-=k*A[c][j]}}
   return A.map((r,i)=>r[6]/(r[i]||1))};
  const ev=(c,i)=>{const u=(i%W)/W-.5,v=((i/W)|0)/H-.5;return c[0]+c[1]*u+c[2]*v+c[3]*u*u+c[4]*v*v+c[5]*u*v};
  let co=fit(null);for(let r=0;r<3;r++){const c0=co;co=fit(i=>Math.abs(L[i*3]-ev(c0,i))<7&&Math.hypot(L[i*3+1],L[i*3+2])<9)}
  let sa=0,sb=0,sn=0;for(let i=0;i<W*H;i+=3)if(Math.abs(L[i*3]-ev(co,i))<5){sa+=L[i*3+1];sb+=L[i*3+2];sn++}
  const a0=sn?sa/sn:0,b0=sn?sb/sn:0;
  let m=new Uint8Array(W*H);for(let i=0;i<W*H;i++){m[i]=(Math.abs(L[i*3]-ev(co,i))>9||Math.hypot(L[i*3+1]-a0,L[i*3+2]-b0)>9)?1:0}
  // erode (removes the thin measuring tape and cables), then dilate back
  const er=new Uint8Array(W*H);for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++){let a=1;for(let dy=-1;dy<=1&&a;dy++)for(let dx=-1;dx<=1;dx++)if(!m[(y+dy)*W+x+dx]){a=0;break}er[y*W+x]=a}
  const lab2=new Int32Array(W*H);let best=null,nc=0;
  for(let s=0;s<W*H;s++){if(!er[s]||lab2[s])continue;nc++;const st=[s];lab2[s]=nc;let n=0,x0=W,x1=0,y0=H,y1=0;
   while(st.length){const p=st.pop(),x=p%W,y=(p/W)|0;n++;if(x<x0)x0=x;if(x>x1)x1=x;if(y<y0)y0=y;if(y>y1)y1=y;
    for(const q of [p-1,p+1,p-W,p+W]){if(q<0||q>=W*H)continue;if(Math.abs((q%W)-x)>1)continue;if(er[q]&&!lab2[q]){lab2[q]=nc;st.push(q)}}}
   // ignore blobs glued to the bottom edge (camera stand, cables) unless they are big
   const edge=y1>=H-3&&n<W*H*.05;if(!edge&&(!best||n>best.n))best={n,x0,x1,y0,y1}}
  let box;if(best&&best.n>W*H*.012){const px=Math.round((best.x1-best.x0)*.08)+3,py=Math.round((best.y1-best.y0)*.08)+3;
   box={x0:Math.max(0,best.x0-px),x1:Math.min(W-1,best.x1+px),y0:Math.max(0,best.y0-py),y1:Math.min(H-1,best.y1+py)}}
  else{fb++;box={x0:Math.round(W*.2),x1:Math.round(W*.8),y0:Math.round(H*.1),y1:Math.round(H*.9)}}
  const im=sharp(f);const md=await im.metadata();const sx=md.width/W,sy=md.height/H;
  const cut=await im.extract({left:Math.round(box.x0*sx),top:Math.round(box.y0*sy),width:Math.max(8,Math.min(md.width-Math.round(box.x0*sx),Math.round((box.x1-box.x0+1)*sx))),height:Math.max(8,Math.min(md.height-Math.round(box.y0*sy),Math.round((box.y1-box.y0+1)*sy)))}).toBuffer();
  await sharp(cut).rotate(-90).resize(448,448,{fit:'inside'}).jpeg({quality:88}).toFile(dst);
  done++;
 }catch(e){console.error('fail',f,e.message)}
 if(done%300===0)console.log(done,files.length);
}
console.log('done',done,'fallback crops',fb,'of',files.length);
