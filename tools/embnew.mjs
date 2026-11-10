import {AutoTokenizer,CLIPTextModelWithProjection,env} from '@xenova/transformers';
import fs from 'fs';env.allowLocalModels=false;env.cacheDir='cache';
const M='Xenova/clip-vit-base-patch32';const tok=await AutoTokenizer.from_pretrained(M),tm=await CLIPTextModelWithProjection.from_pretrained(M,{quantized:true});
const l2=a=>{const n=Math.hypot(...a)||1;return Array.from(a,x=>x/n)};
const all=JSON.parse(fs.readFileSync('newp.json','utf8'));const out={};
for(let i=0;i<all.length;i+=32){const b=all.slice(i,i+32);const t=await tm(tok(b,{padding:true,truncation:true}));const D=t.text_embeds.dims[1];b.forEach((p,k)=>out[p]=l2(t.text_embeds.data.slice(k*D,(k+1)*D)))}
fs.writeFileSync('newtx.json',JSON.stringify(out));console.log(Object.keys(out).length);
