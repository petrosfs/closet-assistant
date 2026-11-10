import {AutoProcessor,CLIPVisionModelWithProjection,RawImage,env} from '@xenova/transformers';
import fs from 'fs';env.allowLocalModels=false;env.cacheDir='cache';
const M='Xenova/clip-vit-base-patch32';
const proc=await AutoProcessor.from_pretrained(M),vm=await CLIPVisionModelWithProjection.from_pretrained(M,{quantized:true});
const files=fs.readFileSync('cal_files.txt','utf8').split('\n');const buf=new Float32Array(files.length*512);
const l2=a=>{const n=Math.hypot(...a)||1;return Array.from(a,x=>x/n)};
let t=Date.now();
for(let i=0;i<files.length;i++){const e=l2((await vm(await proc(await RawImage.read(files[i])))).image_embeds.data);buf.set(e,i*512);if(i%100===0){console.log(i,((Date.now()-t)/1000).toFixed(0)+'s');}}
fs.writeFileSync('cal.f32',Buffer.from(buf.buffer));console.log('done');
