// usage: node embed_imgs.mjs <list.txt> <out-prefix>   (list: one image path per line)
import {AutoProcessor,CLIPVisionModelWithProjection,RawImage,env} from '@xenova/transformers';
import fs from 'fs';
env.allowLocalModels=false;env.cacheDir=process.env.CA_CACHE||'cache';
const M='Xenova/clip-vit-base-patch32';
const [,,list,out]=process.argv;
const files=fs.readFileSync(list,'utf8').split('\n').filter(Boolean);
const proc=await AutoProcessor.from_pretrained(M);
const vm=await CLIPVisionModelWithProjection.from_pretrained(M,{quantized:true});
const buf=new Float32Array(files.length*512);const ok=[];let t0=Date.now();
for(let i=0;i<files.length;i++){
 try{const im=await RawImage.read(files[i]);const e=(await vm(await proc(im))).image_embeds.data;let n=0;for(const x of e)n+=x*x;n=Math.sqrt(n)||1;
  for(let k=0;k<512;k++)buf[ok.length*512+k]=e[k]/n;ok.push(files[i])}catch(e){console.error('fail',files[i],e.message)}
 if(i%200===0)console.log(i,files.length,((Date.now()-t0)/1000).toFixed(0)+'s');
}
fs.writeFileSync(out+'.f32',Buffer.from(buf.buffer,0,ok.length*512*4));fs.writeFileSync(out+'.txt',ok.join('\n'));console.log('done',ok.length);
