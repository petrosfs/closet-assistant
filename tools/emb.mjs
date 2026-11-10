import {AutoProcessor,CLIPVisionModelWithProjection,AutoTokenizer,CLIPTextModelWithProjection,RawImage,env} from '@xenova/transformers';
import fs from 'fs';
env.allowLocalModels=false;env.cacheDir=process.env.CA_CACHE||'cache';
const M='Xenova/clip-vit-base-patch32';
const files=fs.readFileSync('files.txt','utf8').split('\n');
const proc=await AutoProcessor.from_pretrained(M);
const vm=await CLIPVisionModelWithProjection.from_pretrained(M,{quantized:true});
const tok=await AutoTokenizer.from_pretrained(M);
const tm=await CLIPTextModelWithProjection.from_pretrained(M,{quantized:true});
const l2=a=>{const n=Math.hypot(...a)||1;return Array.from(a,x=>x/n)};
if(!fs.existsSync('imgs.json')){
 const out={};
 for(const f of files){const im=await RawImage.read('photos/'+f);out[f]=l2((await vm(await proc(im))).image_embeds.data)}
 fs.writeFileSync('imgs.json',JSON.stringify(out));console.log('images',Object.keys(out).length)}
const prompts=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
let cache=fs.existsSync('texts.json')?JSON.parse(fs.readFileSync('texts.json','utf8')):{};
const need=prompts.filter(p=>!cache[p]);
for(let i=0;i<need.length;i+=32){const b=need.slice(i,i+32);const t=await tm(tok(b,{padding:true,truncation:true}));const D=t.text_embeds.dims[1];b.forEach((p,k)=>cache[p]=l2(t.text_embeds.data.slice(k*D,(k+1)*D)))}
fs.writeFileSync('texts.json',JSON.stringify(cache));console.log('texts',Object.keys(cache).length);
