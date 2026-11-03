/** Compare configurations on deterministic page-disjoint splits, with resumable results. */
const fs=require('node:fs'),path=require('node:path'),R=require('./runtime.cjs');
const data=JSON.parse(fs.readFileSync(path.join(R.CACHE,'dataset.json'))),args=process.argv.slice(2);
const split=args[0]||'val',limit=Number(args[1]||300),profile=args[2]||'select';
const profiles={select:[['gray_010',.1,{mode:'pad-gray'}],['gray_005',.05,{mode:'pad-gray'}],['width480_010',.1,{minWidth:480}],['gray_020',.2,{mode:'pad-gray'}]],
 confirm:[['production',.25,{}],['candidate',.1,{mode:'pad-gray'}]]};
if(!profiles[profile])throw new Error('Unknown comparison profile');
const output=path.join(__dirname,'results',`${profile}_${split}_${limit||'all'}.jsonl`);
async function main(){
 const api=R.ocr();console.log('runtime',JSON.stringify(await api.init()));
 const items=R.selectExamples(data,split,limit),rows=fs.existsSync(output)?fs.readFileSync(output,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[],done=new Set(rows.map(r=>r.id));
 for(const e of items){
  if(done.has(e.id))continue;
  const image=await R.loadImage(path.join(R.CACHE,e.crop)),d=await api.detect(image),variants={};
  for(const [name,conf,config]of profiles[profile]){const boxes=api.boxes(d,conf);variants[name]={...await api.recognize(image,boxes,config),boxes,line_count:boxes.length};}
  const row={...e,detect_ms:d.ms,variants};fs.appendFileSync(output,JSON.stringify(row)+'\n');rows.push(row);
  if(rows.length%50===0)console.log(JSON.stringify({progress:rows.length,total:items.length}));
 }
 const summary={n:rows.length,split,profile,source_revision:data.revisions,variants:{}};
 for(const [name,conf,config]of profiles[profile]){
  const flat=rows.map(r=>({...r,...r.variants[name]}));
  summary.variants[name]={conf,config,strict:R.score(flat),raw:R.score(flat,'raw'),post:R.score(flat,'post'),training:R.score(flat.map(r=>({...r,reference:R.trainingNormalize(r.reference),text:R.trainingNormalize(r.text)}))),no_lines:flat.filter(r=>!r.line_count).map(r=>r.id)};
 }
 fs.writeFileSync(output.replace('.jsonl','.summary.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1});
