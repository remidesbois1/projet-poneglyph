const fs=require('node:fs'),path=require('node:path');
const R=require('./runtime.cjs');
const DATA=JSON.parse(fs.readFileSync(path.join(R.CACHE,'dataset.json')));
const RESULTS=path.join(__dirname,'results');fs.mkdirSync(RESULTS,{recursive:true});
const args=process.argv.slice(2);const option=(name,fallback)=>{const i=args.indexOf('--'+name);return i<0?fallback:args[i+1];};
const split=option('split','test'),limit=Number(option('limit','0')),mode=option('mode','baseline');
const output=path.join(RESULTS,option('output',`${mode}_${split}${limit?'_'+limit:''}.jsonl`));

async function main(){
 const api=R.ocr();console.log('runtime',JSON.stringify(await api.init()));
 const items=R.selectExamples(DATA,split,limit),existing=fs.existsSync(output)?fs.readFileSync(output,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];
 const done=new Set(existing.map(r=>r.id));let count=0;
 const configs=mode==='ablation'?[
  ['baseline',{}],['pad_white',{mode:'pad-white'}],['pad_gray',{mode:'pad-gray'}],
  ['width480',{minWidth:480}],['width640',{minWidth:640}],['pad4',{pad:4}],
 ]:[['baseline',{}]];
 for(const e of items){
  if(done.has(e.id))continue;
  const image=await R.loadImage(path.join(R.CACHE,e.crop));
  const d=await api.detect(image),boxes=api.boxes(d,.25);
  const results={};
  for(const [name,config]of configs)results[name]=await api.recognize(image,boxes,config);
  const permissive={};
  for(const conf of [.2,.1,.05]){
   const b=api.boxes(d,conf);const same=JSON.stringify(b)===JSON.stringify(boxes);
   permissive[String(conf)]={boxes:b,same,...(same?results.baseline:await api.recognize(image,b))};
  }
  const result={...e,...results.baseline,line_count:boxes.length,boxes,detect_ms:d.ms,variants:results,thresholds:permissive};
  fs.appendFileSync(output,JSON.stringify(result)+'\n');existing.push(result);
  if(++count%25===0)console.log(JSON.stringify({progress:existing.length,total:items.length,score:R.score(existing),mean_ms:existing.reduce((s,r)=>s+r.ms+r.detect_ms,0)/existing.length}));
 }
 const summary={split,mode,source_revision:DATA.revisions,n:existing.length,baseline:R.score(existing),raw:R.score(existing,'raw'),post:R.score(existing,'post'),long_dialogue_proxy:R.score(existing.filter(r=>r.reference.length>=30)),short_proxy:R.score(existing.filter(r=>r.reference.length<=12)),no_lines:existing.filter(r=>r.line_count===0).map(r=>r.id),variants:{},thresholds:{}};
 for(const [name]of configs)summary.variants[name]=R.score(existing.map(r=>({...r,...r.variants[name]})));
 for(const conf of ['0.2','0.1','0.05'])summary.thresholds[conf]={...R.score(existing.map(r=>({...r,...r.thresholds[conf]}))),changed_boxes:existing.filter(r=>!r.thresholds[conf].same).length};
 fs.writeFileSync(output.replace('.jsonl','.summary.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
