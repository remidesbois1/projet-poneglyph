/** Paired OCR on predicted page crops. Missing validated bubbles count as deletions.
 * Precision is deliberately not estimated: annotation completeness is unproven.
 * Canvas rasterization uses Skia; browser parity is checked separately.
 */
const fs=require('node:fs'),path=require('node:path'),R=require('./runtime.cjs');
const data=JSON.parse(fs.readFileSync(path.join(R.CACHE,'dataset.json')));
const pages=fs.readFileSync(path.join(__dirname,'results/pages.jsonl'),'utf8').trim().split('\n').map(JSON.parse).filter(p=>p.split==='test');
const output=path.join(__dirname,'results/page_ocr_test.jsonl');
async function main(){
 const api=R.ocr();console.log('runtime',JSON.stringify(await api.init()));
 const rows=fs.existsSync(output)?fs.readFileSync(output,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[],done=new Set(rows.map(r=>r.page_id));
 for(const page of pages){
  if(done.has(page.id))continue;
  const source=data.pages.find(p=>p.id===page.id),image=await R.loadImage(path.join(R.CACHE,source.file));
  const predictions=[];
  for(let index=0;index<page.ordered.length;index++){
   const box=page.ordered[index],w=Math.max(1,Math.round(box.w)),h=Math.max(1,Math.round(box.h));
   const crop=R.createCanvas(w,h),ctx=crop.getContext('2d');ctx.drawImage(image,Math.round(box.x),Math.round(box.y),w,h,0,0,w,h);
   const d=await api.detect(crop),b25=api.boxes(d,.25),b10=api.boxes(d,.1);
   predictions.push({index,box,detect_ms:d.ms,production:{...await api.recognize(crop,b25),boxes:b25},candidate:{...await api.recognize(crop,b10,{mode:'pad-gray'}),boxes:b10}});
  }
  const matched=new Map(page.matches.map(m=>[m.i,m.j]));
  const bubbles=page.gt.map((gt,i)=>({id:gt.id,reference:gt.text,detected:matched.has(i),prediction_index:matched.get(i)??null,...(matched.has(i)?predictions[matched.get(i)]:{production:{raw:'',post:'',text:''},candidate:{raw:'',post:'',text:''}})}));
  const row={page_id:page.id,bubbles,predictions,matched_order:page.matched_order};rows.push(row);fs.appendFileSync(output,JSON.stringify(row)+'\n');
  if(rows.length%10===0)console.log(JSON.stringify({pages:rows.length,total:pages.length,bubbles:rows.reduce((s,r)=>s+r.bubbles.length,0)}));
 }
 const bubbles=rows.flatMap(r=>r.bubbles),summary={pages:rows.length,bubbles:bubbles.length,detected:bubbles.filter(b=>b.detected).length,variants:{}};
 for(const name of ['production','candidate']){
  const flat=bubbles.map(b=>({...b,...b[name]}));
  summary.variants[name]={all:R.score(flat),matched:R.score(flat.filter(b=>b.detected)),post:R.score(flat,'post'),page_text_exact:rows.filter(r=>r.bubbles.every(b=>b.detected&&R.normalize(b.reference)===R.normalize(b[name].text))).length,
    page_content_exact:rows.filter(r=>r.bubbles.every(b=>b.detected&&R.contentNormalize(b.reference)===R.contentNormalize(b[name].text))).length,
    page_text_and_order_exact:rows.filter(r=>r.matched_order?.exact&&r.bubbles.every(b=>b.detected&&R.normalize(b.reference)===R.normalize(b[name].text))).length};
 }
 fs.writeFileSync(output.replace('.jsonl','.summary.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1});
