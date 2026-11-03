/** Full-page detection and reading order audit. Only validated annotations are scored. */
const fs=require('node:fs'),path=require('node:path'),R=require('./runtime.cjs');
const data=JSON.parse(fs.readFileSync(path.join(R.CACHE,'dataset.json')));
const dir=path.join(__dirname,'results');fs.mkdirSync(dir,{recursive:true});
const output=path.join(dir,'pages.jsonl');
function orderScore(ids,gt){
 const ranks=new Map(gt.map(b=>[b.id,b.order]));
 if(gt.some(b=>b.order===null||b.order===undefined)||new Set(gt.map(b=>b.order)).size!==gt.length)return null;
 let pairs=0,inversions=0;for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){pairs++;inversions+=Number(ranks.get(ids[i])>ranks.get(ids[j]));}
 return {pairs,inversions,exact:inversions===0};
}
async function main(){
 const api=R.detection();console.log('runtime',JSON.stringify(await api.init()));
 const rows=fs.existsSync(output)?fs.readFileSync(output,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];
 const done=new Set(rows.map(r=>r.id));
 for(const page of data.pages){
  if(done.has(page.id))continue;
  const image=await R.loadImage(path.join(R.CACHE,page.file));
  const pred=await api.run(image),matches=R.matchBoxes(page.bubbles,pred.ordered),permissive=R.matchBoxes(page.bubbles,pred.permissive);
  const matchedOrder=matches.slice().sort((a,b)=>a.j-b.j).map(m=>page.bubbles[m.i].id);
  const gtOrdered=await api.orderGt(image,page.bubbles,pred.panels);
  const row={id:page.id,split:data.examples.find(e=>e.page_id===page.id).split,width:page.width,height:page.height,
   gt:page.bubbles,...pred,matches,permissive_matches:permissive,matched_order:orderScore(matchedOrder,page.bubbles),gt_box_order:orderScore(gtOrdered.map(b=>b.id),page.bubbles),gt_order_ids:gtOrdered.map(b=>b.id)};
  fs.appendFileSync(output,JSON.stringify(row)+'\n');rows.push(row);
  if(rows.length%25===0)console.log(JSON.stringify({pages:rows.length,total:data.pages.length,gt:rows.reduce((s,r)=>s+r.gt.length,0),matched:rows.reduce((s,r)=>s+r.matches.length,0)}));
 }
 const summary={n:rows.length,gt:rows.reduce((s,r)=>s+r.gt.length,0),detected:rows.reduce((s,r)=>s+r.ordered.length,0),matched:rows.reduce((s,r)=>s+r.matches.length,0),matched_permissive:rows.reduce((s,r)=>s+r.permissive_matches.length,0),order:{}};
 for(const name of ['matched_order','gt_box_order']){let rs=rows.filter(r=>r[name]&&r[name].pairs);summary.order[name]={pages:rs.length,exact:rs.filter(r=>r[name].exact).length,pairs:rs.reduce((s,r)=>s+r[name].pairs,0),inversions:rs.reduce((s,r)=>s+r[name].inversions,0)};}
 fs.writeFileSync(path.join(dir,'pages.summary.json'),JSON.stringify(summary,null,2));console.log(summary);
}
main().catch(e=>{console.error(e);process.exitCode=1});
