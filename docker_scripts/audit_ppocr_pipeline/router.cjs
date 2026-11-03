/** Exploratory single-threshold router fitted on 100 visually reviewed validation
 * crops, then frozen and evaluated on 100 disjoint reviewed test crops.
 * It predicts unusual typography/layout, not the semantic presence of a sound.
 * Geometry only: no transcript, OCR output, text length, or test labels at fit time.
 */
const fs=require('node:fs'),path=require('node:path'),R=require('./runtime.cjs');
const read=name=>fs.readFileSync(path.join(__dirname,'results',name),'utf8').trim().split('\n').map(JSON.parse);
const labels=JSON.parse(fs.readFileSync(path.join(__dirname,'visual-labels.json'))).labels;
const val=new Map(read('ablation_val_300.jsonl').map(r=>[r.id,r]));
const test=new Map(read('confirm_test_all.jsonl').map(r=>[r.id,{...r,boxes:r.variants.production.boxes}]));
function features(row){
 const a=row.boxes.map(b=>(b.x2-b.x1)/Math.max(b.y2-b.y1,1)).sort((a,b)=>a-b);
 return {crop_aspect:row.bbox[2]/row.bbox[3],line_count:a.length,min_line_aspect:a[0]??0,median_line_aspect:a[Math.floor(a.length/2)]??0,mean_line_aspect:a.reduce((a,b)=>a+b,0)/Math.max(a.length,1),max_line_aspect:a.at(-1)??0};
}
const samples=labels.filter(l=>l.label!=='ambiguous').map(l=>{const row=(l.split==='val'?val:test).get(l.id);if(!row)throw new Error('Missing completed inference for '+l.id);return {...l,positive:l.label!=='ordinary',features:features(row)}});
const train=samples.filter(r=>r.split==='val'),holdout=samples.filter(r=>r.split==='test');
const predicts=(rule,r)=>rule.direction==='lt'?r.features[rule.feature]<rule.threshold:r.features[rule.feature]>rule.threshold;
function confusion(rule,rows){let tp=0,fp=0,tn=0,fn=0;const false_skip_ids=[],missed_ids=[],routed_ids=[];
 for(const r of rows){const p=predicts(rule,r);if(p)routed_ids.push(r.id);if(p&&r.positive)tp++;else if(p){fp++;false_skip_ids.push(r.id)}else if(r.positive){fn++;missed_ids.push(r.id)}else tn++;}
 return {n:rows.length,tp,fp,tn,fn,precision:tp/Math.max(tp+fp,1),recall:tp/Math.max(tp+fn,1),ordinary_false_skip_rate:fp/Math.max(fp+tn,1),false_skip_ids,missed_ids,routed_ids};
}
const candidates=[];
for(const feature of Object.keys(train[0].features)){
 const values=[...new Set(train.map(r=>r.features[feature]))].sort((a,b)=>a-b);
 const thresholds=[values[0]-1,...values.slice(1).map((v,i)=>(v+values[i])/2),values.at(-1)+1];
 for(const threshold of thresholds)for(const direction of ['lt','gt']){const rule={feature,threshold,direction},score=confusion(rule,train);if(!score.fp)candidates.push({rule,score});}
}
candidates.sort((a,b)=>b.score.tp-a.score.tp||a.rule.feature.localeCompare(b.rule.feature)||a.rule.threshold-b.rule.threshold);
const chosen=candidates[0],result={protocol:'Exploratory stump; fitted on 99 reviewed validation images; held-out evaluation on 99 reviewed test images; ambiguous excluded. One visual annotator. Selection maximizes recall subject to zero observed ordinary false routes on fit sample.',rule:chosen.rule,fit:chosen.score,test:confusion(chosen.rule,holdout),samples};
fs.writeFileSync(path.join(__dirname,'results/router.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({rule:result.rule,fit:result.fit,test:result.test},null,2));
