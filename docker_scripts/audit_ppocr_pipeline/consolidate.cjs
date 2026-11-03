/** Consolidate completed runs. No inference or remote access; fail on incomplete data. */
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),cp=require('node:child_process'),R=require('./runtime.cjs');
const dir=path.join(__dirname,'results');
const read=name=>JSON.parse(fs.readFileSync(path.join(dir,name),'utf8'));
const lines=name=>fs.readFileSync(path.join(dir,name),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const dataset=JSON.parse(fs.readFileSync(path.join(R.CACHE,'dataset.json'))),labels=JSON.parse(fs.readFileSync(path.join(__dirname,'visual-labels.json'))).labels;
const val=lines('ablation_val_300.jsonl'),selection=lines('select_val_300.jsonl'),test=lines('confirm_test_all.jsonl'),pages=lines('pages.jsonl'),pageOcr=lines('page_ocr_test.jsonl');
const browser=read('browser_test_100.json'),browserSummary=read('browser_test_100.summary.json');
for(const [name,n,expected]of [['validation',val.length,300],['selection',selection.length,300],['test',test.length,1219],['pages',pages.length,258],['page OCR',pageOcr.length,129],['browser',browser.rows.length,100]])if(n!==expected)throw new Error(`${name}: ${n}/${expected}, not complete`);
for(const [name,rows,key]of [['validation',val,'id'],['selection',selection,'id'],['test',test,'id'],['pages',pages,'id'],['page OCR',pageOcr,'page_id'],['browser',browser.rows,'id']])if(new Set(rows.map(r=>r[key])).size!==rows.length)throw new Error(`${name}: duplicate rows`);
const err=(ref,text,norm=R.normalize)=>R.distance(norm(ref),norm(text));
const flat=(rows,variant)=>rows.map(r=>({...r,...r.variants[variant]}));
const paired=(rows,a,b)=>{
 let improved=0,worsened=0,same=0,errorBefore=0,errorAfter=0,becameExact=0,lostExact=0;
 for(const r of rows){const x=err(r.reference,a(r)),y=err(r.reference,b(r));errorBefore+=x;errorAfter+=y;if(y<x)improved++;else if(y>x)worsened++;else same++;becameExact+=x>0&&y===0;lostExact+=x===0&&y>0;}
 return {n:rows.length,improved,worsened,same,error_before:errorBefore,error_after:errorAfter,became_exact:becameExact,lost_exact:lostExact};
};
const result={created_at:new Date().toISOString(),protocol:'Native ORT CPU+Skia; separate Chrome/WASM check. Historical page-disjoint splits, not a new blind holdout. CER is micro-averaged Levenshtein, exact is per bubble. Extra unmatched predictions not penalized by GT-conditioned page coverage scores.',
 snapshot:{created_at:dataset.created_at,validated_total:dataset.validated_total,pages:dataset.pages.length,examples:dataset.examples.length,validation:dataset.examples.filter(r=>r.split==='val').length,test:dataset.examples.filter(r=>r.split==='test').length,changed_references:dataset.examples.filter(r=>r.text_changed).length,invalid_bbox:dataset.examples.filter(r=>r.invalid_bbox).length,missing:dataset.missing.length,revisions:dataset.revisions,artifacts:dataset.artifacts},
 validation:read('ablation_val_300.summary.json'),selection:read('select_val_300.summary.json'),test:read('confirm_test_all.summary.json'),page_detection:read('pages.summary.json'),page_ocr:read('page_ocr_test.summary.json'),browser:browserSummary,diagnostic_browser:read('browser_parity.summary.json'),router:read('router.json'),classes:{},paired:{},postprocess:{},stage:{},review:{},provenance:{}};
delete result.router.samples;
const valMap=new Map(val.map(r=>[r.id,r])),selMap=new Map(selection.map(r=>[r.id,r])),testMap=new Map(test.map(r=>[r.id,r]));
for(const split of ['val','test']){
 result.classes[split]={};
 for(const label of ['ordinary','stylized_sfx','special_layout','ambiguous']){
  const group=labels.filter(l=>l.split===split&&l.label===label);
  const native=group.map(l=>split==='test'?testMap.get(l.id):{...valMap.get(l.id),variants:{production:valMap.get(l.id),candidate:selMap.get(l.id).variants.gray_010}});
  result.classes[split][label]={n:group.length,production:R.score(flat(native,'production')),candidate:R.score(flat(native,'candidate')),candidate_raw:R.score(flat(native,'candidate'),'raw'),candidate_post:R.score(flat(native,'candidate'),'post')};
  if(split==='test'){
   const ids=new Set(group.map(l=>l.id)),bs=browser.rows.filter(r=>ids.has(r.id)).map(r=>({...r,reference:testMap.get(r.id).reference}));
   result.classes[split][label].browser=Object.fromEntries(['production','candidate','jpegCandidate'].map(k=>[k,R.score(bs.map(r=>({...r,...r[k]})))]));
  }
 }
}
result.paired.test=paired(test,r=>r.variants.production.text,r=>r.variants.candidate.text);
const browserRows=browser.rows.map(r=>({...r,reference:testMap.get(r.id).reference}));
result.paired.browser_test=paired(browserRows,r=>r.production.text,r=>r.candidate.text);
result.paired.browser_jpeg=paired(browserRows,r=>r.candidate.text,r=>r.jpegCandidate.text);
result.paired.validation_threshold_010=paired(val,r=>r.text,r=>r.thresholds['0.1'].text);
result.paired.validation_gray_padding=paired(val,r=>r.text,r=>r.variants.pad_gray.text);
for(const variant of ['production','candidate']){
 const rows=flat(test,variant);
 result.postprocess[variant]={raw_to_lexicon:paired(rows,r=>r.raw,r=>r.post),lexicon_to_final:paired(rows,r=>r.post,r=>r.text),quote_stripping:rows.filter(r=>r.post.includes('"')&&!r.text.includes('"')).map(r=>({id:r.id,reference:r.reference,post:r.post,text:r.text}))};
}
const valPages=new Set(dataset.examples.filter(r=>r.split==='val').map(r=>r.page_id)),testPages=new Set(dataset.examples.filter(r=>r.split==='test').map(r=>r.page_id));
result.snapshot.val_test_page_overlap=[...valPages].filter(id=>testPages.has(id)).length;
result.snapshot.invalid_order_pages=pages.filter(p=>!p.matched_order).map(p=>p.id);
for(const split of ['val','test']){
 const ps=pages.filter(p=>p.split===split),total=ps.reduce((s,p)=>s+p.gt.length,0),matched=ps.reduce((s,p)=>s+p.matches.length,0),permissive=ps.reduce((s,p)=>s+p.permissive_matches.length,0);
 const order=ps.filter(p=>p.matched_order&&p.matched_order.pairs);
 result.stage[split]={pages:ps.length,gt:total,matched,permissive,missed:total-matched,missed_permissive:total-permissive,recall:matched/total,recall_permissive:permissive/total,valid_order_pages:order.length,exact_matched_order_pages:order.filter(p=>p.matched_order.exact).length};
 result.stage[split].unmatched_predictions=ps.reduce((s,p)=>s+p.ordered.length-p.matches.length,0);
 result.stage[split].unmatched_predictions_permissive=ps.reduce((s,p)=>s+p.permissive.length-p.permissive_matches.length,0);
 result.stage[split].recovered_bubbles=ps.flatMap(p=>{const base=new Set(p.matches.map(m=>m.i));return p.permissive_matches.filter(m=>!base.has(m.i)).map(m=>({id:p.gt[m.i].id,reference:p.gt[m.i].text}));});
}
const pageBubbles=pageOcr.flatMap(p=>p.bubbles),common=pageBubbles.filter(b=>testMap.has(b.id)),matchedCommon=common.filter(b=>b.detected),missing=pageBubbles.filter(b=>!b.detected);
result.stage.missing_bubbles={n:missing.length,reference_characters:missing.reduce((s,b)=>s+Array.from(R.normalize(b.reference)).length,0),ids:missing.map(b=>b.id)};
result.stage.common_test_subset={n:common.length,matched:matchedCommon.length,variants:{}};
for(const variant of ['production','candidate']){
 const crop=common.map(b=>({...b,...testMap.get(b.id).variants[variant]})),page=common.map(b=>({...b,...b[variant]}));
 const matchedCrop=matchedCommon.map(b=>({...b,...testMap.get(b.id).variants[variant]})),matchedPage=matchedCommon.map(b=>({...b,...b[variant]}));
 result.stage.common_test_subset.variants[variant]={gt_crops:R.score(crop),predicted_crops_with_misses:R.score(page),matched_gt_crops:R.score(matchedCrop),matched_predicted_crops:R.score(matchedPage),crop_geometry_change:paired(matchedCommon,b=>testMap.get(b.id).variants[variant].text,b=>b[variant].text)};
}
const orderValid=pageOcr.filter(p=>p.matched_order),allGtIds=new Set(test.map(r=>r.id));
result.stage.page_order_eligible=orderValid.length;
result.stage.extra_page_gt=pageBubbles.filter(b=>!allGtIds.has(b.id)).map(b=>({id:b.id,reference:b.reference}));
result.review.ordinary_test_failures=labels.filter(l=>l.split==='test'&&l.label==='ordinary').map(l=>testMap.get(l.id)).filter(r=>err(r.reference,r.variants.candidate.text)>0).map(r=>({id:r.id,reference:r.reference,raw:r.variants.candidate.raw,post:r.variants.candidate.post,text:r.variants.candidate.text,confidence:r.variants.candidate.confidence,errors:err(r.reference,r.variants.candidate.text),content_errors:err(r.reference,r.variants.candidate.text,R.contentNormalize),lines:r.variants.candidate.line_count}));
result.review.remaining_test_errors=test.map(r=>({id:r.id,reference:r.reference,production:r.variants.production.text,candidate:r.variants.candidate.text,confidence:r.variants.candidate.confidence,errors:err(r.reference,r.variants.candidate.text),content_errors:err(r.reference,r.variants.candidate.text,R.contentNormalize),lines:r.variants.candidate.line_count})).filter(r=>r.errors>0).sort((a,b)=>b.errors-a.errors).slice(0,40);
result.review.validation_line_recovery=val.filter(r=>!r.thresholds['0.1'].same).map(r=>({id:r.id,reference:r.reference,baseline:r.text,threshold_010:r.thresholds['0.1'].text,lines_before:r.boxes.length,lines_after:r.thresholds['0.1'].boxes.length,errors_before:err(r.reference,r.text),errors_after:err(r.reference,r.thresholds['0.1'].text)}));
result.review.native_empty_candidate=test.filter(r=>!r.variants.candidate.text).map(r=>({id:r.id,reference:r.reference,lines:r.variants.candidate.line_count}));
result.review.native_candidate_max_width=test.filter(r=>r.variants.candidate.input_width===3200).map(r=>({id:r.id,reference:r.reference,text:r.variants.candidate.text,errors:err(r.reference,r.variants.candidate.text)}));
result.review.confidence_diagnostic={protocol:'Descriptive only, uncalibrated mean CTC confidence on test GT crops. Thresholds are not selected or validated acceptance rules; missing bubbles are outside this population.',thresholds:[0,.9,.95,.98,.99,.995,.999].map(threshold=>{const selected=test.filter(r=>r.variants.candidate.confidence>=threshold);return {threshold,coverage:selected.length/test.length,...R.score(flat(selected,'candidate'))};})};
result.review.order_examples=pages.filter(p=>p.split==='test'&&p.gt_box_order&&!p.gt_box_order.exact).slice(0,5).map(p=>({id:p.id,ground_truth:p.gt.slice().sort((a,b)=>a.order-b.order).map(b=>b.id),with_gt_boxes:p.gt_order_ids,score:p.gt_box_order}));
result.provenance.git_head=cp.execFileSync('git',['rev-parse','HEAD'],{cwd:R.ROOT,encoding:'utf8'}).trim();
result.provenance.node=process.version;
result.provenance.source_sha256=Object.fromEntries(['frontend/src/workers/ocr.worker.js','frontend/src/workers/detection.worker.js','frontend/src/lib/ocr-utils.js','frontend/src/lib/utils.js','packages/shared/src/model-registry.json'].map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(path.join(R.ROOT,file))).digest('hex')]));
result.provenance.result_sha256=Object.fromEntries(['ablation_val_300.jsonl','select_val_300.jsonl','confirm_test_all.jsonl','pages.jsonl','page_ocr_test.jsonl','browser_test_100.json','browser_parity.json','router.json'].map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(path.join(dir,file))).digest('hex')]));
fs.writeFileSync(path.join(__dirname,'summary.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({file:'summary.json',paired:result.paired,classes:result.classes,stage:result.stage,postprocess:Object.fromEntries(Object.entries(result.postprocess).map(([k,v])=>[k,{raw_to_lexicon:v.raw_to_lexicon,lexicon_to_final:v.lexicon_to_final,quotes_stripped:v.quote_stripping.length}]))},null,2));
