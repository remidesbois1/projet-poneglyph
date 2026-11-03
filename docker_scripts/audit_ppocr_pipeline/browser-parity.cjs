/** Isolated headless Chromium + real Canvas + ONNX Runtime Web WASM.
 * No app server, credentials or existing browser profile is used.
 * Uses built-in WebSocket/CDP, so no browser automation package is required.
 */
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),{spawn}=require('node:child_process'),R=require('./runtime.cjs');
const dataset=JSON.parse(fs.readFileSync(path.join(R.CACHE,'dataset.json')));
const split=process.argv[2]||'targeted',limit=Number(process.argv[3]||100);
if(!['targeted','test','val'].includes(split)||!Number.isInteger(limit)||limit<1)throw new Error('Usage: browser-parity.cjs [targeted|test|val] [positive limit]');
const ids=split==='targeted'?[4351,8553,4249,5556,5568,2779,6171,2416,5412,1454,7254,5343,3545,8394,3878,483,4347,2606,6717,3562]:R.selectExamples(dataset,split,limit).map(e=>e.id);
const stem=split==='targeted'?'browser_parity':`browser_${split}_${limit}`;
const progressFile=path.join(__dirname,'results',stem+'.jsonl');
const previous=fs.existsSync(progressFile)?fs.readFileSync(progressFile,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse):[];
const completed=new Set(previous.map(r=>r.id)),remaining=ids.filter(id=>!completed.has(id));
const utils=fs.readFileSync(path.join(R.ROOT,'frontend/src/lib/ocr-utils.js'),'utf8').replace(/^export /gm,'');
const worker=fs.readFileSync(path.join(R.ROOT,'frontend/src/workers/ocr.worker.js'),'utf8').replace(/^import .*;\r?\n/gm,'').replace(/const (PPOCR_LINE_CONF|PPOCR_LINE_PAD|PPOCR_LINE_GAP) =/g,'let $1 =');
const source=`import * as ort from '/ort/ort.wasm.mjs';
ort.env.wasm.wasmPaths='/ort/';ort.env.wasm.numThreads=2;
const modelRegistry=${fs.readFileSync(path.join(R.ROOT,'packages/shared/src/model-registry.json'),'utf8')};
const _manifest=${JSON.stringify(R.manifest)},_models='/cache/models',_options={executionProviders:['wasm'],logSeverityLevel:3};let audit;
${utils}\n${worker}\n${R.ocrAppend}
ort.env.wasm.wasmPaths='/ort/';ort.env.wasm.numThreads=2;
window.auditState={status:'running',rows:[]};
(async()=>{const metadata=await audit.init();
for(const id of ${JSON.stringify(remaining)}){
 const image=await createImageBitmap(await(await fetch('/cache/crops/'+id+'.png')).blob());
 const d=await audit.detect(image),b25=audit.boxes(d,.25),b10=audit.boxes(d,.1);
 const production=await audit.recognize(image,b25),candidate=await audit.recognize(image,b10,{mode:'pad-gray'});
 const c=new OffscreenCanvas(image.width,image.height);c.getContext('2d').drawImage(image,0,0);
 const jpeg=await createImageBitmap(await c.convertToBlob({type:'image/jpeg',quality:.95}));
 const jd=await audit.detect(jpeg),jb=audit.boxes(jd,.1),jpegCandidate=await audit.recognize(jpeg,jb,{mode:'pad-gray'});
 window.auditState.rows.push({id,production,candidate,boxes25:b25,boxes10:b10,jpegCandidate,jpegBoxes:jb});image.close();jpeg.close();
}
window.auditState={...window.auditState,status:'complete',metadata,userAgent:navigator.userAgent,webgpu_api_available:!!navigator.gpu};
})().catch(e=>window.auditState={...window.auditState,status:'error',error:String(e.stack||e)});`;
let browser,server,ws;
async function main(){
 server=http.createServer((req,res)=>{
  res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
  if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><script type="module" src="/audit.mjs"></script>');return;}
  if(req.url==='/audit.mjs'){res.setHeader('Content-Type','text/javascript');res.end(source);return;}
  const url=decodeURIComponent(req.url.split('?')[0]);let file;
  if(url.startsWith('/cache/'))file=path.resolve(R.CACHE,url.slice(7));
  else if(url.startsWith('/ort/'))file=path.resolve(R.ROOT,'node_modules/onnxruntime-web/dist',url.slice(5));
  const allowed=file&&(file.startsWith(R.CACHE+path.sep)||file.startsWith(path.join(R.ROOT,'node_modules/onnxruntime-web/dist')+path.sep));
  if(!allowed||!fs.existsSync(file)){res.statusCode=404;res.end();return;}
  res.setHeader('Content-Type',file.endsWith('.mjs')?'text/javascript':file.endsWith('.wasm')?'application/wasm':file.endsWith('.png')?'image/png':'application/octet-stream');fs.createReadStream(file).pipe(res);
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const origin='http://127.0.0.1:'+server.address().port;
 const profile=path.join(R.CACHE,'chromium-'+stem+'-profile');fs.mkdirSync(profile,{recursive:true});
 browser=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',['--headless=new','--no-first-run','--no-default-browser-check','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe']});
 const endpoint=await new Promise((resolve,reject)=>{let log='';const timer=setTimeout(()=>reject(new Error('Chromium CDP startup timeout')),30000);browser.once('error',reject);browser.stderr.on('data',chunk=>{log+=chunk.toString();const m=log.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timer);resolve(m[1])}});});
 const port=new URL(endpoint).port,targets=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json(),target=targets.find(t=>t.type==='page');
 ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 let next=0;const pending=new Map();ws.onmessage=event=>{const m=JSON.parse(event.data);if(pending.has(m.id)){const {resolve,reject,timer}=pending.get(m.id);pending.delete(m.id);clearTimeout(timer);m.error?reject(new Error(JSON.stringify(m.error))):resolve(m.result)}};
 const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++next,timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout: '+method))},30000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}))});
 await send('Page.navigate',{url:origin});let state,last=-1;
 const deadline=Date.now()+Math.max(240000,remaining.length*20000);
 while(Date.now()<deadline){
  const r=await send('Runtime.evaluate',{expression:'window.auditState',returnByValue:true});state=r.result?.value;
  for(const row of state?.rows||[])if(!completed.has(row.id)){fs.appendFileSync(progressFile,JSON.stringify(row)+'\n');completed.add(row.id);previous.push(row);}
  if(state?.rows?.length!==last){last=state?.rows?.length??0;console.log(JSON.stringify({browser_progress:completed.size,total:ids.length,status:state?.status}));}
  if(state?.status==='error')throw new Error(state.error);
  if(state?.status==='complete')break;
  await new Promise(resolve=>setTimeout(resolve,1000));
 }
 if(state?.status!=='complete')throw new Error('Browser parity timeout');
 state.rows=previous.filter(r=>ids.includes(r.id));
 fs.writeFileSync(path.join(__dirname,'results',stem+'.json'),JSON.stringify(state,null,2));
 const ablation=new Map(fs.readFileSync(path.join(__dirname,'results/ablation_val_300.jsonl'),'utf8').trim().split('\n').map(JSON.parse).map(r=>[r.id,r]));
 const select=new Map(fs.readFileSync(path.join(__dirname,'results/select_val_300.jsonl'),'utf8').trim().split('\n').map(JSON.parse).map(r=>[r.id,r]));
 const nativeTestPath=path.join(__dirname,'results/confirm_test_all.jsonl');
 const nativeTest=fs.existsSync(nativeTestPath)?new Map(fs.readFileSync(nativeTestPath,'utf8').trim().split('\n').filter(Boolean).map(JSON.parse).map(r=>[r.id,r])):new Map();
 const summary={n:state.rows.length,split,sampling:split==='targeted'?'diagnostic cases':'deterministic hash sample',userAgent:state.userAgent,provider:'wasm',production_text_differences:[],candidate_text_differences:[],jpeg_text_differences:[],scores:{}};
 for(const r of state.rows){const baseline=split==='test'?nativeTest.get(r.id)?.variants.production.text:ablation.get(r.id)?.text,candidate=split==='test'?nativeTest.get(r.id)?.variants.candidate.text:select.get(r.id)?.variants.gray_010.text;if(baseline!==undefined&&r.production.text!==baseline)summary.production_text_differences.push({id:r.id,node:baseline,browser:r.production.text});if(candidate!==undefined&&r.candidate.text!==candidate)summary.candidate_text_differences.push({id:r.id,node:candidate,browser:r.candidate.text});if(r.jpegCandidate.text!==r.candidate.text)summary.jpeg_text_differences.push({id:r.id,png:r.candidate.text,jpeg:r.jpegCandidate.text});}
 for(const name of ['production','candidate','jpegCandidate'])summary.scores[name]=R.score(state.rows.map(r=>({...r[name],reference:dataset.examples.find(e=>e.id===r.id).reference})));
 fs.writeFileSync(path.join(__dirname,'results',stem+'.summary.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary,null,2));
 await send('Browser.close').catch(()=>{});
}
main().catch(e=>{console.error(e);process.exitCode=1}).finally(()=>{ws?.close();browser?.kill();server?.closeAllConnections();server?.close()});
