/** Execute the application's actual worker functions using native ONNX CPU and Skia.
 * Only configurable constants and dependency imports are adapted in memory.
 * Chromium validation is separate: Skia/ORT Node is not a browser-speed benchmark.
 */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ort = require('onnxruntime-node');
const { createCanvas, loadImage } = require('./cache/runtime/node_modules/@napi-rs/canvas');
const ROOT = path.resolve(__dirname, '../..');
const CACHE = path.join(__dirname, 'cache');
const modelRegistry = JSON.parse(fs.readFileSync(path.join(ROOT, 'packages/shared/src/model-registry.json')));
const manifest = JSON.parse(fs.readFileSync(path.join(CACHE, 'models/browser_manifest.json')));
const sessionOptions = { executionProviders: ['cpu'], graphOptimizationLevel: 'all', intraOpNumThreads: 4, interOpNumThreads: 1, logSeverityLevel: 3 };

function contextFor(filename, appended) {
  let source = fs.readFileSync(path.join(ROOT, filename), 'utf8').replace(/^import .*;\r?\n/gm, '');
  source = source.replace(/const (PPOCR_LINE_CONF|PPOCR_LINE_PAD|PPOCR_LINE_GAP|BUBBLE_SCORE_THRESHOLD) =/g, 'let $1 =');
  const context = vm.createContext({
    ort: { Tensor: ort.Tensor, InferenceSession: ort.InferenceSession, env: { wasm: {}, webgpu: {} } },
    OffscreenCanvas: function (w, h) { return createCanvas(w, h); },
    self: { location: { origin: 'http://localhost' }, navigator: { hardwareConcurrency: 4 }, addEventListener() {}, postMessage() {} },
    modelRegistry, console, URL, performance, Float32Array, Uint8Array, Map, Set, Math,
    setTimeout, clearTimeout, fetch, _manifest: manifest, _options: sessionOptions,
    _models: path.join(CACHE, 'models').replaceAll('\\', '/'),
  });
  const utils = fs.readFileSync(path.join(ROOT, 'frontend/src/lib/ocr-utils.js'), 'utf8').replace(/^export /gm, '');
  vm.runInContext(utils + '\n' + source + '\n' + appended, context, { filename });
  return context.audit;
}

const ocrAppend = `
function auditDecode(output) {
    const vocab = ppocrManifest.character_list.length;
    let prev = -1; const probabilities = []; let text = '';
    for (let t = 0; t < output.length; t += vocab) {
        let best = 0, value = -Infinity, total = 0;
        for (let k = 0; k < vocab; k++) { const v = output[t + k]; if (v > value) { best = k; value = v; } total += v; }
        if (best !== 0 && best !== prev) { text += ppocrManifest.character_list[best]; probabilities.push(value); }
        prev = best;
    }
    return { raw: text, confidence: probabilities.length ? probabilities.reduce((a,b)=>a+b,0)/probabilities.length : 0,
        min_confidence: probabilities.length ? Math.min(...probabilities) : 0 };
}
function auditTensor(canvas, mode, minWidth) {
    const h = ppocrManifest.image_height;
    const aspect = Math.ceil(h * canvas.width / canvas.height);
    const width = Math.min(ppocrManifest.max_image_width, Math.max(minWidth, aspect));
    const out = new OffscreenCanvas(width,h); const ctx = out.getContext('2d');
    ctx.fillStyle = mode === 'pad-gray' ? '#808080' : '#ffffff'; ctx.fillRect(0,0,width,h);
    ctx.drawImage(canvas,0,0,mode.startsWith('pad') ? Math.min(width,aspect) : width,h);
    const rgba=ctx.getImageData(0,0,width,h).data, n=width*h, data=new Float32Array(3*n);
    for(let i=0;i<n;i++){ data[i]=(rgba[4*i+2]/255-.5)/.5; data[n+i]=(rgba[4*i+1]/255-.5)/.5; data[2*n+i]=(rgba[4*i]/255-.5)/.5; }
    return new ort.Tensor('float32',data,[1,3,h,width]);
}
audit = {
  async init() {
    ppocrManifest=_manifest;
    ppocrLineDetectorSession=await ort.InferenceSession.create(_models+'/bubble_line_detector_yolo26n.onnx',_options);
    ppocrRecSession=await ort.InferenceSession.create(_models+'/ppocrv6_bubble_line_rec_webgpu.onnx',_options);
    return { detector: ppocrLineDetectorSession.inputMetadata, recognizer: ppocrRecSession.inputMetadata, size: detectorInputSize(ppocrLineDetectorSession) };
  },
  async detect(image) {
    const t=performance.now(); const {height,width}=detectorInputSize(ppocrLineDetectorSession);
    const p=preprocessYoloBitmap(image,height,width);
    const r=await ppocrLineDetectorSession.run({[ppocrLineDetectorSession.inputNames[0]]:p.inputTensor});
    return { output:Array.from(r[ppocrLineDetectorSession.outputNames[0]].data),scale:p.scale,padX:p.padX,padY:p.padY,width:image.width,height:image.height,ms:performance.now()-t };
  },
  boxes(d,conf=.25) { PPOCR_LINE_CONF=conf; return postprocessLineDetections(d.output,d.scale,d.padX,d.padY,d.width,d.height); },
  async recognize(image, boxes, config={}) {
    const start=performance.now(); PPOCR_LINE_PAD=config.pad??2; PPOCR_LINE_GAP=config.gap??8;
    const stitched=stitchLineBoxes(image,boxes);
    const tensor=config.mode || config.minWidth ? auditTensor(stitched,config.mode||'stretch',config.minWidth||960) : preprocessPpocrCanvas(stitched);
    const result=await ppocrRecSession.run({[ppocrRecSession.inputNames[0]]:tensor});
    const decoded=auditDecode(result[ppocrRecSession.outputNames[0]].data);
    if(decoded.raw !== decodePpocrCtc(result[ppocrRecSession.outputNames[0]].data)) throw new Error('Audit CTC decoder differs from production');
    const post=postprocessPpocrText(decoded.raw);
    return {...decoded, post, text:fixFrenchPunctuation(post), ms:performance.now()-start, stitched_width:stitched.width, stitched_height:stitched.height, input_width:tensor.dims[3]};
  },
  post(raw) { return {raw,post:postprocessPpocrText(raw),text:fixFrenchPunctuation(postprocessPpocrText(raw))}; },
  stitch: stitchLineBoxes,
};
`;

const detectionAppend = `
audit = {
 async init() {
    bubbleSession=await ort.InferenceSession.create(_models+'/bubble_detector.onnx',_options);
    panelSession=await ort.InferenceSession.create(_models+'/panel_detector.onnx',_options);
    orderingSession=await ort.InferenceSession.create(_models+'/ordering.onnx',_options);
    return {bubble:bubbleSession.inputMetadata,panel:panelSession.inputMetadata};
 },
 async run(image,conf=.45) {
    const start=performance.now(); BUBBLE_SCORE_THRESHOLD=conf;
    const {height,width}=getImageInputSize(bubbleSession,800,800);
    const p=preprocessBubble(image,height,width);
    const r=await bubbleSession.run({[bubbleSession.inputNames[0]]:p.inputTensor});
    const raw=r[bubbleSession.outputNames[0]].data;
    const boxes=simplifyPostProcess(raw,p.scale,p.padX,p.padY);
    BUBBLE_SCORE_THRESHOLD=.25; const permissive=simplifyPostProcess(raw,p.scale,p.padX,p.padY); BUBBLE_SCORE_THRESHOLD=conf;
    const panels=await detectPanels(image);
    const orderedPanels=await rankPanels(panels,image);
    let ordered=boxes;
    if(boxes.length>1&&orderedPanels.length) ordered=(await sortBubblesWithReadingOrderDetails(boxes,orderedPanels,assignBubblesToPanelsDetailed(boxes,orderedPanels,image),image)).boxes;
    else if(boxes.length>1) ordered=mangaOrderSort(boxes);
    return {boxes,permissive,ordered,panels:orderedPanels,ms:performance.now()-start};
 },
 async orderGt(image, boxes, panels) {
    if(!panels.length) return mangaOrderSort(boxes);
    return (await sortBubblesWithReadingOrderDetails(boxes,panels,assignBubblesToPanelsDetailed(boxes,panels,image),image)).boxes;
 }
};
`;

function distance(a,b) {
  a=Array.from(a); b=Array.from(b); let prev=Array.from({length:b.length+1},(_,i)=>i);
  for(let i=0;i<a.length;i++){const row=[i+1];for(let j=0;j<b.length;j++)row.push(Math.min(row[j]+1,prev[j+1]+1,prev[j]+(a[i]!==b[j])));prev=row;}
  return prev[b.length];
}
function normalize(t){return String(t??'').normalize('NFKC').replace(/[\u2018\u2019\u00b4`]/g,"'").replace(/\u2026/g,'...').replace(/\s+/g,' ').trim();}
function contentNormalize(t){return normalize(t).toLocaleLowerCase('fr-FR').replace(/\s+([?!.,;:])/g,'$1');}
// Mirrors train_ppocrv6_rec.py:normalize_for_metrics, preserving case.
function trainingNormalize(t){return String(t??'').normalize('NFKC').replace(/[\u2018\u2019\u201a\u201b\u2032\u00b4`]/g,"'").replace(/[\u201c\u201d\u201e\u201f\u2033\u00ab\u00bb]/g,'"').replace(/[\u2010-\u2014]/g,'-').replace(/\u2026/g,'...').replace(/\u0153/g,'oe').replace(/\u0152/g,'OE').replace(/\s+/g,' ').trim().replace(/\s+([?!.,;:])/g,'$1').replace(/([([{])\s+/g,'$1').replace(/\s+([\])}])/g,'$1');}
function score(rows, key='text', reference='reference') {
  let chars=0,errors=0,exact=0,contentErrors=0,contentChars=0,contentExact=0,empty=0;
  for(const r of rows){const a=normalize(r[reference]),b=normalize(r[key]);const ac=contentNormalize(a),bc=contentNormalize(b);chars+=Array.from(a).length;errors+=distance(a,b);exact+=a===b;contentErrors+=distance(ac,bc);contentChars+=Array.from(ac).length;contentExact+=ac===bc;empty+=!b;}
  return {n:rows.length,chars,errors,cer:errors/Math.max(chars,1),exact:exact/Math.max(rows.length,1),exact_count:exact,empty,content_cer:contentErrors/Math.max(contentChars,1),content_exact:contentExact/Math.max(rows.length,1)};
}
function hashOrder(id){return require('node:crypto').createHash('sha256').update('ppocr-audit-20261103:'+id).digest('hex');}
function selectExamples(dataset,split,limit){return dataset.examples.filter(e=>e.split===split&&!e.invalid_bbox).sort((a,b)=>hashOrder(a.id).localeCompare(hashOrder(b.id))).slice(0,limit||Infinity);}
function iou(a,b){const inter=Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));return inter/Math.max(a.w*a.h+b.w*b.h-inter,1e-9);}
function matchBoxes(gt,pred){const edges=[];gt.forEach((g,i)=>pred.forEach((p,j)=>{const v=iou(g,p);if(v>=.5)edges.push({i,j,iou:v});}));edges.sort((a,b)=>b.iou-a.iou);const usedGt=new Set(),usedPred=new Set();return edges.filter(e=>{if(usedGt.has(e.i)||usedPred.has(e.j))return false;usedGt.add(e.i);usedPred.add(e.j);return true;});}
module.exports={ROOT,CACHE,manifest,ort,createCanvas,loadImage,trainingNormalize,ocrAppend,ocr:()=>contextFor('frontend/src/workers/ocr.worker.js',ocrAppend),detection:()=>contextFor('frontend/src/workers/detection.worker.js',detectionAppend),distance,normalize,contentNormalize,score,selectExamples,hashOrder,iou,matchBoxes};
