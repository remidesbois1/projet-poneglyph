/** Real isolated Chrome Canvas + ORT Web, without using a user's browser profile. */
const fs = require('node:fs'), path = require('node:path'), http = require('node:http'), crypto = require('node:crypto');
const {spawn} = require('node:child_process');
const ROOT = path.resolve(__dirname, '../..');
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
async function main() {
  const args = process.argv.slice(2), value = k => args[args.indexOf(k) + 1];
  for (const k of ['--models', '--dataset', '--output']) if (!args.includes(k)) throw new Error(`${k} required`);
  const models = path.resolve(value('--models')), dataset = path.resolve(value('--dataset')), output = path.resolve(value('--output'));
  const detectorMode = args.includes('--detector');
  const provider = args.includes('--provider') ? value('--provider') : 'wasm';
  if (!['wasm', 'webgpu'].includes(provider)) throw new Error('Unknown provider');
  if (detectorMode && provider !== 'wasm') throw new Error('The production bubble detector uses WASM');
  const threshold = args.includes('--confidence') ? Number(value('--confidence')) : .45;
  const limit = args.includes('--limit') ? Number(value('--limit')) : 20;
  const sourceEntries = detectorMode ? JSON.parse(fs.readFileSync(path.join(dataset, 'plan.json'))).pages.map(p => ({...p, bubble_id: p.id}))
    : fs.readFileSync(path.join(dataset, 'recognizer/line_manifest.jsonl'), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
  const entries = sourceEntries.filter(r => r.split === 'val').sort((a, b) => shaText(String(a.bubble_id)).localeCompare(shaText(String(b.bubble_id)))).slice(0, limit);
  fs.mkdirSync(output, {recursive: true});
  const manifest = JSON.parse(fs.readFileSync(path.join(models, 'browser_manifest.json')));
  const utils = fs.readFileSync(path.join(ROOT, 'frontend/src/lib/ocr-utils.js'), 'utf8').replace(/^export /gm, '');
  const modelFile = detectorMode ? 'bubble_detector.onnx' : 'ppocrv6_bubble_line_rec_webgpu.onnx';
  const workerPath = path.join(ROOT, 'frontend/src/workers', detectorMode ? 'detection.worker.js' : 'ocr.worker.js');
  const worker = fs.readFileSync(workerPath, 'utf8').replace(/^import .*;\r?\n/gm, '').replace(/const BUBBLE_SCORE_THRESHOLD =/g, 'let BUBBLE_SCORE_THRESHOLD =');
  const initialize = detectorMode
    ? `BUBBLE_SCORE_THRESHOLD=${threshold}; bubbleSession=await ort.InferenceSession.create('/models/${modelFile}',{executionProviders:['wasm'],graphOptimizationLevel:'all',logSeverityLevel:3});`
    : `ppocrManifest=${JSON.stringify(manifest)}; ppocrRecSession=await ort.InferenceSession.create('/models/${modelFile}',{executionProviders:[${JSON.stringify(provider)}],logSeverityLevel:3});`;
  const infer = detectorMode ? `const size=getImageInputSize(bubbleSession,800,800);const p=preprocessBubble(bitmap,size.height,size.width);
        const result=await bubbleSession.run({[bubbleSession.inputNames[0]]:p.inputTensor});
        window.checkState.rows.push({id:entry.id,boxes:simplifyPostProcess(result[bubbleSession.outputNames[0]].data,p.scale,p.padX,p.padY)});`
    : `const tensor=preprocessPpocrCanvas(bitmap);
        const result=await ppocrRecSession.run({[ppocrRecSession.inputNames[0]]:tensor});
        const raw=decodePpocrCtc(result[ppocrRecSession.outputNames[0]].data);
        window.checkState.rows.push({id:entry.id,raw,text:postprocessPpocrText(raw),width:tensor.dims[3]});`;
  const moduleSource = `import * as ort from '/ort/ort.${provider === 'webgpu' ? 'webgpu' : 'wasm'}.mjs';
    ort.env.wasm.wasmPaths='/ort/'; ort.env.wasm.numThreads=2;
    const modelRegistry=${fs.readFileSync(path.join(ROOT, 'packages/shared/src/model-registry.json'), 'utf8')};
    ${utils}\n${worker}
    ort.env.wasm.wasmPaths='/ort/'; ort.env.wasm.numThreads=2;
    window.checkState={status:'running',provider:${JSON.stringify(provider)},rows:[]};
    (async()=>{
      if(${JSON.stringify(provider)}==='webgpu') {
        const adapter=await navigator.gpu?.requestAdapter();
        if(!adapter){window.checkState.status='unavailable';window.checkState.reason='No WebGPU adapter';return;}
        window.checkState.adapter=adapter.info ? {vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description} : {};
      }
      ${initialize}
      for(const entry of ${JSON.stringify(entries.map((r, i) => ({id: r.bubble_id, index: i})))} ) {
        const bitmap=await createImageBitmap(await(await fetch('/input/'+entry.index+'.png')).blob());
        ${infer}
        bitmap.close();
      }
      window.checkState.status='complete';window.checkState.userAgent=navigator.userAgent;
    })().catch(e=>{window.checkState.status='error';window.checkState.error=String(e.stack||e)});`;
  let server, browser, ws;
  try {
    server = http.createServer((req, res) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin'); res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      if (req.url === '/') {res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><script type="module" src="/verify.mjs"></script>'); return;}
      if (req.url === '/verify.mjs') {res.setHeader('Content-Type', 'text/javascript'); res.end(moduleSource); return;}
      let file;
      const input = req.url.match(/^\/input\/(\d+)\.png$/);
      if (input && entries[Number(input[1])]) file = detectorMode ? path.join(dataset, 'pages', entries[Number(input[1])].id + '.png') : path.join(dataset, 'recognizer', entries[Number(input[1])].single_line_image);
      if (req.url === '/models/' + modelFile) file = path.join(models, modelFile);
      if (/^\/ort\/[a-zA-Z0-9_.-]+$/.test(req.url)) file = path.join(ROOT, 'node_modules/onnxruntime-web/dist', req.url.slice(5));
      if (!file || !fs.existsSync(file)) {res.statusCode = 404; res.end(); return;}
      res.setHeader('Content-Type', file.endsWith('.mjs') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm' : file.endsWith('.png') ? 'image/png' : 'application/octet-stream');
      fs.createReadStream(file).pipe(res);
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const origin = 'http://127.0.0.1:' + server.address().port;
    const chrome = process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
    browser = spawn(chrome, ['--headless=new', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', '--user-data-dir=' + path.join(output, 'chrome-profile'), 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true});
    const endpoint = await new Promise((resolve, reject) => {
      let log = ''; const timer = setTimeout(() => reject(new Error('Chrome startup timeout')), 30000);
      browser.once('error', reject); browser.stderr.on('data', chunk => {log += chunk; const match = log.match(/DevTools listening on (ws:\/\/[^\s]+)/); if (match) {clearTimeout(timer); resolve(match[1]);}});
    });
    const targets = await (await fetch('http://127.0.0.1:' + new URL(endpoint).port + '/json/list')).json();
    const target = targets.find(t => t.type === 'page');
    ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {ws.addEventListener('open', resolve); ws.addEventListener('error', reject);});
    let nextId = 0; const pending = new Map();
    ws.addEventListener('message', event => {const r = JSON.parse(String(event.data)); const p = pending.get(r.id); if (p) {pending.delete(r.id); r.error ? p.reject(new Error(JSON.stringify(r.error))) : p.resolve(r.result);}});
    const send = (method, params = {}) => new Promise((resolve, reject) => {const id = ++nextId; pending.set(id, {resolve, reject}); ws.send(JSON.stringify({id, method, params}));});
    await send('Page.navigate', {url: origin});
    let state, last = -1; const deadline = Date.now() + Math.max(240000, limit * 10000);
    while (Date.now() < deadline) {
      const r = await send('Runtime.evaluate', {expression: 'window.checkState', returnByValue: true}); state = r.result?.value;
      if (state?.rows?.length !== last) {last = state?.rows?.length ?? 0; console.log(JSON.stringify({provider, rows: last, total: entries.length, status: state?.status}));}
      if (['complete', 'error', 'unavailable'].includes(state?.status)) break;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    if (!['complete', 'unavailable'].includes(state?.status)) throw new Error(state?.error || 'Browser timeout');
    state.protocol = {model_sha256: sha(path.join(models, modelFile)), manifest_sha256: sha(path.join(models, 'browser_manifest.json')),
      worker_sha256: sha(workerPath), confidence: detectorMode ? threshold : null,
      input_mode: detectorMode ? 'Full pages; actual production detector preprocessing' : 'Production-prepared PNGs; real browser Canvas normalization', ids: entries.map(r => r.bubble_id)};
    fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(state, null, 2));
    console.log(JSON.stringify({provider, status: state.status, samples: state.rows.length, adapter: state.adapter, reason: state.reason}));
    await send('Browser.close').catch(() => {});
  } finally {ws?.close(); browser?.kill(); server?.closeAllConnections(); server?.close();}
}
function shaText(x) {return crypto.createHash('sha256').update('ppocr-browser-20261103:' + x).digest('hex');}
main().catch(e => {console.error(e); process.exitCode = 1;});
