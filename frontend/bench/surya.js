const worker = new Worker(new URL('../src/workers/surya.worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
let onProgress;
const engine = {
    async load(progress, options) {
        this.options = options; onProgress = progress;
        await new Promise((resolve, reject) => { pending.set('init', { resolve, reject }); worker.postMessage({ type: 'init', options }); });
    },
    recognize(imageBlob) {
        const requestId = crypto.randomUUID();
        return new Promise((resolve, reject) => { pending.set(requestId, { resolve, reject }); worker.postMessage({ type: 'run', requestId, imageBlob, options: this.options }); });
    },
};
worker.onmessage = ({ data }) => {
    if (data.status === 'download_progress') onProgress?.(data);
    if (data.status === 'ready') { engine.gpuName = data.gpu; pending.get('init')?.resolve(); pending.delete('init'); }
    if (data.status === 'complete') { pending.get(data.requestId)?.resolve(data); pending.delete(data.requestId); }
    if (data.status === 'error') { pending.get(data.requestId || 'init')?.reject(new Error(data.detail || data.error)); pending.delete(data.requestId || 'init'); }
};
worker.onerror = event => { for (const request of pending.values()) request.reject(new Error(event.message)); pending.clear(); };
const $ = id => document.getElementById(id);
const examples = await (await fetch('/models/surya-ocr/fixtures/examples.json')).json();
const images = await Promise.all(examples.map(async item => ({ ...item, blob: await (await fetch(`/models/surya-ocr/${item.image}`)).blob() })));
$('load').onclick = async () => {
    $('load').disabled = true;
    try {
        await engine.load(p => { $('status').textContent = `${p.file} · ${Math.round(p.progress)} %`; }, {
            profile: $('profile').checked, mode: $('mode').value, gemv: $('gemv').value, tokenBatch: Number($('batch').value),
            prefill: $('prefill').value,
        });
        $('status').textContent = `Prêt · ${engine.gpuName}`; $('run').disabled = false;
        $('sweep').disabled = engine.options.mode !== 'fused';
        for (const id of ['mode', 'profile', 'prefill']) $(id).disabled = true;
    } catch (e) { $('error').textContent = e.stack; $('load').disabled = false; }
};
async function measure(sweep = false) {
    $('run').disabled = true; $('sweep').disabled = true; $('error').textContent = ''; $('rows').replaceChildren();
    engine.options.tokenBatch = Number($('batch').value); engine.options.gemv = $('gemv').value;
    const results = [];
    let items = $('corpus').value === 'all' ? images : [images[$('corpus').value === 'short' ? 0 : 23]];
    if (['test128', 'full1612'].includes($('corpus').value)) {
        const corpus = $('corpus').value;
        items = await (await fetch(`/models/surya-ocr/${corpus}/examples.json`)).json();
        // Read just before inference: full test sets must not trigger thousands of parallel requests.
        items = items.map(item => ({ ...item, url: `/models/surya-ocr/${corpus}/${item.image.split('/').pop()}` }));
    }
    const ids = $('ids').value.trim().split(/[\s,]+/).filter(Boolean);
    if (ids.length) items = items.filter(item => ids.includes(item.id));
    const settings = sweep ? ['shared', 'subgroup'].flatMap(gemv => [1, 2, 4, 8].map(tokenBatch => ({ gemv, tokenBatch }))) : [{}];
    if (sweep) items = [images[23]];
    try {
        for (const setting of settings) {
        Object.assign(engine.options, setting);
        $('status').textContent = 'Échauffement (hors mesures)…';
        if (!items.length) throw new Error('Aucune bulle ne correspond aux IDs demandés.');
        const getBlob = async item => item.blob || (await fetch(item.url)).blob();
        await engine.recognize(await getBlob(items[0]));
        for (let repeat = 0; repeat < (sweep ? 3 : Number($('repeats').value)); repeat++) for (const item of items) {
            $('status').textContent = `${engine.options.gemv} × ${engine.options.tokenBatch} · passe ${repeat + 1} · bulle ${item.id}`;
            const prediction = await engine.recognize(await getBlob(item));
            const row = { id: item.id, repeat, options: { ...engine.options }, mode: engine.options.mode, groundTruth: item.reference,
                reference: item.native || item.reference, ...prediction };
            results.push(row); $('results').textContent = JSON.stringify(results, null, 2);
            const tr = document.createElement('tr');
            for (const text of [item.id, `${prediction.timings.totalMs} ms`, `${prediction.timings.prefillMs} ms`,
                (prediction.tokens / ((prediction.timings.totalMs - prediction.timings.prefillMs) / 1000)).toFixed(1),
                row.text === row.reference ? 'Conforme' : `À examiner : ${row.text}`]) {
                const td = document.createElement('td'); td.textContent = text; tr.append(td);
            }
            $('rows').append(tr);
        }
        }
        $('status').textContent = `Terminé · ${results.filter(r => r.text === r.reference).length}/${results.length} conformes`;
    } catch (e) { $('error').textContent = e.stack; $('status').textContent = 'Erreur'; }
    finally { $('run').disabled = false; $('sweep').disabled = engine.options.mode !== 'fused'; }
}
$('run').onclick = () => measure();
$('sweep').onclick = () => measure(true);
