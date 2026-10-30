const worker = new Worker(new URL('../src/workers/falcon.worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
const $ = id => document.getElementById(id);
worker.onmessage = ({ data }) => {
    if (data.status === 'download_progress') $('status').textContent = `${data.file} · ${Math.round(data.progress)} %`;
    if (data.status === 'ready') { pending.get('init')?.resolve(data); pending.delete('init'); }
    if (data.status === 'complete') { pending.get(data.requestId)?.resolve(data); pending.delete(data.requestId); }
    if (data.status === 'error') { const key = data.requestId || 'init'; pending.get(key)?.reject(new Error(data.detail || data.error)); pending.delete(key); }
};
worker.onerror = event => { for (const request of pending.values()) request.reject(new Error(event.message)); pending.clear(); };
function recognize(blob) {
    const requestId = crypto.randomUUID();
    return new Promise((resolve, reject) => { pending.set(requestId, { resolve, reject }); worker.postMessage({ type: 'run', imageBlob: blob, requestId }); });
}
$('load').onclick = async () => {
    $('load').disabled = true;
    try {
        const info = await new Promise((resolve, reject) => {
            pending.set('init', { resolve, reject }); worker.postMessage({ type: 'init', options: { mode: $('mode').value, tokenBatch: Number($('batch').value) } });
        });
        $('status').textContent = `Prêt · ${info.gpu}`; $('run').disabled = false; $('mode').disabled = true;
    } catch (error) { $('error').textContent = error.stack || String(error); $('load').disabled = false; }
};
$('run').onclick = async () => {
    $('run').disabled = true; $('error').textContent = '';
    try {
        const corpus = $('corpus').value;
        const items = (await (await fetch(`/__falcon_fixtures/${corpus}/examples.json`)).json()).slice(0, Number($('limit').value));
        if (!items.length) throw new Error('Aucune bulle sélectionnée.');
        const image = async item => {
            const response = await fetch(`/__falcon_fixtures/${corpus}/${item.image.split('/').pop()}`);
            if (!response.ok) throw new Error(`Image absente : ${item.id}`);
            return response.blob();
        };
        $('status').textContent = 'Échauffement…';
        await recognize(await image(items[0]));
        const rows = [];
        for (const item of items) {
            $('status').textContent = `Mesure ${rows.length + 1}/${items.length}`;
            const result = await recognize(await image(item));
            rows.push({ id: item.id, reference: item.reference, ...result });
        }
        const result = await fetch(`/__falcon_results?name=${encodeURIComponent($('name').value)}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ corpus, rows }),
        });
        if (!result.ok) throw new Error(await result.text());
        $('status').textContent = `Terminé · ${rows.length} bulles · résultats enregistrés`;
    } catch (error) { $('error').textContent = error.stack || String(error); $('status').textContent = 'Erreur'; }
    finally { $('run').disabled = false; }
};
