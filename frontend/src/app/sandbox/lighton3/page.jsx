'use client';
import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';

export default function Lighton3Sandbox() {
    const worker = useRef(null), pending = useRef(new Map()), activeId = useRef(null);
    const [status, setStatus] = useState('idle'), [progress, setProgress] = useState({ file: '', progress: 0 });
    const [gpu, setGpu] = useState(''), [selected, setSelected] = useState(null), [examples, setExamples] = useState([]);
    const [result, setResult] = useState(null), [results, setResults] = useState([]), [stream, setStream] = useState('');
    const [error, setError] = useState(''), [batchProgress, setBatchProgress] = useState('');
    const cancelBatch = useRef(false);
    useEffect(() => {
        const url = selected?.blob ? selected.url : null;
        return () => { if (url) URL.revokeObjectURL(url); };
    }, [selected]);
    useEffect(() => {
        const requests = pending.current;
        const w = new Worker(new URL('../../../workers/lighton3.worker.js', import.meta.url), { type: 'module' });
        worker.current = w;
        w.onmessage = ({ data }) => {
            if (data.status === 'download_progress') setProgress(data);
            if (data.status === 'ready') { setStatus('ready'); setGpu(data.gpu); }
            if (data.status === 'stream') setStream(data.text);
            if (data.status === 'disposed') setStatus('idle');
            if (data.status === 'complete') { pending.current.get(data.requestId)?.resolve(data); pending.current.delete(data.requestId); }
            if (data.status === 'error') {
                if (!data.cancelled) setError(data.detail || data.error);
                if (!data.requestId) setStatus('error');
                pending.current.get(data.requestId)?.reject(new Error(data.error)); pending.current.delete(data.requestId);
            }
        };
        w.onerror = event => {
            setError(event.message); setStatus('error');
            pending.current.forEach(job => job.reject(new Error(event.message))); pending.current.clear();
        };
        if (process.env.NODE_ENV === 'development') {
            fetch('/models/lighton3-ocr/fixtures/examples.json').then(r => r.ok ? r.json() : []).then(items => {
                setExamples(items); if (items[0]) setSelected({ ...items[0], url: `/models/lighton3-ocr/${items[0].image}` });
            }).catch(() => {});
        }
        return () => { w.terminate(); requests.forEach(job => job.reject(new Error('Page fermée.'))); requests.clear(); };
    }, []);
    const recognize = async item => {
        const requestId = crypto.randomUUID();
        activeId.current = requestId;
        const blob = item.blob || await (await fetch(item.url)).blob();
        if (cancelBatch.current) throw new Error('Lecture annulée.');
        return new Promise((resolve, reject) => {
            pending.current.set(requestId, { resolve, reject });
            worker.current.postMessage({ type: 'run', imageBlob: blob, requestId });
        });
    };
    const run = async (items = [selected]) => {
        cancelBatch.current = false;
        setStatus('running'); setError(''); setResult(null); setResults([]); setStream('');
        const completed = [];
        try {
            for (const [index, item] of items.entries()) {
                if (cancelBatch.current) break;
                setSelected(item); setResult(null); setStream(''); setBatchProgress(`${index + 1} / ${items.length}`);
                const prediction = await recognize(item);
                setResult(prediction);
                completed.push({ ...prediction, id: item.id, reference: item.reference, native: item.native,
                    control: item.control, split: item.split, matchesNative: prediction.text === item.native });
                setResults([...completed]);
            }
        } catch (e) { if (!cancelBatch.current) setError(e.message); }
        finally { activeId.current = null; setStatus('ready'); }
    };
    const runBenchmark = async () => {
        const response = await fetch('/models/lighton3-ocr/fixtures/benchmark.json');
        if (!response.ok) { setError('Le jeu de contrôle local est absent.'); return; }
        const items = await response.json();
        await run(items.map(item => ({ ...item, url: `/models/lighton3-ocr/${item.image}` })));
    };
    return <main className="mx-auto max-w-6xl px-6 py-10 space-y-6">
        <Link href="/sandbox" className="text-sm text-sky-400">← Sandbox Poneglyph</Link>
        <header><p className="text-sm uppercase tracking-widest text-sky-400">Lecture de pages sur votre GPU</p>
            <h1 className="text-4xl font-semibold mt-3">LightOnOCR 3 · texte et bbox</h1>
            <p className="mt-3 text-slate-400">Le modèle 0.8B Poneglyph lit la page entière. Les images restent sur votre appareil.</p></header>
        <section className="rounded-xl border border-slate-700 p-5 space-y-3">
            <div className="flex flex-wrap gap-4 items-center">
                <button className="rounded-lg bg-sky-600 px-5 py-3 disabled:opacity-40" disabled={!['idle', 'error'].includes(status)}
                    onClick={() => { setStatus('loading'); setError(''); worker.current.postMessage({ type: 'init' }); }}>Charger LightOnOCR 3 sur WebGPU</button>
                <span role="status">{status === 'loading' ? `${progress.file} · ${Math.round(progress.progress)} %` :
                    status === 'running' ? `Lecture en cours · ${batchProgress}` : status === 'ready' ? `Prêt · ${gpu}` : 'Modèle non chargé · environ 1,72 Go'}</span>
                {status === 'ready' && <button className="text-sm text-slate-400" onClick={() => worker.current.postMessage({ type: 'dispose' })}>Libérer le modèle</button>}
            </div>
            {status === 'loading' && <progress aria-label="Téléchargement du modèle" className="w-full" max="100" value={progress.progress} />}
            {error && <pre role="alert" className="text-red-400 whitespace-pre-wrap break-all text-sm">{error}</pre>}
        </section>
        <div className="grid gap-6 lg:grid-cols-2">
            <section className="rounded-xl border border-slate-700 p-5 space-y-4">
                <label className="block text-sm">Choisir une page <input className="block mt-2" type="file" accept="image/*" disabled={status === 'running'}
                    onChange={event => { const blob = event.target.files?.[0]; if (blob) { setResult(null); setSelected({ blob, url: URL.createObjectURL(blob) }); } }} /></label>
                <div className="flex flex-wrap gap-3">
                    <button className="rounded-lg bg-sky-600 px-4 py-2 disabled:opacity-40" disabled={status !== 'ready' || !selected} onClick={() => run()}>Lire la page entière</button>
                    {status === 'running' && <button className="rounded-lg border border-slate-600 px-4 py-2" onClick={() => {
                        cancelBatch.current = true; worker.current.postMessage({ type: 'cancel', requestId: activeId.current });
                    }}>Annuler la lecture</button>}
                </div>
                {selected && <div className="relative">
                    {/* A local blob is displayed at its native aspect ratio for bbox overlays. */}
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img className="block w-full" src={selected.url} alt="Page à transcrire" />
                    <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" className="absolute inset-0 h-full w-full pointer-events-none" aria-label="Zones de texte détectées">
                        {result?.bubbles.map((bubble, i) => <g key={i}><rect x={bubble.bbox[0]} y={bubble.bbox[1]} width={bubble.bbox[2] - bubble.bbox[0]} height={bubble.bbox[3] - bubble.bbox[1]}
                            fill="rgba(14,165,233,.08)" stroke="#0ea5e9" strokeWidth="2" /><text x={bubble.bbox[0] + 3} y={bubble.bbox[1] + 15} fill="#0ea5e9" fontSize="14">{i + 1}</text></g>)}
                    </svg>
                </div>}
            </section>
            <section className="rounded-xl border border-slate-700 p-5 space-y-4">
                <h2 className="text-xl font-medium">Texte et localisation</h2>
                {result && <p className="text-sm text-sky-300">{result.bubbles.length} zones · {result.tokens} tokens · {(result.timings.totalMs / 1000).toFixed(2)} s · WebGPU</p>}
                {result?.tokenLimit && <p role="alert" className="text-amber-300">La limite de génération est atteinte. Vérifiez la fin de la page.</p>}
                {result?.invalidBlocks > 0 && <p role="alert" className="text-amber-300">{result.invalidBlocks} zone(s) invalide(s) ignorée(s).</p>}
                <ol className="space-y-3">{result?.bubbles.map((bubble, i) => <li key={i} className="border-b border-slate-800 pb-3">
                    <p><span className="text-sky-400 mr-2">{i + 1}.</span>{bubble.content}</p><p className="text-xs text-slate-500 mt-1">{bubble.bbox.join(', ')} · coordonnées sur 1000</p>
                </li>)}</ol>
                <details open={status === 'running'}><summary className="text-sm text-slate-400 cursor-pointer">Sortie grounding</summary>
                    <pre className="mt-3 whitespace-pre-wrap text-sm max-h-96 overflow-auto">{result?.text || stream || 'La transcription apparaîtra ici.'}</pre></details>
            </section>
        </div>
        {process.env.NODE_ENV === 'development' && <section className="rounded-xl border border-slate-700 p-5 space-y-4">
            <h2 className="text-xl font-medium">Contrôles locaux de conversion</h2>
            <p className="text-sm text-slate-400">Pages fixes avec références BF16. Les exemples et prédictions restent locaux.</p>
            <div className="flex gap-3"><button disabled={status !== 'ready' || !examples.length} className="rounded-lg border border-slate-600 px-4 py-2 disabled:opacity-40"
                onClick={() => run(examples.map(item => ({ ...item, url: `/models/lighton3-ocr/${item.image}` })))}>Vérifier les {examples.length} pages fixes</button>
                <button disabled={status !== 'ready'} className="rounded-lg border border-slate-600 px-4 py-2 disabled:opacity-40" onClick={runBenchmark}>Évaluer le test local complet</button></div>
            {results.length > 0 && <><p>{results.length} pages terminées · {results.filter(item => item.matchesNative).length} sorties identiques au BF16</p>
                <details><summary className="text-sm text-slate-400 cursor-pointer">Résultats JSON</summary><pre id="lighton3-validation-json" className="max-h-80 overflow-auto text-xs">{JSON.stringify(results)}</pre></details></>}
        </section>}
    </main>;
}
