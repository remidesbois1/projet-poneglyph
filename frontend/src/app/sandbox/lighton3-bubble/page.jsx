'use client';
import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { LIGHTON3_BUBBLE_MODEL_BASE } from '@/lib/lighton3BubbleModel';

const MODEL_PATH = '/models/lighton3-bubble';

export default function Lighton3BubbleSandbox() {
    const worker = useRef(null), pending = useRef(new Map()), active = useRef(null), cancelled = useRef(false);
    const [status, setStatus] = useState('idle'), [progress, setProgress] = useState({ progress: 0, file: '' });
    const [gpu, setGpu] = useState(''), [selected, setSelected] = useState(null), [examples, setExamples] = useState([]);
    const [result, setResult] = useState(null), [results, setResults] = useState([]), [stream, setStream] = useState('');
    const [error, setError] = useState(''), [batch, setBatch] = useState(''), [metrics, setMetrics] = useState(null);
    useEffect(() => () => { if (selected?.blob) URL.revokeObjectURL(selected.url); }, [selected]);
    useEffect(() => {
        const jobs = pending.current;
        const w = new Worker(new URL('../../../workers/lighton3Bubble.worker.js', import.meta.url), { type: 'module' });
        worker.current = w;
        w.onmessage = ({ data }) => {
            if (data.status === 'download_progress') setProgress(data);
            if (data.status === 'ready') { setStatus('ready'); setGpu(data.gpu); }
            if (data.status === 'stream') setStream(data.text);
            if (data.status === 'disposed') setStatus('idle');
            if (data.status === 'complete') { jobs.get(data.requestId)?.resolve(data); jobs.delete(data.requestId); }
            if (data.status === 'error') {
                if (!data.cancelled) setError(data.detail || data.error);
                if (!data.requestId) setStatus('error');
                jobs.get(data.requestId)?.reject(new Error(data.error)); jobs.delete(data.requestId);
            }
        };
        w.onerror = event => {
            setError(event.message); setStatus('error');
            jobs.forEach(job => job.reject(new Error(event.message))); jobs.clear();
        };
        const metricsBase = process.env.NODE_ENV === 'development' && !process.env.NEXT_PUBLIC_LIGHTON3_BUBBLE_MODEL_BASE
            ? MODEL_PATH : LIGHTON3_BUBBLE_MODEL_BASE;
        fetch(`${metricsBase}/webgpu_metrics.json`).then(r => r.ok ? r.json() : null).then(setMetrics).catch(() => {});
        if (process.env.NODE_ENV === 'development') {
            fetch(`${MODEL_PATH}/fixtures/examples.json`).then(r => r.ok ? r.json() : []).then(items => {
                setExamples(items); if (items[0]) setSelected({ ...items[0], url: `${MODEL_PATH}/${items[0].image}` });
            }).catch(() => {});
        }
        return () => { w.terminate(); jobs.forEach(job => job.reject(new Error('Page fermée.'))); jobs.clear(); };
    }, []);
    const run = async (items = [selected]) => {
        cancelled.current = false;
        setStatus('running'); setError(''); setResults([]); setResult(null); setStream('');
        const completed = [];
        try {
            for (const [index, item] of items.entries()) {
                if (cancelled.current) break;
                setSelected(item); setResult(null); setStream(''); setBatch(`${index + 1} / ${items.length}`);
                const requestId = crypto.randomUUID(); active.current = requestId;
                const blob = item.blob || await (await fetch(item.url)).blob();
                if (cancelled.current) break;
                const prediction = await new Promise((resolve, reject) => {
                    pending.current.set(requestId, { resolve, reject });
                    worker.current.postMessage({ type: 'run', imageBlob: blob, requestId });
                });
                const record = { ...prediction, id: item.id, page_id: item.page_id, reference: item.reference,
                    native: item.native, control: item.control, split: item.split, matchesNative: prediction.text === item.native };
                completed.push(record); setResult(prediction); setResults([...completed]);
            }
        } catch (e) { if (!cancelled.current) setError(e.message); }
        finally { active.current = null; setStatus('ready'); }
    };
    const runBenchmark = async () => {
        const response = await fetch(`${MODEL_PATH}/fixtures/benchmark.json`);
        if (!response.ok) { setError('Le jeu de contrôle local est absent.'); return; }
        await run((await response.json()).map(item => ({ ...item, url: `${MODEL_PATH}/${item.image}` })));
    };
    return <main className="mx-auto max-w-5xl px-6 py-10 space-y-6">
        <Link href="/sandbox" className="text-sm text-sky-400">← Sandbox Poneglyph</Link>
        <header><p className="text-sm uppercase tracking-widest text-sky-400">Transcription sur votre GPU</p>
            <h1 className="text-4xl font-semibold mt-3">LightOnOCR 3 · bulles</h1>
            <p className="mt-3 text-slate-400">Le modèle Poneglyph 0.8B lit une bulle entière. Les images restent sur votre appareil.</p>
            <a className="block mt-3 text-sm text-sky-400" href="https://huggingface.co/Remidesbois/LightOnOCR-3-Poneglyph" target="_blank" rel="noreferrer">Modèle et mesures de précision ↗</a></header>
        {metrics?.verified && <section className="grid grid-cols-3 gap-4" aria-label="Précision WebGPU sur le test figé">
            {['cer', 'wer', 'exact_match'].map(key => <div key={key} className="rounded-xl border border-slate-700 p-4">
                <p className="text-sm text-slate-400">{key === 'exact_match' ? 'Bulles exactes' : key.toUpperCase()} · WebGPU</p>
                <p className="text-2xl mt-2">{(100 * metrics.metrics[key]).toFixed(key === 'exact_match' ? 2 : 3)} %</p>
            </div>)}
        </section>}
        <section className="rounded-xl border border-slate-700 p-5 space-y-3">
            <div className="flex flex-wrap gap-4 items-center">
                <button className="rounded-lg bg-sky-600 px-5 py-3 disabled:opacity-40" disabled={!['idle', 'error'].includes(status)}
                    onClick={() => { setStatus('loading'); setError(''); worker.current.postMessage({ type: 'init' }); }}>Charger LightOnOCR 3 sur WebGPU</button>
                <span role="status">{status === 'loading' ? `${progress.file} · ${Math.round(progress.progress)} %` :
                    status === 'running' ? `Lecture en cours · ${batch}` : status === 'ready' ? `Prêt · ${gpu}` : 'Modèle non chargé · environ 1,74 Go'}</span>
                {status === 'ready' && <button className="text-sm text-slate-400" onClick={() => worker.current.postMessage({ type: 'dispose' })}>Libérer le modèle</button>}
            </div>
            {status === 'loading' && <progress aria-label="Téléchargement du modèle" className="w-full" max="100" value={progress.progress} />}
            {error && <pre role="alert" className="text-red-400 whitespace-pre-wrap break-all text-sm">{error}</pre>}
        </section>
        <div className="grid gap-6 lg:grid-cols-2">
            <section className="rounded-xl border border-slate-700 p-5 space-y-4">
                <label className="block text-sm">Choisir une bulle <input className="block mt-2" type="file" accept="image/*" disabled={status === 'running'}
                    onChange={event => { const blob = event.target.files?.[0]; if (blob) { setResult(null); setSelected({ blob, url: URL.createObjectURL(blob) }); } }} /></label>
                <div className="flex flex-wrap gap-3"><button className="rounded-lg bg-sky-600 px-4 py-2 disabled:opacity-40" disabled={status !== 'ready' || !selected} onClick={() => run()}>Lire la bulle</button>
                    {status === 'running' && <button className="rounded-lg border border-slate-600 px-4 py-2" onClick={() => {
                        cancelled.current = true; worker.current.postMessage({ type: 'cancel', requestId: active.current });
                    }}>Annuler la lecture</button>}</div>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {selected && <img className="block max-h-96 max-w-full mx-auto" src={selected.url} alt="Bulle à transcrire" />}
            </section>
            <section className="rounded-xl border border-slate-700 p-5 space-y-4">
                <h2 className="text-xl font-medium">Transcription</h2>
                {result && <p className="text-sm text-sky-300">{result.tokens} tokens · {(result.timings.totalMs / 1000).toFixed(2)} s · WebGPU</p>}
                {result?.tokenLimit && <p role="alert" className="text-amber-300">La limite de génération est atteinte. Vérifiez la fin du texte.</p>}
                <p className="text-lg whitespace-pre-wrap">{result?.text || stream || 'La transcription apparaîtra ici.'}</p>
            </section>
        </div>
        {process.env.NODE_ENV === 'development' && <section className="rounded-xl border border-slate-700 p-5 space-y-4">
            <h2 className="text-xl font-medium">Contrôles locaux de conversion</h2>
            <p className="text-sm text-slate-400">Références natives BF16, guillemets et casse conservés. Images et transcriptions restent locales.</p>
            <div className="flex flex-wrap gap-3"><button disabled={status !== 'ready' || !examples.length} className="rounded-lg border border-slate-600 px-4 py-2 disabled:opacity-40"
                onClick={() => run(examples.map(item => ({ ...item, url: `${MODEL_PATH}/${item.image}` })))}>Vérifier les {examples.length} bulles fixes</button>
                <button disabled={status !== 'ready'} className="rounded-lg border border-slate-600 px-4 py-2 disabled:opacity-40" onClick={runBenchmark}>Évaluer les 1 612 bulles de test</button></div>
            {results.length > 0 && <><p>{results.length} bulles terminées · {results.filter(item => item.matchesNative).length} sorties identiques au BF16</p>
                <details><summary className="text-sm text-slate-400 cursor-pointer">Résultats JSON</summary><pre id="lighton3-bubble-validation-json" className="max-h-80 overflow-auto text-xs">{JSON.stringify(results)}</pre></details></>}
        </section>}
    </main>;
}
