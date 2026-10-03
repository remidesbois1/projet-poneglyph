import Link from "next/link";
import { ArrowRight, Check, Cpu, Database, ScanText, Search, ShieldCheck } from "lucide-react";

const STEPS = [
    {
        number: "01",
        title: "Analyser la page",
        description: "Les modèles repèrent les cases et les bulles, retrouvent leur ordre de lecture et transcrivent le texte.",
        detail: "Détection & reconnaissance de texte",
        icon: ScanText,
    },
    {
        number: "02",
        title: "Valider le contenu",
        description: "Les contributeurs corrigent les annotations et vérifient les transcriptions avant leur publication dans l’index.",
        detail: "Correction & validation humaine",
        icon: ShieldCheck,
    },
    {
        number: "03",
        title: "Retrouver une scène",
        description: "Les pages validées sont indexées. Une réplique ou une description suffit pour lancer une recherche dans le manga.",
        detail: "Recherche textuelle & sémantique",
        icon: Search,
    },
];

function StepIllustration({ step }) {
    if (step === "01") {
        return (
            <div className="relative flex h-full items-center justify-center gap-3 lg:gap-5" aria-hidden="true">
                <div className="grid h-24 w-16 shrink-0 -rotate-6 grid-cols-2 gap-1.5 rounded-md border border-slate-500/40 bg-[#0b1b2c] p-2 shadow-lg lg:h-28 lg:w-20">
                    <div className="relative col-span-2 rounded-sm border border-[#8dbbff]/60 bg-[#3d86ff]/10">
                        <div className="absolute right-1.5 top-1.5 h-3 w-7 rounded-full border border-[#8dbbff]/60" />
                    </div>
                    <div className="rounded-sm border border-[#8dbbff]/50 bg-[#3d86ff]/10" />
                    <div className="relative rounded-sm border border-[#8dbbff]/50 bg-[#3d86ff]/10">
                        <div className="absolute left-1 top-2 h-4 w-4 rounded-full border border-[#8dbbff]/60" />
                    </div>
                </div>
                <ArrowRight size={16} className="shrink-0 text-slate-500" />
                <div className="min-w-0 max-w-28 flex-1 space-y-3">
                    <div className="text-xs text-[#b7d3ff]">Texte extrait</div>
                    <div className="h-1.5 w-full rounded-full bg-[#8dbbff]/35" />
                    <div className="h-1.5 w-4/5 rounded-full bg-[#8dbbff]/20" />
                    <div className="h-1.5 w-3/5 rounded-full bg-[#8dbbff]/20" />
                </div>
            </div>
        );
    }

    if (step === "02") {
        return (
            <div className="mx-auto flex h-full max-w-60 flex-col justify-center gap-2.5" aria-hidden="true">
                {["Ordre de lecture", "Transcription", "Annotations"].map((label) => (
                    <div key={label} className="flex items-center justify-between gap-5 rounded-md border border-white/8 bg-[#0b1b2c] px-3 py-2">
                        <span className="text-xs text-slate-300">{label}</span>
                        <Check size={14} className="text-emerald-300" />
                    </div>
                ))}
            </div>
        );
    }

    return (
        <div className="mx-auto flex h-full max-w-60 flex-col justify-center gap-3" aria-hidden="true">
            <div className="flex items-center gap-2 rounded-md border border-[#8dbbff]/25 bg-[#0b1b2c] px-3 py-2.5">
                <Search size={14} className="shrink-0 text-[#8dbbff]" />
                <span className="text-xs text-slate-300">Une scène, une réplique…</span>
            </div>
            <div className="flex items-center gap-3 px-3">
                <div className="flex h-12 w-9 shrink-0 items-center justify-center rounded border border-[#8dbbff]/30 bg-[#3d86ff]/10"><ScanText size={18} className="text-[#8dbbff]" /></div>
                <div className="space-y-2">
                    <span className="text-xs text-[#b7d3ff]">Pages correspondantes</span>
                    <div className="h-1.5 w-24 rounded-full bg-[#8dbbff]/20" />
                </div>
            </div>
        </div>
    );
}

export default function LandingArchitecture() {
    return (
        <section id="features" aria-labelledby="architecture-title" className="relative scroll-mt-32 border-b border-white/8 py-16 md:scroll-mt-20 sm:py-24">
            <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_50%_50%,rgba(61,134,255,0.06),transparent_70%)]" />
            <div className="relative mx-auto max-w-6xl px-5 sm:px-8">
                <div className="flex flex-col justify-between gap-6 md:flex-row md:items-end md:gap-12">
                    <div className="max-w-xl">
                        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#8dbbff]">Architecture & technologies</p>
                        <h2 id="architecture-title" className="mt-4 text-3xl font-semibold leading-tight tracking-tight text-white sm:text-4xl lg:text-[2.75rem]">De la page à la recherche.</h2>
                        <p className="mt-4 max-w-lg text-sm leading-7 text-slate-300 sm:text-base">L’IA déchiffre. La communauté vérifie.<br className="hidden sm:block" /> Vous retrouvez le passage que vous cherchez.</p>
                    </div>
                    <Link href="/sandbox" className="inline-flex min-h-11 w-fit shrink-0 items-center gap-3 rounded-lg border border-[#8dbbff]/30 bg-[#3d86ff]/10 px-5 py-3 text-sm font-semibold text-[#bdd6ff] transition hover:border-[#8dbbff]/60 hover:bg-[#3d86ff]/20">
                        Essayer la Sandbox <ArrowRight size={16} aria-hidden="true" />
                    </Link>
                </div>

                <ol className="mt-10 grid overflow-hidden rounded-2xl border border-white/12 bg-[#071321]/85 md:grid-cols-3">
                    {STEPS.map(({ number, title, description, detail, icon: Icon }) => (
                        <li key={number} className="relative flex flex-col border-white/10 p-6 not-first:border-t sm:p-7 md:not-first:border-l md:not-first:border-t-0">
                            <div className="flex items-center justify-between">
                                <span className="font-mono text-xs tracking-[0.18em] text-slate-400">{number}</span>
                                <Icon size={20} className="text-[#8dbbff]" aria-hidden="true" />
                            </div>
                            <div className="my-6 h-36"><StepIllustration step={number} /></div>
                            <h3 className="text-xl font-semibold tracking-tight text-white">{title}</h3>
                            <p className="mb-6 mt-3 text-sm leading-6 text-slate-300">{description}</p>
                            <p className="mt-auto border-t border-white/8 pt-4 text-xs leading-5 text-slate-400">{detail}</p>
                        </li>
                    ))}
                </ol>

                <div className="mt-6 grid gap-5 rounded-xl border border-white/8 bg-[#071321]/45 p-5 sm:p-6 lg:grid-cols-[1.1fr_1fr] lg:gap-10">
                    <div className="flex gap-3">
                        <Cpu size={18} className="mt-0.5 shrink-0 text-[#8dbbff]" aria-hidden="true" />
                        <div>
                            <h3 className="text-sm font-medium text-slate-200">Une architecture hybride</h3>
                            <p className="mt-2 text-xs leading-6 text-slate-400">Détection et OCR dans le navigateur. L’application de bureau et les services cloud prennent le relais pour les modèles plus lourds.</p>
                        </div>
                    </div>
                    <ul aria-label="Technologies utilisées" className="flex flex-wrap content-center items-center gap-2 lg:justify-end">
                        {["WebGPU / ONNX", "Transformers.js", "Tauri", "GPU local / cloud", "PostgreSQL / pgvector"].map((technology) => (
                            <li key={technology} className="rounded-md border border-white/10 px-2.5 py-1.5 font-mono text-[11px] text-slate-300">{technology}</li>
                        ))}
                    </ul>
                </div>
                <p className="mt-4 flex items-start gap-2 text-xs leading-5 text-slate-400">
                    <Database size={14} className="mt-0.5 shrink-0" aria-hidden="true" />
                    Seuls les contenus validés alimentent la recherche publique.
                </p>
            </div>
        </section>
    );
}
