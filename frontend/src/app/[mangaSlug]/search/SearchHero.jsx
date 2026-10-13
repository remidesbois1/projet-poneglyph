"use client";

import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { SEARCH_MODES } from './searchModes';

const PONEGLYPH_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

function generateSlots(count, seed = 0) {
    const slots = [];
    for (let i = 0; i < count; i++) {
        const s = (seed + i) * 2654435761;
        const hash = (v) => ((s * (v + 1)) >>> 0) / 4294967296;
        slots.push({
            x: 2 + hash(1) * 96,
            y: 4 + hash(2) * 88,
            size: 20 + Math.floor(hash(3) * 30),
            rotate: Math.floor(hash(4) * 80) - 40,
            char: PONEGLYPH_LETTERS[Math.floor(hash(5) * 26)],
            opacity: 0,
        });
    }
    return slots;
}

function PoneglyphGlyphs({ count = 16 }) {
    const [glyphs, setGlyphs] = useState(() => generateSlots(count, 29));

    useEffect(() => {
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

        // One pending timer per glyph: each step replaces the previous one.
        const timers = [];
        Array.from({ length: count }).forEach((_, i) => {
            const cycle = () => {
                timers[i] = setTimeout(() => {
                    setGlyphs(prev => prev.map((g, idx) => (
                        idx === i ? { ...g, opacity: 1, char: PONEGLYPH_LETTERS[Math.floor(Math.random() * 26)] } : g
                    )));
                    timers[i] = setTimeout(() => {
                        setGlyphs(prev => prev.map((g, idx) => (idx === i ? { ...g, opacity: 0 } : g)));
                        timers[i] = setTimeout(cycle, 1000 + Math.random() * 2000);
                    }, 2000 + Math.random() * 4000);
                }, 500 + Math.random() * 3000);
            };
            cycle();
        });

        return () => timers.forEach(timer => clearTimeout(timer));
    }, [count]);

    return (
        <div className="absolute inset-0 text-(--mode) transition-colors duration-700 motion-reduce:hidden">
            {glyphs.map((g, i) => (
                <span
                    key={i}
                    className="absolute select-none"
                    style={{
                        fontFamily: "'Poneglyph', serif",
                        fontSize: `${g.size}px`,
                        left: `${g.x}%`,
                        top: `${g.y}%`,
                        transform: `rotate(${g.rotate}deg)`,
                        opacity: g.opacity * 0.24,
                        transition: 'opacity 1.4s ease-in-out',
                        lineHeight: 1,
                    }}
                >
                    {g.char}
                </span>
            ))}
        </div>
    );
}

/** Glow and drifting glyphs behind the title and the console. One glow per mode, cross-faded, because gradients do not animate. */
export function SearchBackdrop({ modeId }) {
    return (
        <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[380px] overflow-hidden [mask-image:linear-gradient(to_bottom,transparent,black_16%,black_70%,transparent)]">
            {SEARCH_MODES.map(mode => (
                <div
                    key={mode.id}
                    style={mode.theme}
                    className={cn(
                        // The width is capped by the container so the glow always fades out before the edges, phones included.
                        'absolute inset-0 bg-[radial-gradient(ellipse_min(680px,48%)_200px_at_50%_60%,rgb(var(--mode-rgb)/0.22),transparent)] transition-opacity duration-700 motion-reduce:transition-none',
                        mode.id === modeId ? 'opacity-100' : 'opacity-0',
                    )}
                />
            ))}
            <PoneglyphGlyphs />
        </div>
    );
}

export function SearchHero({ mode }) {
    return (
        <header className="px-2 pb-6 pt-5 text-center sm:pt-9">
            <h1 className="poneglyph-title text-4xl font-extrabold sm:text-5xl">Voix de Toute Chose</h1>
            <p className="poneglyph-muted mx-auto mt-3 max-w-xl text-sm leading-relaxed sm:text-base">
                <span className="font-semibold text-(--mode) transition-colors duration-500">{mode.stone}</span> : {mode.tagline}
            </p>
        </header>
    );
}

export function SearchModeSwitch({ value, onChange }) {
    const buttonsRef = useRef([]);
    const activeIndex = Math.max(0, SEARCH_MODES.findIndex(mode => mode.id === value));

    const move = (step) => {
        const next = (activeIndex + step + SEARCH_MODES.length) % SEARCH_MODES.length;
        onChange(SEARCH_MODES[next].id);
        buttonsRef.current[next]?.focus();
    };

    return (
        <div
            role="radiogroup"
            aria-label="Mode de recherche"
            onKeyDown={event => {
                if (event.key === 'ArrowRight' || event.key === 'ArrowDown') { event.preventDefault(); move(1); }
                if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') { event.preventDefault(); move(-1); }
            }}
            className="relative mx-auto grid h-12 w-full max-w-md grid-cols-3 rounded-xl border border-white/12 bg-[#06111e]/80 p-1 backdrop-blur-md"
        >
            <span
                aria-hidden="true"
                className="pointer-events-none absolute inset-y-1 left-1 w-[calc((100%-0.5rem)/3)] rounded-lg border border-(--mode)/40 bg-(--mode)/12 transition-[transform,background-color,border-color] duration-300 ease-out motion-reduce:transition-none"
                style={{ transform: `translateX(${activeIndex * 100}%)` }}
            />
            {SEARCH_MODES.map(({ id, label, Icon }, index) => {
                const active = index === activeIndex;
                return (
                    <button
                        key={id}
                        ref={node => { buttonsRef.current[index] = node; }}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        tabIndex={active ? 0 : -1}
                        onClick={() => { if (!active) onChange(id); }}
                        className={cn(
                            'relative z-10 flex cursor-pointer items-center justify-center gap-2 rounded-lg text-sm font-medium outline-none transition-colors duration-300 focus-visible:ring-2 focus-visible:ring-(--mode)/60',
                            active ? 'text-(--mode)' : 'text-slate-300 hover:text-white',
                        )}
                    >
                        <Icon aria-hidden="true" className="size-4" />
                        {label}
                    </button>
                );
            })}
        </div>
    );
}
