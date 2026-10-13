"use client";

import { useRef, useState } from 'react';
import { ArrowRight, Check, ChevronRight, Cpu, ImageUp, Loader2, ScanText, Search, SlidersHorizontal, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem } from '@/components/ui/command';
import { Label } from '@/components/ui/label';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { cn } from '@/lib/utils';

const FIELD = 'relative flex items-center gap-3 overflow-hidden rounded-xl border bg-[#071625] transition-[border-color,box-shadow,background-color] duration-300';
const ICON_BUTTON = 'flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 outline-none transition-colors hover:bg-white/8 hover:text-white focus-visible:ring-2 focus-visible:ring-(--mode)/60';
const SUBMIT = 'inline-flex min-h-11 min-w-11 shrink-0 cursor-pointer items-center justify-center gap-2 rounded-lg bg-(--mode-strong) px-3 text-xs font-semibold text-(--mode-ink) shadow-[0_8px_24px_rgb(var(--mode-rgb)/0.28)] outline-none transition-[background-color,box-shadow,opacity] duration-300 hover:bg-(--mode-strong-hover) focus-visible:ring-2 focus-visible:ring-(--mode)/60 focus-visible:ring-offset-2 focus-visible:ring-offset-[#071625] disabled:pointer-events-none disabled:opacity-45 disabled:shadow-none sm:px-4';

/** Thin bar on the bottom edge of the field: model download or bubble reading, without moving the layout. */
function ProgressEdge({ value }) {
    if (value == null) return null;
    const percentage = Math.round(Math.min(1, Math.max(0, value)) * 100);
    return (
        <span aria-hidden="true" className="pointer-events-none absolute inset-x-0 bottom-0 h-0.5 bg-white/8">
            <span className="block h-full bg-(--mode) transition-[width] duration-300 motion-reduce:transition-none" style={{ width: `${percentage}%` }} />
        </span>
    );
}

function QueryField({ mode, query, inputRef, onChange, onClear, onSubmit, isLoading, progress }) {
    return (
        <div role="search" className={cn(FIELD, 'min-h-14 border-(--mode)/25 pl-4 pr-1.5 focus-within:border-(--mode)/55 focus-within:shadow-[0_0_0_3px_rgb(var(--mode-rgb)/0.12)]')}>
            {isLoading
                ? <Loader2 aria-hidden="true" className="size-[19px] shrink-0 animate-spin text-(--mode) motion-reduce:animate-none" />
                : <Search aria-hidden="true" className="size-[19px] shrink-0 text-(--mode) transition-colors duration-500" />}
            <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={onChange}
                onKeyDown={event => {
                    if (event.key === 'Enter') onSubmit();
                    if (event.key === 'Escape' && query) { event.preventDefault(); onClear(); }
                }}
                enterKeyHint="search"
                autoComplete="off"
                aria-label={`Recherche en mode ${mode.label.toLowerCase()}`}
                placeholder={mode.placeholder}
                className="h-14 min-w-0 flex-1 !border-0 !bg-transparent text-base text-white outline-none placeholder:text-slate-500"
            />
            {query && (
                <button type="button" aria-label="Effacer la recherche" onClick={onClear} className={ICON_BUTTON}>
                    <X aria-hidden="true" className="size-4" />
                </button>
            )}
            <button type="button" aria-label="Lancer la recherche" onClick={onSubmit} disabled={isLoading || query.trim().length < 2} className={SUBMIT}>
                <span className="hidden sm:inline">Rechercher</span>
                <ArrowRight aria-hidden="true" className="size-4" />
            </button>
            <ProgressEdge value={progress} />
        </div>
    );
}

function ImageField({ image, onSubmit, isLoading, progress }) {
    const fileInputRef = useRef(null);
    const hasImage = Boolean(image.previewUrl);
    const title = image.fileName || (image.bubbleCount > 0 ? 'Image précédente' : 'Déposez, collez ou choisissez une image');
    const hint = image.status || (image.bubbleCount > 0
        ? `${image.bubbleCount} ${image.bubbleCount === 1 ? 'bulle lue' : 'bulles lues'}`
        : 'Capture, photo ou extrait de page · Ctrl + V pour coller');

    return (
        <div className={cn(FIELD, 'border-dashed p-2', image.isDragging ? 'border-(--mode) bg-(--mode)/8' : 'border-(--mode)/35')}>
            <button
                type="button"
                aria-label={hasImage ? 'Changer d’image' : 'Choisir une image'}
                title={hasImage ? 'Changer d’image' : 'Choisir une image'}
                onClick={() => fileInputRef.current?.click()}
                className="flex size-14 shrink-0 cursor-pointer items-center justify-center overflow-hidden rounded-lg border border-white/10 bg-white/[0.045] text-(--mode) outline-none transition-colors hover:border-(--mode)/50 hover:bg-white/8 focus-visible:ring-2 focus-visible:ring-(--mode)/60"
            >
                {hasImage
                    // Local object URL of the visitor's own file: nothing for next/image to optimise.
                    // eslint-disable-next-line @next/next/no-img-element
                    ? <img src={image.previewUrl} alt="" className="size-full object-cover" />
                    : <ImageUp aria-hidden="true" className="size-6" strokeWidth={1.5} />}
            </button>
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={event => {
                    image.onPick(event.target.files?.[0]);
                    // Lets the same file be picked again after it was removed.
                    event.target.value = '';
                }}
            />

            <div className="min-w-0 flex-1 text-left">
                <p className="truncate text-sm font-medium text-white">{title}</p>
                <p role="status" className={cn('mt-1 truncate text-xs', image.hasError ? 'text-red-300' : 'text-slate-400')}>{hint}</p>
            </div>

            {(hasImage || image.bubbleCount > 0) && (
                <button type="button" aria-label="Retirer l’image" title="Retirer l’image" onClick={image.onClear} className={ICON_BUTTON}>
                    <X aria-hidden="true" className="size-4" />
                </button>
            )}
            <button type="button" aria-label="Scanner l’image" onClick={onSubmit} disabled={isLoading || (!hasImage && image.bubbleCount === 0)} className={SUBMIT}>
                {isLoading
                    ? <Loader2 aria-hidden="true" className="size-4 animate-spin motion-reduce:animate-none" />
                    : <ScanText aria-hidden="true" className="size-4" />}
                <span className="hidden sm:inline">Scanner</span>
            </button>
            <ProgressEdge value={progress} />
        </div>
    );
}

function FilterChip({ children, removeLabel, onRemove }) {
    return (
        <span className="inline-flex h-8 max-w-full items-center gap-0.5 rounded-full border border-(--mode)/30 bg-(--mode)/10 pl-3 pr-1 text-xs font-medium text-slate-100">
            <span className="truncate">{children}</span>
            <button type="button" aria-label={removeLabel} onClick={onRemove} className="flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-slate-300 outline-none transition-colors hover:bg-white/10 hover:text-white focus-visible:ring-2 focus-visible:ring-(--mode)/60">
                <X aria-hidden="true" className="size-3" />
            </button>
        </span>
    );
}

function SearchFilters({ filters, theme }) {
    const [charPopoverOpen, setCharPopoverOpen] = useState(false);
    const { tomes, selectedTome, selectedArc, selectedCharacters } = filters;
    const activeCount = selectedCharacters.length + (selectedArc !== 'all' ? 1 : 0) + (selectedTome !== 'all' ? 1 : 0);

    return (
        <>
            <Popover>
                <PopoverTrigger asChild>
                    <button
                        type="button"
                        className={cn(
                            'inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-full border px-3 text-xs font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-(--mode)/60 data-[state=open]:border-(--mode)/45 data-[state=open]:text-white',
                            activeCount > 0 ? 'border-(--mode)/45 bg-(--mode)/12 text-white' : 'border-white/10 bg-white/[0.035] text-slate-300 hover:border-white/20 hover:text-white',
                        )}
                    >
                        <SlidersHorizontal aria-hidden="true" className="size-3.5" />
                        Filtres
                        {activeCount > 0 && (
                            <span className="flex size-4 items-center justify-center rounded-full bg-(--mode-strong) text-[10px] font-semibold tabular-nums text-(--mode-ink)">{activeCount}</span>
                        )}
                    </button>
                </PopoverTrigger>
                <PopoverContent align="start" sideOffset={10} style={theme} className="w-[min(21rem,calc(100vw-2rem))] rounded-xl border-white/12 p-4 text-slate-100">
                    <Label className="mb-2 block text-xs font-medium leading-5 text-slate-300">Tome</Label>
                    <Select value={selectedTome} onValueChange={filters.onTomeChange}>
                        <SelectTrigger aria-label="Filtrer par tome" className="w-full rounded-lg shadow-none data-[size=default]:h-10">
                            <SelectValue placeholder="Tous les tomes" />
                        </SelectTrigger>
                        <SelectContent position="popper" className="max-h-64">
                            <SelectItem value="all">Tous les tomes</SelectItem>
                            {tomes.map(tome => (
                                <SelectItem key={tome.numero} value={tome.numero.toString()}>
                                    <span className="shrink-0 tabular-nums">Tome {tome.numero}</span>
                                    {tome.titre && <span className="truncate text-slate-400">{tome.titre}</span>}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>

                    <div className="mt-4 border-t border-white/10 pt-4">
                        <p className="mb-3 flex items-center justify-between gap-3 text-xs font-medium leading-5 text-slate-400">
                            Personnages et arc narratif
                            <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10px] font-normal">Indisponible</span>
                        </p>
                        <div className="grid grid-cols-2 gap-2 opacity-50">
                            <Popover open={charPopoverOpen} onOpenChange={setCharPopoverOpen}>
                                <PopoverTrigger asChild>
                                    <Button disabled variant="outline" className="h-10 w-full justify-between rounded-lg border-white/12 bg-transparent px-3 text-sm font-normal text-slate-300 shadow-none">
                                        {selectedCharacters.length > 0 ? `${selectedCharacters.length} persos.` : 'Personnages'}
                                        <ChevronRight aria-hidden="true" className={cn('size-4 transition-transform', charPopoverOpen && 'rotate-90')} />
                                    </Button>
                                </PopoverTrigger>
                                <PopoverContent className="w-[300px] rounded-xl border-white/12 p-0" align="start">
                                    <Command className="rounded-xl bg-transparent">
                                        <CommandInput placeholder="Rechercher…" className="h-11" />
                                        <CommandEmpty>Aucun résultat.</CommandEmpty>
                                        <CommandGroup className="max-h-64 overflow-auto p-2">
                                            {filters.characters.map(char => (
                                                <CommandItem
                                                    key={char}
                                                    className="mb-0.5 h-9 rounded-lg text-xs"
                                                    onSelect={() => filters.onCharactersChange(prev => (
                                                        prev.includes(char) ? prev.filter(c => c !== char) : [...prev, char]
                                                    ))}
                                                >
                                                    <span className={cn('mr-2 flex size-4 items-center justify-center rounded-sm border border-white/25', selectedCharacters.includes(char) && 'border-[#3d86ff] bg-[#3d86ff]')}>
                                                        <Check aria-hidden="true" className={cn('size-3 text-white', selectedCharacters.includes(char) ? 'opacity-100' : 'opacity-0')} />
                                                    </span>
                                                    {char}
                                                </CommandItem>
                                            ))}
                                        </CommandGroup>
                                    </Command>
                                </PopoverContent>
                            </Popover>
                            <Select value={selectedArc} onValueChange={filters.onArcChange} disabled>
                                <SelectTrigger aria-label="Filtrer par arc narratif" className="w-full rounded-lg shadow-none data-[size=default]:h-10">
                                    <SelectValue placeholder="Arc narratif" />
                                </SelectTrigger>
                                <SelectContent position="popper" className="max-h-64">
                                    <SelectItem value="all">Tous les arcs</SelectItem>
                                    {filters.arcs.map(arc => <SelectItem key={arc} value={arc}>{arc}</SelectItem>)}
                                </SelectContent>
                            </Select>
                        </div>
                    </div>
                </PopoverContent>
            </Popover>

            {selectedTome !== 'all' && (
                <FilterChip removeLabel={`Retirer le filtre tome ${selectedTome}`} onRemove={() => filters.onTomeChange('all')}>
                    Tome <span className="tabular-nums">{selectedTome}</span>
                </FilterChip>
            )}
            {selectedArc !== 'all' && (
                <FilterChip removeLabel={`Retirer le filtre ${selectedArc}`} onRemove={() => filters.onArcChange('all')}>{selectedArc}</FilterChip>
            )}
            {selectedCharacters.map(char => (
                <FilterChip key={char} removeLabel={`Retirer le filtre ${char}`} onRemove={() => filters.onCharactersChange(prev => prev.filter(c => c !== char))}>{char}</FilterChip>
            ))}
            {activeCount > 1 && (
                <button type="button" onClick={filters.onReset} className="h-8 cursor-pointer rounded-full px-2 text-xs text-slate-400 outline-none transition-colors hover:text-white focus-visible:ring-2 focus-visible:ring-(--mode)/60">
                    Tout réinitialiser
                </button>
            )}
        </>
    );
}

/** Field, filters and mode options. Presentation only: every search decision stays in SearchClient. */
export default function SearchConsole({ mode, query, inputRef, onQueryChange, onClearQuery, onSubmit, isLoading, progress, localOnly, onLocalOnlyChange, localModelStatus, image, filters }) {
    return (
        <div className="mx-auto w-full max-w-3xl rounded-2xl border border-(--mode)/25 bg-[#071625]/92 p-2 shadow-[0_24px_80px_rgba(0,0,0,0.38),0_0_45px_rgb(var(--mode-rgb)/0.08)] backdrop-blur-xl transition-[border-color,box-shadow] duration-500">
            {mode.id === 'ocr'
                ? <ImageField image={image} onSubmit={onSubmit} isLoading={isLoading} progress={progress} />
                : <QueryField mode={mode} query={query} inputRef={inputRef} onChange={onQueryChange} onClear={onClearQuery} onSubmit={onSubmit} isLoading={isLoading} progress={progress} />}

            <div className="flex flex-wrap items-center gap-2 px-1.5 pb-0.5 pt-2.5 text-xs text-slate-400">
                <SearchFilters filters={filters} theme={mode.theme} />

                <div className="ml-auto flex min-w-0 items-center gap-2 pl-2">
                    {mode.id === 'keyword' && <span className="hidden sm:inline">Résultats pendant la frappe</span>}
                    {mode.id === 'semantic' && (
                        <label
                            htmlFor="local-only-search"
                            title="L’embedding de votre requête est calculé dans le navigateur avec F2LLM."
                            className="flex h-8 min-w-0 cursor-pointer items-center gap-2 text-slate-300"
                        >
                            <Cpu aria-hidden="true" className="size-3.5 shrink-0 text-slate-400" />
                            <span className="truncate">{(localOnly && localModelStatus) || 'Calcul local'}</span>
                            <Switch
                                id="local-only-search"
                                checked={localOnly}
                                onCheckedChange={onLocalOnlyChange}
                                className="data-[state=checked]:bg-(--mode-strong) data-[state=unchecked]:bg-white/15 [&>[data-slot=switch-thumb]]:bg-white"
                            />
                        </label>
                    )}
                    {mode.id === 'ocr' && (
                        <span className="flex min-w-0 items-center gap-1.5">
                            <Cpu aria-hidden="true" className="size-3.5 shrink-0" />
                            <span className="truncate">Lecture dans votre navigateur</span>
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}
