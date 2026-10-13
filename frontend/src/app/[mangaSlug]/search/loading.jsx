import React from 'react';
import { Skeleton } from "@/components/ui/skeleton";

// Same outline as the search page (title, mode switch, console), so nothing jumps when it takes over.
export default function Loading() {
    return (
        <div role="status" aria-label="Chargement de la recherche" className="h-full w-full overflow-hidden">
            <span className="sr-only">Chargement de la recherche…</span>
            <div aria-hidden="true">
                <div className="flex flex-col items-center px-2 pb-6 pt-5 sm:pt-9">
                    <Skeleton className="h-10 w-72 max-w-full bg-white/8 motion-reduce:animate-none sm:h-12 sm:w-[28rem]" />
                    <Skeleton className="mt-3 h-5 w-80 max-w-full bg-white/5 motion-reduce:animate-none" />
                </div>
                <Skeleton className="mx-auto h-12 w-full max-w-md rounded-xl bg-white/8 motion-reduce:animate-none" />
                <div className="mx-auto mt-4 w-full max-w-3xl rounded-2xl border border-white/10 bg-[#071625]/92 p-2">
                    <Skeleton className="h-14 w-full rounded-xl bg-white/8 motion-reduce:animate-none" />
                    <div className="px-1.5 pb-0.5 pt-2.5">
                        <Skeleton className="h-8 w-24 rounded-full bg-white/5 motion-reduce:animate-none" />
                    </div>
                </div>
            </div>
        </div>
    );
}
