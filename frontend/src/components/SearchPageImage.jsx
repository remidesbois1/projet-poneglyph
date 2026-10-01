"use client";

import { useAuth } from '@/context/AuthContext';
import { useOriginalPageImage } from '@/hooks/useOriginalPageImage';
import { getProxiedImageUrl } from '@/lib/utils';

export default function SearchPageImage({ url, pageId, thumbnail = false, width = 640, alt, ...imageProps }) {
    const { session, loading: checkingSession } = useAuth();
    const original = useOriginalPageImage(pageId, { thumbnail, width });
    const authenticated = Boolean(session?.access_token);
    const source = authenticated ? original.url : getProxiedImageUrl(url, pageId);

    if (checkingSession || (authenticated && !source)) {
        return (
            <div role={original.error ? 'alert' : 'status'}
                className="flex h-full w-full items-center justify-center p-4 text-center text-xs text-slate-400">
                {original.error ? 'Image indisponible. Rechargez la page pour réessayer.' : 'Chargement de l’image…'}
            </div>
        );
    }

    return <img {...imageProps} src={source} crossOrigin="anonymous" alt={alt} />;
}
