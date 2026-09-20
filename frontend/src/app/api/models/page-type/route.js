const MODEL_REVISION = '5830f91ae3524a7a9da8d33a7a1b5242b677e5f1';
const MODEL_URL = `https://huggingface.co/Remidesbois/Poneglyph-Classifier/resolve/${MODEL_REVISION}/page_type_classifier.onnx`;

export const runtime = 'nodejs';

/**
 * Hugging Face's Xet download redirect is not CORS-readable from the isolated
 * browser worker. Stream the remote model through the Next origin instead;
 * this remains a remote Hugging Face model and is never bundled as a static
 * frontend asset.
 */
export async function GET() {
    const upstream = await fetch(MODEL_URL, {
        next: { revalidate: 3600 },
    });
    if (!upstream.ok || !upstream.body) {
        return Response.json(
            { error: `Le modèle Hugging Face est indisponible (HTTP ${upstream.status}).` },
            { status: 502 },
        );
    }

    const headers = new Headers({
        'Content-Type': upstream.headers.get('content-type') || 'application/octet-stream',
        'Cache-Control': 'public, max-age=3600, s-maxage=3600',
    });
    const contentLength = upstream.headers.get('content-length');
    if (contentLength) headers.set('Content-Length', contentLength);
    return new Response(upstream.body, { headers });
}
