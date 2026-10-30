import { createServer } from 'vite';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const fixtureRoot = await realpath(process.env.FALCON_BENCH_FIXTURES || fileURLToPath(new URL(
    '../../docker_scripts/finetune_surya_bubble_ocr/outputs_surya_bubble_ocr/rtx5090-20261030/private-webgpu-fixtures/', import.meta.url)));
const outputRoot = path.resolve(process.env.FALCON_BENCH_OUTPUT || path.join(fixtureRoot, '..', 'falcon-bench'));
const port = Number(process.env.FALCON_BENCH_PORT || 3002);
await mkdir(outputRoot, { recursive: true });

const bench = {
    name: 'falcon-private-bench',
    configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
            const url = new URL(req.url, `http://127.0.0.1:${port}`);
            if (url.pathname.startsWith('/__falcon_fixtures/')) {
                try {
                    const relative = decodeURIComponent(url.pathname.slice('/__falcon_fixtures/'.length));
                    if (!/^(test128|full1691|falcon-stress)\/(examples\.json|bubble-[0-9]+\.png)$/.test(relative)) throw new Error('Invalid fixture');
                    const resolved = await realpath(path.join(fixtureRoot, relative));
                    if (!resolved.startsWith(fixtureRoot + path.sep)) throw new Error('Invalid fixture path');
                    res.setHeader('Content-Type', relative.endsWith('.json') ? 'application/json' : 'image/png');
                    createReadStream(resolved).pipe(res);
                } catch { res.statusCode = 404; res.end(); }
                return;
            }
            if (url.pathname === '/__falcon_results' && req.method === 'POST') {
                if (req.headers.origin !== `http://127.0.0.1:${port}` || !/^[a-z0-9_-]{1,64}$/.test(url.searchParams.get('name') || '')) {
                    res.statusCode = 400; res.end(); return;
                }
                const chunks = []; let size = 0;
                for await (const chunk of req) { size += chunk.length; if (size > 4_000_000) { res.statusCode = 413; res.end(); return; } chunks.push(chunk); }
                try {
                    const data = Buffer.concat(chunks); JSON.parse(data.toString('utf8'));
                    await writeFile(path.join(outputRoot, `${url.searchParams.get('name')}.json`), data, { flag: 'wx' });
                    res.end('saved');
                } catch (error) { res.statusCode = 400; res.end(error.code === 'EEXIST' ? 'Result already exists' : 'Invalid result'); }
                return;
            }
            next();
        });
    },
};
// Private evaluation data is served only on loopback, never copied to public/.
await readFile(path.join(fixtureRoot, 'test128', 'examples.json'));
const server = await createServer({ configFile: false, root, plugins: [bench],
    define: { 'process.env.NODE_ENV': '"development"',
        'process.env.NEXT_PUBLIC_FALCON_MODEL_BASE': JSON.stringify(process.env.FALCON_BENCH_MODEL_BASE || '') },
    server: { host: '127.0.0.1', port, strictPort: true, headers: {
        'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
    } },
});
await server.listen();
console.log(`Falcon benchmark: http://127.0.0.1:${port}/bench/falcon.html`);
