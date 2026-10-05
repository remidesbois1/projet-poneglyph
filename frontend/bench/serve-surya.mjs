import { createServer } from 'vite';
import { fileURLToPath } from 'node:url';

const server = await createServer({
    configFile: false,
    root: fileURLToPath(new URL('../', import.meta.url)),
    define: { 'process.env.NODE_ENV': '"development"', 'process.env.NEXT_PUBLIC_SURYA_MODEL_BASE': '""' },
    server: { host: '127.0.0.1', port: 3002, strictPort: true, headers: {
        'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
    } },
});
await server.listen();
console.log('Surya benchmark: http://127.0.0.1:3002/bench/surya.html');
