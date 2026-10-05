// Local, read-only report viewer. Usage: node serve_report.cjs <run-directory> [port]
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = fs.realpathSync(process.argv[2]);
const port = Number(process.argv[3] || 8765);
const mime = {'.html':'text/html; charset=utf-8', '.json':'application/json',
  '.svg':'image/svg+xml', '.png':'image/png', '.csv':'text/csv; charset=utf-8'};
http.createServer((req,res) => {
  try {
    if (!['GET','HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
    let route = decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    if (route === '/') route = fs.existsSync(path.join(root,'report.html')) ? '/report.html' : '/training_dashboard.html';
    const file = fs.realpathSync(path.resolve(root, '.' + route));
    if (!file.startsWith(root + path.sep) || !fs.statSync(file).isFile()) { res.writeHead(403).end(); return; }
    res.writeHead(200, {'Content-Type':mime[path.extname(file)] || 'application/octet-stream', 'Cache-Control':'no-store'});
    if (req.method === 'HEAD') res.end(); else fs.createReadStream(file).pipe(res);
  } catch { res.writeHead(404).end('Not found'); }
}).listen(port,'127.0.0.1',()=>console.log(`Surya report: http://127.0.0.1:${port}`));
