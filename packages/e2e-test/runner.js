import puppeteer from 'puppeteer';
import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const server = http.createServer((req, res) => {
  const publicDir = path.join(__dirname, 'public');
  const urlStr = req.url || '/';
  const pathname = urlStr.split('?')[0];

  if (pathname === '/api/wheels') {
    fs.readdir(publicDir, (err, files) => {
      if (err) { res.writeHead(500); return res.end(); }
      const wheels = files.filter(f => f.endsWith('.whl'));
      wheels.sort((a, b) => {
        const statA = fs.statSync(path.join(publicDir, a));
        const statB = fs.statSync(path.join(publicDir, b));
        return statB.mtimeMs - statA.mtimeMs;
      });
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(wheels));
    });
    return;
  }
  
  let filePath = path.join(publicDir, pathname === '/' ? 'index.html' : pathname);
  
  const extname = path.extname(filePath);
  let contentType = 'text/html';
  switch (extname) {
    case '.js': contentType = 'text/javascript'; break;
    case '.whl': contentType = 'application/octet-stream'; break;
    case '.py': contentType = 'text/plain'; break;
  }
  
  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found', 'utf-8');
    } else {
      res.writeHead(200, { 
        'Content-Type': contentType,
        'Cache-Control': 'no-store, no-cache, must-revalidate'
      });
      res.end(content, 'utf-8');
    }
  });
});

server.listen(3000, async () => {
  console.log('Server running on http://localhost:3000');
  
  const browser = await puppeteer.launch({
    executablePath: process.env.BROWSER_PATH || undefined,
    headless: "new",
    args: [
      '--enable-unsafe-webgpu',
      '--disable-gpu-sandbox',
      '--no-sandbox',
      '--disable-setuid-sandbox'
    ]
  });
  
  const page = await browser.newPage();
  
  page.on('console', msg => {
    const text = msg.text();
    console.log('[Browser]', text);
    if (text.includes('E2E_SUCCESS')) {
      console.log('Test Passed!');
      browser.close().then(() => { server.close(); process.exit(0); });
    } else if (text.includes('E2E_ERROR')) {
      console.error('Test Failed!');
      browser.close().then(() => { server.close(); process.exit(1); });
    }
  });
  
  await page.goto('http://localhost:3000');
});
