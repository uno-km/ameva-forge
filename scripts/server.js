const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
// L-01: Limit report payload to 1MB
app.use(express.text({ limit: '1mb' })); 
app.use(express.static(path.join(__dirname, '..')));

const REPORTS_DIR = path.join(__dirname, '..', 'docs', 'tests', 'results');
const TESTS_DIR = path.join(__dirname, '..', 'packages', 'forge-py', 'tests');

if (!fs.existsSync(REPORTS_DIR)) fs.mkdirSync(REPORTS_DIR, { recursive: true });

// List test files
app.get('/api/tests', (req, res) => {
    if (!fs.existsSync(TESTS_DIR)) return res.json([]);
    const files = fs.readdirSync(TESTS_DIR).filter(f => f.endsWith('.py'));
    res.json(files);
});

// List saved reports
app.get('/api/reports', (req, res) => {
    const files = fs.readdirSync(REPORTS_DIR).filter(f => f.endsWith('.txt') || f.endsWith('.json')).sort().reverse();
    res.json(files);
});

// Save report
app.post('/api/reports/:filename', (req, res) => {
    // L-01: Path traversal 방어 및 확장자 화이트리스트 검사
    const safeName = path.basename(req.params.filename);
    const ext = path.extname(safeName).toLowerCase();
    if (ext !== '.txt' && ext !== '.json') {
        return res.status(400).send('Bad Request: only .txt and .json files are allowed');
    }
    const resolvedReportsDir = path.resolve(REPORTS_DIR);
    const filepath = path.join(resolvedReportsDir, safeName);
    if (!filepath.startsWith(resolvedReportsDir)) {
        return res.status(403).send('Forbidden: invalid filename');
    }
    fs.writeFileSync(filepath, req.body, 'utf8');
    res.sendStatus(200);
});

const PORT = 8000;
app.listen(PORT, () => console.log(Gladiator Backend running on http://localhost:));
