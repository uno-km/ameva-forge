const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.text({ limit: '50mb' })); 
app.use(express.static(path.join(__dirname, '..')));

const REPORTS_DIR = path.join(__dirname, 'docs', 'tests', 'results');
const TESTS_DIR = path.join(__dirname, 'packages', 'forge-py', 'tests');

if (!fs.existsSync(REPORTS_DIR)) fs.mkdirSync(REPORTS_DIR);

// List test files
app.get('/api/tests', (req, res) => {
    if (!fs.existsSync(TESTS_DIR)) return res.json([]);
    const files = fs.readdirSync(TESTS_DIR).filter(f => f.endsWith('.py'));
    res.json(files);
});

// List saved reports
app.get('/api/reports', (req, res) => {
    const files = fs.readdirSync(REPORTS_DIR).filter(f => f.endsWith('.txt')).sort().reverse();
    res.json(files);
});

// Save report
app.post('/api/reports/:filename', (req, res) => {
    // FE-C01 Fix: Path traversal ë°©ì?
    const safeName = path.basename(req.params.filename);
    const filepath = path.join(REPORTS_DIR, safeName);
    // Verify the resolved path is still within REPORTS_DIR
    if (!filepath.startsWith(path.resolve(REPORTS_DIR))) {
        return res.status(403).send('Forbidden: invalid filename');
    }
    fs.writeFileSync(filepath, req.body);
    res.sendStatus(200);
});

const PORT = 8000;
app.listen(PORT, () => console.log(`Gladiator Backend running on http://localhost:${PORT}`));

