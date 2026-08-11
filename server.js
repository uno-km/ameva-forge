const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.text({ limit: '50mb' })); 
app.use(express.static(__dirname));

const REPORTS_DIR = path.join(__dirname, 'reports');
const TESTS_DIR = path.join(__dirname, 'packages', 'ameva-tensor-py', 'tests');

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
    const filename = req.params.filename;
    const filepath = path.join(REPORTS_DIR, filename);
    fs.writeFileSync(filepath, req.body);
    res.sendStatus(200);
});

const PORT = 8000;
app.listen(PORT, () => console.log(`Gladiator Backend running on http://localhost:${PORT}`));
