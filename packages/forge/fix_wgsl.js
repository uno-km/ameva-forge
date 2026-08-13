const fs = require('fs');
const path = require('path');
const dir = 'src/tensor/kernels';
const files = fs.readdirSync(dir).filter(f => f.endsWith('.wgsl.ts'));
for (const file of files) {
  const filePath = path.join(dir, file);
  let content = fs.readFileSync(filePath, 'utf8');
  content = content.replace(/\\\`;/g, '\`;');
  fs.writeFileSync(filePath, content, 'utf8');
}
console.log('Fixed WGSL files');
