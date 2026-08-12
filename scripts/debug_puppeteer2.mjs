import puppeteer from 'puppeteer';

(async () => {
  console.log('Launching browser...');
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: "new",
    args: ['--enable-unsafe-webgpu']
  });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('BROWSER_LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE_ERROR:', err.toString()));
  
  console.log('Navigating to benchmark.html...');
  await page.goto('http://localhost:8000/benchmark.html');
  
  console.log('Running test...');
  await page.evaluate(() => window.runTestFile('test_vram_crusher.py'));
  
  console.log('Waiting for test to finish...');
  await page.waitForFunction(() => {
    const el = document.getElementById('reportBoard');
    return el && el.innerText.includes('COMBAT FINISHED');
  }, { timeout: 30000 });
  
  console.log('Test finished! Closing browser...');
  await browser.close();
})();
