const puppeteer = require('puppeteer');

(async () => {
  console.log("Launching Headless Chrome with WebGPU enabled...");
  const browser = await puppeteer.launch({
    headless: "new",
    args: [
      '--enable-unsafe-webgpu',
      '--disable-dawn-features=disallow_unsafe_apis',
      '--no-sandbox'
    ]
  });

  const page = await browser.newPage();
  
  // Forward console logs to terminal
  page.on('console', msg => {
    console.log(`[BROWSER] ${msg.text()}`);
  });

  console.log("Navigating to Gladiator Arena (http://localhost:8000/benchmark.html)...");
  await page.goto('http://localhost:8000/benchmark.html', { waitUntil: 'networkidle0' });

  console.log("Waiting for initialization (Pyodide & WHL loading)...");
  await page.waitForFunction(() => {
    const btn = document.getElementById('runBtn');
    return btn && !btn.disabled;
  }, { timeout: 60000 });

  console.log("Initialization complete. Clicking the ⚔️ RUN EXTREME BENCHMARK ⚔️ button!");
  await page.evaluate(() => {
    document.getElementById('runBtn').click();
  });

  console.log("Combat started... Waiting for validation to pass...");
  try {
    await page.waitForFunction(() => {
      const board = document.getElementById('reportBoard');
      return board && board.innerText.includes('Validation Passed');
    }, { timeout: 120000 });

    console.log("\n================ TEST COMPLETED ================\n");
    const finalLog = await page.evaluate(() => document.getElementById('reportBoard').innerText);
    console.log(finalLog);
    console.log("\n================================================\n");
  } catch (e) {
    console.error("Test failed or timed out:", e);
  } finally {
    await browser.close();
  }
})();
