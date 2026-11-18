import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true,
  args: ['--disable-gpu-blocklist', '--enable-unsafe-webgpu', '--no-sandbox'],
});
try {
  const page = await browser.newPage();
  page.on('pageerror', e => console.log('[pageerror]', e.stack || e));
  await page.goto('http://localhost:5199/gpu-perf.html', { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => (window as any).__perfResult !== undefined, { timeout: 120000 });
  console.log(await page.evaluate(() => document.getElementById('log')!.textContent));
} finally { await browser.close(); }
