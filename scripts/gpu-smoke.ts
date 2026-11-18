/**
 * 用 puppeteer-core 驱动本机 Chrome，在真实 WebGPU 环境里跑 gpu-smoke.html。
 * 用法：npx tsx scripts/gpu-smoke.ts
 */
import puppeteer from 'puppeteer-core';

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const URL = 'http://localhost:5199/gpu-smoke.html';

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    '--disable-gpu-blocklist',
    '--enable-unsafe-webgpu',
    '--enable-webgpu-developer-features',
    '--no-sandbox',
    '--window-size=1280,900',
  ],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  const consoleMsgs: string[] = [];
  page.on('console', (m) => consoleMsgs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => consoleMsgs.push(`[pageerror] ${e.stack || e}`));

  await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30000 });

  const result = await page.waitForFunction(
    () => (window as unknown as { __smokeResult?: unknown }).__smokeResult,
    { timeout: 60000 },
  ).then((h) => h.jsonValue() as Promise<{ passed: number; total: number; results: Array<{ name: string; ok: boolean }> }>);

  const body = await page.evaluate(() => document.getElementById('log')?.textContent ?? '');
  console.log(body);
  if (consoleMsgs.length) {
    console.log('--- 浏览器控制台 ---');
    console.log(consoleMsgs.join('\n'));
  }
  console.log(`\n结果：${result.passed}/${result.total}`);
  if (result.passed !== result.total) process.exitCode = 1;
} finally {
  await browser.close();
}
