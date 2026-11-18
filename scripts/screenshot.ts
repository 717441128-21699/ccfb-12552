/**
 * 打开真实应用，推进一段时间让尾流/烟线/压力场稳定后截图。
 * 输出 scripts/windtunnel.png
 */
import puppeteer from 'puppeteer-core';

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true,
  args: [
    '--disable-gpu-blocklist', '--enable-unsafe-webgpu', '--no-sandbox',
    '--window-size=1600,1000',
  ],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.log('[pageerror]', e.stack || e));
  await page.goto('http://localhost:5199/', { waitUntil: 'networkidle0', timeout: 30000 });
  // 等待仿真推进并让烟线飘出尾流（约 12 秒）
  await new Promise((r) => setTimeout(r, 25000));
  await page.screenshot({ path: 'scripts/windtunnel.png' });
  const stats = await page.evaluate(() => ({
    steps: document.getElementById('m-steps')?.textContent,
    sps: document.getElementById('m-sps')?.textContent,
    cd: document.getElementById('m-cd')?.textContent,
    cdAvg: document.getElementById('m-cd-avg')?.textContent,
    voxel: document.getElementById('voxel-info')?.textContent,
    status: document.getElementById('status')?.textContent,
  }));
  console.log(JSON.stringify(stats, null, 2));
} finally {
  await browser.close();
}
