/**
 * 端到端验证 A/B 阻力方案对比：等仿真稳定后存方案 A、改造型再存方案 B，
 * 检查面板出现带 ΔCd 的对比结论。
 */
import puppeteer from 'puppeteer-core';

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true,
  args: ['--disable-gpu-blocklist', '--enable-unsafe-webgpu', '--no-sandbox'],
});
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('[pageerror]', e.stack || e));
  await page.goto('http://localhost:5199/', { waitUntil: 'networkidle0', timeout: 30000 });

  // 等气动力采样出 Cd
  await page.waitForFunction(
    () => {
      const v = document.getElementById('m-cd-avg')?.textContent ?? '';
      return Number.isFinite(parseFloat(v));
    },
    { timeout: 30000 },
  );
  const cdA = await page.evaluate(() => document.getElementById('m-cd-avg')!.textContent);

  await page.click('#btn-snapshot-a');
  // 改尾翼角度并热启动重体素化
  await page.evaluate(() => {
    const wing = document.getElementById('wing') as HTMLInputElement;
    wing.value = '25'; wing.dispatchEvent(new Event('input'));
  });
  await page.click('#btn-revoxel');
  // 等热启动后重新累积 Cd
  await new Promise((r) => setTimeout(r, 4000));
  await page.click('#btn-snapshot-b');

  const result = await page.evaluate(() => document.getElementById('ab-result')!.textContent);
  const hasDelta = result.includes('ΔCd');
  console.log('CdA =', cdA);
  console.log('A/B 结论:', result.replace(/\n/g, ' | '));
  console.log(hasDelta ? 'PASS: 阻力差对比已生成' : 'FAIL: 缺少 ΔCd');
  if (!hasDelta) process.exitCode = 1;
} finally {
  await browser.close();
}
