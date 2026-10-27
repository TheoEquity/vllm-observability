// Real Grafana screenshots, with absolute UTC ranges. Bundled Playwright only.
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require(path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
const root = path.resolve(__dirname, '..');
const out = path.resolve(process.argv[2]);

(async () => {
  const browser = await chromium.launch({executablePath: path.join(process.env.LOCALAPPDATA, 'ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-win64/chrome-headless-shell.exe'), headless: true, args: ['--disable-gpu']});
  const page = await browser.newPage({viewport: {width: 1800, height: 1400}, deviceScaleFactor: 1});
  const evidence = [];
  for (const name of (process.argv[3] === '--preview' ? [] : ['A', 'B'])) {
    const data = JSON.parse(fs.readFileSync(path.join(out, `case-${name}.json`), 'utf8'));
    const from = Date.parse(data.started_at_utc) - 15000;
    const to = Date.parse(data.finished_at_utc) + 20000;
    const url = `http://127.0.0.1:48175/d/vllm-observability?orgId=1&from=${from}&to=${to}&timezone=utc&refresh=`;
    await page.goto(url, {waitUntil: 'networkidle'});
    await page.getByText('End-to-end latency — histogram estimates', {exact: true}).waitFor();
    await page.waitForTimeout(3000);
    await page.screenshot({path: path.join(out, `grafana-case-${name}.png`), fullPage: true});
    evidence.push({case: name, url, captured_at_utc: new Date().toISOString(), from_utc: new Date(from).toISOString(), to_utc: new Date(to).toISOString(), panel_title_visible: true});
  }
  if (evidence.length) fs.writeFileSync(path.join(out, 'screenshots.json'), JSON.stringify(evidence, null, 2));
  if (fs.existsSync(path.join(root, 'report.html'))) {
    await page.goto(pathToFileURL(path.join(root, 'report.html')).href, {waitUntil: 'networkidle'});
    await page.setViewportSize({width: 1800, height: 1600});
    await page.screenshot({path: path.join(out, 'report-preview.png'), fullPage: false});
  }
  await browser.close();
  console.log(JSON.stringify(evidence));
})().catch(error => {console.error(error); process.exitCode = 1;});
