// End-to-end smoke test of Closet Assistant (headless Chromium).
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const URL = process.env.APP || 'http://127.0.0.1:8765/index.html';
const IMG = process.env.IMGDIR || './test-photos', OUT = process.env.OUT || './smoke-out';
fs.mkdirSync(OUT, { recursive: true });
const log = [], errs = [];
const L = (...a) => { const s = a.join(' '); log.push(s); console.log(s); };
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const ctx = await chromium.launchPersistentContext(process.env.PROFILE || './smoke-profile', {
    headless: true, viewport: { width: 390, height: 844 }, deviceScaleFactor: 2,
    acceptDownloads: true, proxy: process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: '127.0.0.1,localhost' } : undefined,
    geolocation: { latitude: 37.98, longitude: 23.72 }, permissions: ['geolocation'],
  });
  const page = ctx.pages()[0] || await ctx.newPage();
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push(`[${m.type()}] ${m.text()}`) });
  page.on('pageerror', e => errs.push(`[pageerror] ${e.message}\n${(e.stack || '').split('\n').slice(0, 3).join('\n')}`));
  page.on('requestfailed', r => errs.push(`[reqfail] ${r.url().slice(0, 120)} ${r.failure() && r.failure().errorText}`));
  page.on('dialog', d => { L('dialog:', d.message()); d.accept() });

  await page.goto(URL);
  if (process.env.FRESH) { await page.evaluate(async () => { localStorage.clear(); await new Promise(r => { const q = indexedDB.deleteDatabase('closet'); q.onsuccess = q.onerror = q.onblocked = r }) }); }
  await page.evaluate(() => { localStorage.setItem('ml', '1'); localStorage.setItem('gender', 'female'); localStorage.setItem('lang', 'el') });
  await page.reload(); await sleep(800);
  await page.screenshot({ path: `${OUT}/01-empty.png` });

  // ---- add photos (gallery, multiple) ----
  const files = fs.readdirSync(IMG).filter(f => f.endsWith('.jpg')).map(f => path.join(IMG, f));
  if (!process.env.NOADD) {
    await page.click('[data-act=fab]');
    await page.locator('input[data-in][multiple]').setInputFiles(files);
    const t0 = Date.now(); let saved = 0, last = '';
    while (Date.now() - t0 < 600000 && saved < files.length) {
      await sleep(400);
      const st = await page.evaluate(() => { const s = document.querySelector('#sheet'); return s ? s.innerText.replace(/\s+/g, ' ').slice(0, 220) : '' });
      const has = async sel => (await page.locator(`#sheet ${sel}`).count()) > 0;
      if (st !== last) { last = st }
      let act = null;
      for (const a of ['yes', 'guessyes', 'subyes', 'dupadd']) if (await has(`[data-act=${a}]`)) { act = a; break }
      if (!act && await has('[data-act=save]:not([disabled])')) act = 'save';
      if (!act) continue;
      L(`  [${act}] ${st.slice(0, 160)}`);
      await page.locator(`#sheet [data-act=${act}]`).first().click();
      if (act === 'save') { saved++; await page.screenshot({ path: `${OUT}/02-saved-${saved}.png` }) }
    }
    L('saved', saved, 'of', files.length, 'in', ((Date.now() - t0) / 1000).toFixed(0), 's');
  }
  await sleep(800);
  await page.screenshot({ path: `${OUT}/03-closet.png`, fullPage: true });
  L('items in grid:', await page.locator('#gridwrap .tile').count());
  const its = await page.evaluate(() => S.items.slice().sort((a,b)=>a.t-b.t).map(i => [i.cat, i.sub||'', cname(i.color), (i.cols||[]).map(cname).join('+'), patOf(i), i.mat||'', (occOf(i)||[]).join(','), (ssnOf(i)||[]).join(',')].join(' | ')));
  its.forEach((x, k) => L('  item', k + 1, x));

  // ---- open an item view, then close ----
  const first = page.locator('#gridwrap [data-act=item]').first();
  if (await first.count()) { await first.click(); await sleep(300); await page.screenshot({ path: `${OUT}/04-view.png` }); await page.keyboard.press('Escape'); const c = page.locator('#sheet [data-act=close]'); if (await c.count()) await c.first().click(); }

  // ---- suggest ----
  await page.click('nav [data-act=tab][data-v=suggest]'); await sleep(300);
  if (await page.locator('#q').count()) { await page.fill('#q', 'Αθήνα'); await page.click('[data-act=search]'); await sleep(2500); const p = page.locator('[data-act=pickloc]'); L('geo results:', await p.count()); if (await p.count()) await p.first().click(); }
  await sleep(3000);
  await page.click('[data-act=go]'); await sleep(1500);
  await page.screenshot({ path: `${OUT}/05-suggest.png`, fullPage: true });
  L('outfits:', await page.locator('.outfit').count());
  L('suggest text:', (await page.locator('#app').innerText()).replace(/\s+/g, ' ').slice(0, 700));
  // collage
  const pic = page.locator('[data-act=pic]').first();
  if (await pic.count()) {
    const dl = page.waitForEvent('download', { timeout: 6000 }).catch(() => null);
    await pic.click(); await sleep(1500);
    const d = await dl; if (d) { await d.saveAs(`${OUT}/collage-dl.png`); L('collage downloaded') }
    await page.screenshot({ path: `${OUT}/06-pic.png` });
    const c = page.locator('#sheet [data-act=close]'); if (await c.count()) await c.first().click();
  }
  // wore it, rate, fav
  for (const a of ['rate', 'favsave', 'wore']) { const b = page.locator(`[data-act=${a}]`).first(); if (await b.count()) { await b.click(); await sleep(400) } }

  // ---- diary ----
  await page.click('nav [data-act=tab][data-v=diary]'); await sleep(400);
  await page.screenshot({ path: `${OUT}/07-diary.png`, fullPage: true });

  // ---- drawer items ----
  for (const a of ['stats', 'pack', 'audit', 'profile', 'settings']) {
    await page.click('header [data-act=menu]'); await sleep(250);
    await page.click(`.drawer [data-act=${a}]`); await sleep(1500);
    await page.screenshot({ path: `${OUT}/08-${a}.png`, fullPage: true });
    const c = page.locator('#sheet [data-act=close]'); if (await c.count()) await c.last().click(); else await page.keyboard.press('Escape');
    await sleep(200);
  }
  // export
  await page.click('header [data-act=menu]'); await sleep(250);
  const dl = page.waitForEvent('download', { timeout: 8000 }).catch(() => null);
  await page.click('.drawer [data-act=export]'); const d = await dl;
  if (d) { await d.saveAs(`${OUT}/backup.json`); L('backup size', fs.statSync(`${OUT}/backup.json`).size) } else L('NO backup download');
  // english + dark
  await page.evaluate(() => { localStorage.setItem('lang', 'en'); localStorage.setItem('theme', 'dark') }); await page.reload(); await sleep(800);
  await page.screenshot({ path: `${OUT}/09-en-dark.png` });
  await page.evaluate(() => { localStorage.setItem('lang', 'el'); localStorage.removeItem('theme') });

  fs.writeFileSync(`${OUT}/log.txt`, log.join('\n') + '\n\nERRORS:\n' + errs.join('\n'));
  console.log('\nERRORS (' + errs.length + '):\n' + errs.join('\n'));
  await ctx.close();
})().catch(e => { console.error('FATAL', e); fs.writeFileSync(`${OUT}/log.txt`, log.join('\n') + '\nFATAL ' + e.stack + '\n' + errs.join('\n')); process.exit(1) });
