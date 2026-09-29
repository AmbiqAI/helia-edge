import { test, expect } from '@playwright/test';

test('API search, filters, empty state and symbol navigation', async ({page})=>{
  await page.goto('reference/');
  const browser=page.getByRole('region',{name:'Search Python API'});
  await expect(browser).toHaveAttribute('data-ready','true');
  await page.getByRole('searchbox',{name:'Search Python API'}).fill('TcnModel');
  await expect(browser.getByRole('link',{name:'TcnModel',exact:true})).toBeVisible();
  await page.getByLabel('Category',{exact:true}).selectOption('Metrics');
  await expect(browser.getByRole('link',{name:'TcnModel',exact:true})).toHaveCount(0);
  await page.getByLabel('Category',{exact:true}).selectOption('Architectures');
  await browser.getByRole('link',{name:'TcnModel',exact:true}).click();
  await expect(page).toHaveURL(/models\/tcn\/#helia_edge.models.tcn.TcnModel$/);
  await expect(page.locator('[id="helia_edge.models.tcn.TcnModel"]')).toBeVisible();
});

test('legacy guide and API URLs resolve',async({page})=>{
  await page.goto('backends/');
  await expect(page).toHaveURL(/getting-started\/backends\/$/);
  await page.goto('api/helia_edge/models/tcn/');
  await expect(page).toHaveURL(/reference\/api\/helia_edge\/models\/tcn\/$/);
});

for (const width of [390,1280]) {
  test(`layouts and themes at ${width}px`,async({page})=>{
    await page.setViewportSize({width,height:900});
    for(const theme of ['light','dark']) {
      await page.addInitScript(value=>localStorage.setItem('starlight-theme',value),theme);
      for (const route of ['', 'reference/', 'guide/', 'examples/', 'getting-started/', 'getting-started/first-model/', 'examples/train-cifar-model/']) {
        await page.goto(route);
        await expect(page.locator('h1')).toHaveCount(1);
        expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1)).toBe(true);
        await page.screenshot({path:`test-results/${width}-${theme}-${route.replaceAll('/','-')||'home'}.png`,fullPage:false});
      }
    }
  });
}

test('site search returns generated API',async({page})=>{
  await page.goto('');
  const results=await page.evaluate(async()=>{
    const path='/helia-edge/pagefind/pagefind.js';
    const pf=await import(/* @vite-ignore */ path);
    const result=await pf.search('TcnModel');
    return Promise.all(result.results.map((r:{data:()=>Promise<{url:string}>})=>r.data()));
  });
  expect(results.some((r:{url:string})=>r.url.includes('models/tcn'))).toBe(true);
});
